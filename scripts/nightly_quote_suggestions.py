"""Suggest exact source quote spans for human review only. No labels are repaired.

CPU/stdlib only. Ingests newline-complete error snapshots while the runner grows
its files. Writes only quote_repair_suggestions.json in the specified private run.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'data/nightly/20261111'
OUTPUT_NAME = 'quote_repair_suggestions.json'
REASONS = ('defect', 'refund', 'delivery', 'question', 'safety', 'misinfo',
           'strong_negative', 'wb_logistics')
PROVENANCE = ('model', 'model_sha256', 'corpus_sha256', 'text_sha256',
              'prompt_sha256', 'generation_sha256')
INVALID_QUOTE_ERROR = 'quote must be a nonempty exact raw substring'
NORMALIZATION = 'NFKC -> casefold -> ё to е -> punctuation/symbol/underscore to space -> whitespace collapse'


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _normalize_with_map(raw):
    """Conservative normalization; each resulting character retains its raw span.

    Combining marks and context-dependent NFKC sequences are unsupported. A
    quote match must also normalize back from a whole raw substring, preventing
    selection of half of a casefold or compatibility-character expansion.
    """
    if not isinstance(raw, str):
        raise ValueError('Text and quotes must be strings')
    if any(unicodedata.category(char).startswith('M') for char in raw):
        raise ValueError('unsupported Unicode combining mark')
    nfkc_parts = [unicodedata.normalize('NFKC', char) for char in raw]
    if ''.join(nfkc_parts) != unicodedata.normalize('NFKC', raw):
        raise ValueError('unsupported contextual Unicode normalization')
    output, positions = [], []
    for index, nfkc in enumerate(nfkc_parts):
        folded = nfkc.casefold().replace('ё', 'е')
        if any(unicodedata.category(char).startswith('M') for char in folded):
            raise ValueError('unsupported Unicode expansion containing marks')
        for char in folded:
            char = char if char.isalnum() else ' '
            if char == ' ':
                if not output:
                    continue
                if output[-1] == ' ':
                    positions[-1] = (positions[-1][0], index + 1)
                else:
                    output.append(' ')
                    positions.append((index, index + 1))
            else:
                output.append(char)
                positions.append((index, index + 1))
    if output and output[-1] == ' ':
        output.pop()
        positions.pop()
    return ''.join(output), positions


def match_quote(text, quote):
    """Return a unique exact raw span or an explicit conservative rejection."""
    try:
        normalized_text, positions = _normalize_with_map(text)
        normalized_quote, _ = _normalize_with_map(quote)
    except ValueError:
        return {'status': 'unsupported_unicode'}
    if not normalized_quote:
        return {'status': 'empty_normalized'}
    spans = set()
    start = 0
    while True:
        index = normalized_text.find(normalized_quote, start)
        if index < 0:
            break
        raw_start, raw_end = positions[index][0], positions[index + len(normalized_quote) - 1][1]
        source = text[raw_start:raw_end]
        # Partial casefold/ligature expansions cannot become invented raw spans.
        if _normalize_with_map(source)[0] == normalized_quote:
            spans.add((raw_start, raw_end))
        start = index + 1
    if len(spans) > 1:
        return {'status': 'ambiguous', 'matching_raw_spans': len(spans)}
    if not spans:
        return {'status': 'no_match'}
    start, end = next(iter(spans))
    return {'status': 'unique', 'start': start, 'end': end,
            'source_quote': text[start:end], 'normalized_quote': normalized_quote}


def _read_corpus(path):
    raw = path.read_bytes()
    rows = {}
    for line_no, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get('id', ''))
        if not key or key in rows or not isinstance(row.get('text'), str):
            raise ValueError(f'Invalid or duplicate corpus record at line {line_no}')
        rows[key] = row['text']
    return rows, _hash(raw)


def _atomic_output(path, payload):
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_suggestions(run_dir=DEFAULT_RUN, output_path=None):
    """Read immutable corpus plus growing errors; overwrite only our own report."""
    run_dir = Path(run_dir).resolve()
    target = run_dir / OUTPUT_NAME
    output = Path(output_path).resolve() if output_path is not None else target.resolve()
    if output != target or (target.exists() and target.is_symlink()):
        raise ValueError('output must be the dedicated quote_repair_suggestions.json file inside this run')
    rows, corpus_hash = _read_corpus(run_dir / 'corpus.jsonl')
    summary = Counter({key: 0 for key in ('error_records', 'invalid_quote_attempts', 'provenance_rejected',
                                         'already_exact_quotes', 'quotes_considered', 'suggestions',
                                         'ambiguous', 'no_match', 'unsupported_unicode', 'empty_normalized',
                                         'malformed_records', 'deferred_tail_bytes')})
    snapshots, suggestions = [], []
    for path in sorted(run_dir.glob('*.errors.jsonl')):
        raw = path.read_bytes()
        complete_end = raw.rfind(b'\n') + 1
        snapshot, tail = raw[:complete_end], raw[complete_end:]
        snapshot_hash = _hash(snapshot)
        summary['deferred_tail_bytes'] += len(tail)
        snapshots.append({'file': path.name, 'complete_bytes': len(snapshot),
                          'complete_sha256': snapshot_hash, 'deferred_tail_bytes': len(tail)})
        for line_no, line in enumerate(snapshot.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode('utf-8-sig'))
            except (ValueError, UnicodeDecodeError):
                summary['malformed_records'] += 1
                continue
            if not isinstance(record, dict):
                summary['malformed_records'] += 1
                continue
            summary['error_records'] += 1
            if INVALID_QUOTE_ERROR not in str(record.get('error', '')):
                continue
            summary['invalid_quote_attempts'] += 1
            key = str(record.get('id', ''))
            text = rows.get(key)
            if (text is None or record.get('corpus_sha256') != corpus_hash
                    or record.get('text_sha256') != _hash(text.encode('utf-8'))
                    or any(not isinstance(record.get(k), str) or not record[k] for k in PROVENANCE)
                    or path.name != record['model'] + '.errors.jsonl'
                    or type(record.get('attempt')) is not int or record['attempt'] < 1):
                summary['provenance_rejected'] += 1
                continue
            parsed = record.get('parsed_label')
            reasons = parsed.get('reasons') if isinstance(parsed, dict) else None
            if not isinstance(reasons, list):
                summary['malformed_records'] += 1
                continue
            for reason_index, reason in enumerate(reasons):
                if (not isinstance(reason, dict) or reason.get('reason') not in REASONS
                        or not isinstance(reason.get('quote'), str)):
                    summary['malformed_records'] += 1
                    continue
                quote = reason['quote']
                if quote and quote in text:
                    summary['already_exact_quotes'] += 1
                    continue
                summary['quotes_considered'] += 1
                match = match_quote(text, quote)
                if match['status'] != 'unique':
                    summary[match['status']] += 1
                    continue
                suggestions.append({
                    **{key: record[key] for key in PROVENANCE}, 'id': key,
                    'attempt': record['attempt'], 'error_at': record.get('at'),
                    'error': record['error'], 'reason': reason['reason'], 'reason_index': reason_index,
                    'raw_quote': quote, 'suggested_source_quote': match['source_quote'],
                    'source_span': {'start': match['start'], 'end': match['end'], 'offset_unit': 'Python Unicode codepoints, end exclusive'},
                    'normalized_quote': match['normalized_quote'],
                    'source_error_file': path.name, 'source_error_line': line_no,
                    'source_error_snapshot_sha256': snapshot_hash,
                    'status': 'suggestions_for_review', 'valid_label': False,
                    'review_required': True})
    summary['suggestions'] = len(suggestions)
    summary['unique_review_ids'] = len({row['id'] for row in suggestions})
    summary['models'] = dict(Counter(row['model'] for row in suggestions))
    payload = {'status': 'suggestions_for_review', 'valid_labels': False, 'silver': False,
               'ground_truth': False, 'created_at': datetime.now(timezone.utc).isoformat(),
               'notice': ('Only normalized surface-form correspondence is suggested. These are NOT valid labels, '
                          'NOT silver, NOT truth. Reason semantics and the full completion remain unvalidated. '
                          'No runner errors, labels, source corpus or manifest are modified.'),
               'normalization': NORMALIZATION,
               'rejections': 'Reject multiple matching raw spans, unsupported Unicode combinations, '
                             'typos/paraphrases, missing provenance and partial expansion matches.',
               'corpus_sha256': corpus_hash, 'error_snapshots': snapshots,
               'script_sha256': _hash(Path(__file__).read_bytes()),
               'summary': dict(summary), 'suggestions': suggestions}
    if _hash((run_dir/'corpus.jsonl').read_bytes()) != corpus_hash:
        raise ValueError('Corpus changed while collecting suggestions; no output saved')
    _atomic_output(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    payload = generate_suggestions(args.run_dir)
    print(json.dumps({'output': str(args.run_dir/OUTPUT_NAME), 'status': payload['status'],
                      'summary': payload['summary']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

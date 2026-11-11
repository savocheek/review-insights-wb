"""Snapshot local nightly labels without inference or promotion to human gold.

python scripts/nightly_report.py --partial
Reads append-only model streams; never repairs, truncates or rewrites inputs.
Raw text and individual model decisions stay under ignored data/nightly/.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('qwen3-14b', 'qwen3.8-27b')
REASONS = ('defect', 'refund', 'delivery', 'question', 'safety', 'misinfo',
           'strong_negative', 'wb_logistics')
SENTIMENTS = ('negative', 'neutral', 'positive', 'mixed')
PROVENANCE = ('model_sha256', 'prompt_sha256', 'generation_sha256', 'corpus_sha256')
DEADLINE = datetime(2026, 11, 11, 6, tzinfo=timezone.utc)
ETA_MIN_SUCCESSES = 100


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold().replace('ё', 'е')
    return ' '.join(re.sub(r'[^\w\s]', ' ', text).split())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_stream(path, partial=True):
    """Read a fixed byte snapshot, preserving incomplete tails in the source."""
    path = Path(path)
    if not path.exists():
        return [], {'missing_file': True, 'partial_tail_lines': 0, 'malformed_lines': 0,
                    'sha256': None, 'bytes': 0}
    raw = path.read_bytes()
    stats = {'missing_file': False, 'partial_tail_lines': 0, 'malformed_lines': 0,
             'sha256': digest(raw), 'bytes': len(raw)}
    records = []
    lines = raw.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line.decode('utf-8-sig'))
            if not isinstance(record, dict):
                raise ValueError('record must be an object')
        except (UnicodeError, ValueError):
            if partial and i == len(lines) - 1 and not line.endswith((b'\n', b'\r')):
                stats['partial_tail_lines'] += 1
            else:
                stats['malformed_lines'] += 1
            continue
        records.append(record)
    return records, stats


def key(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError('IDs and products must be nonempty strings or integers')
    return str(value)


def load_corpus(path):
    rows, stats = read_stream(path, partial=False)
    if not rows or stats['malformed_lines']:
        raise ValueError('Corpus must be a complete nonempty JSONL file')
    seen = {field: set() for field in ('id', 'nmId', 'text')}
    for row in rows:
        row['id'] = key(row.get('id'))
        product = key(row.get('nmId'))
        text = row.get('text')
        if not isinstance(text, str) or not normalize(text):
            raise ValueError('Corpus review text must be nonempty')
        for field, value in [('id', row['id']), ('nmId', product), ('text', normalize(text))]:
            if value in seen[field]:
                raise ValueError(f'Duplicate corpus {field}')
            seen[field].add(value)
        if row.get('text_sha256', digest(text.encode('utf-8'))) != digest(text.encode('utf-8')):
            raise ValueError('Corpus text hash mismatch')
    return rows, stats


def quote_status(quote, text):
    if not isinstance(quote, str) or not quote.strip():
        return 'mismatch'
    if quote in text:
        return 'exact'
    normalized = normalize(quote)
    return 'normalized_only' if normalized and normalized in normalize(text) else 'mismatch'


def valid_label(label):
    if (not isinstance(label, dict)
            or set(label) != {'needs_human', 'reasons', 'sentiment', 'uncertain', 'conflict_flags'}):
        return False
    need = label.get('needs_human')
    if (not isinstance(need, dict) or set(need) != {'need', 'why'} or type(need.get('need')) is not bool
            or not isinstance(need.get('why'), str) or not need['why'].strip() or len(need['why']) > 500
            or label.get('sentiment') not in SENTIMENTS
            or type(label.get('uncertain')) is not bool
            or not isinstance(label.get('conflict_flags'), list)
            or len(label['conflict_flags']) > 8
            or any(not isinstance(flag, str) or not flag.strip() or len(flag) > 120 for flag in label['conflict_flags'])
            or bool(label['conflict_flags']) and not label['uncertain']):
        return False
    reasons = label.get('reasons')
    if not isinstance(reasons, list) or len(reasons) > 8:
        return False
    seen = set()
    for reason in reasons:
        if (not isinstance(reason, dict) or set(reason) != {'reason', 'quote'}
                or reason.get('reason') not in REASONS or reason['reason'] in seen
                or not isinstance(reason.get('quote'), str) or not reason['quote'].strip()):
            return False
        seen.add(reason['reason'])
    return True


def _seconds(record):
    value = record.get('seconds')
    return float(value) if type(value) in (int, float) and math.isfinite(value) and value > 0 else None


def _timestamp(value):
    try:
        date = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return date if date.tzinfo is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _signature(label):
    return (label['needs_human']['need'], frozenset(r['reason'] for r in label['reasons']), label['sentiment'])


def policy_flags(label):
    """Conservative routing hints, never automatic corrections or gold labels."""
    flags = []
    need, reasons, _ = _signature(label)
    if not need and 'safety' in reasons:
        flags.append('possible_policy_conflict_need_false_safety')
    if not need and 'question' in reasons:
        flags.append('possible_policy_conflict_need_false_question')
    if need and not reasons:
        flags.append('possible_policy_conflict_need_true_without_reasons')
    if label['uncertain']:
        flags.append('model_uncertain')
    if label['conflict_flags']:
        flags.append('model_flagged_conflict')
    return flags


def aggregate(run_dir, partial=False):
    run_dir = Path(run_dir)
    corpus, corpus_info = load_corpus(run_dir / 'corpus.jsonl')
    by_id = {r['id']: r for r in corpus}
    model_data, model_stats, private_issues, source_hashes, failed_quote_ids = {}, {}, [], {}, set()
    manifest_path = run_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if not isinstance(manifest, dict):
        raise ValueError('Run manifest must be an object')
    if manifest and manifest.get('corpus_sha256') != corpus_info['sha256']:
        raise ValueError('Run manifest does not match corpus hash')
    specs = manifest.get('models', [])
    if not isinstance(specs, list) or any(not isinstance(s, dict) or s.get('model') not in MODELS for s in specs):
        raise ValueError('Run manifest has unknown model specifications')
    if len({s['model'] for s in specs}) != len(specs):
        raise ValueError('Duplicate model specifications in manifest')
    specs = {s['model']: s for s in specs}
    independent_manifest = (manifest.get('independent_labellers') is True
                            and manifest.get('manual_gold_fewshot') is False
                            and set(specs) == set(MODELS)
                            and specs[MODELS[0]].get('model_sha256') != specs[MODELS[1]].get('model_sha256'))
    runtime, runtime_info = read_stream(run_dir / 'runtime.jsonl', partial=partial)
    now = datetime.now(timezone.utc)
    original_deadline = _timestamp(manifest.get('hard_deadline')) or DEADLINE
    hard_deadline = original_deadline
    for window in runtime:
        if window.get('event') == 'execution_window':
            parsed = _timestamp(window.get('hard_deadline'))
            if parsed is None:
                raise ValueError('Invalid execution window hard deadline')
            hard_deadline = parsed.astimezone(timezone.utc)
    finished_at = max((i for i, event in enumerate(runtime) if event.get('event') == 'run_finished'), default=-1)
    execution_events = {'execution_window', 'progress', 'server_starting', 'server_stopped', 'request_error', 'deadline_stop'}
    owned_run_finished = (finished_at >= 0 and not runtime_info['malformed_lines']
                          and not runtime_info['partial_tail_lines'] and not any(
                              event.get('event') in execution_events for event in runtime[finished_at + 1:]))
    finished_results = runtime[finished_at].get('results', {}) if owned_run_finished else {}
    if not isinstance(finished_results, dict):
        finished_results = {}
    for model in MODELS:
        records, stream = read_stream(run_dir / f'{model}.labels.jsonl', partial=partial)
        errors, error_stream = read_stream(run_dir / f'{model}.errors.jsonl', partial=partial)
        accepted, seen, fingerprints = {}, set(), set()
        quotes, invalid = Counter(), 0
        for record in records:
            item_id = key(record.get('id'))
            if item_id not in by_id:
                raise ValueError(f'Model label ID outside corpus: {model}')
            if item_id in seen:
                raise ValueError(f'Duplicate model success ID: {model}')
            seen.add(item_id)
            text = by_id[item_id]['text']
            hashes_valid = all(isinstance(record.get(k), str) and len(record[k]) == 64
                               and all(c in '0123456789abcdef' for c in record[k]) for k in PROVENANCE)
            if (record.get('model') != model or not hashes_valid
                    or record.get('text_sha256') != digest(text.encode('utf-8'))
                    or record.get('corpus_sha256') != corpus_info['sha256']
                    or model in specs and any(record.get(k) != specs[model].get(k) for k in PROVENANCE)
                    or not valid_label(record.get('label'))):
                invalid += 1
                private_issues.append({'id': item_id, 'model': model, 'issue': 'invalid_schema_or_provenance'})
                continue
            fingerprints.add(tuple(record[k] for k in PROVENANCE))
            label = record['label']
            statuses = [quote_status(r['quote'], text) for r in label['reasons']]
            quotes.update(statuses)
            accepted[item_id] = {**record, 'quote_statuses': statuses,
                                 'exact_quotes': all(s == 'exact' for s in statuses)}
        if len(fingerprints) > 1:
            raise ValueError(f'Mixed model/prompt/generation provenance in stream: {model}')
        failed_quotes = Counter()
        quote_errors_without_completion = 0
        for record in errors:
            if record.get('id') is not None and key(record['id']) not in by_id:
                raise ValueError(f'Model error ID outside corpus: {model}')
            failed_label = record.get('parsed_label', record.get('label'))
            if failed_label is None and isinstance(record.get('raw_content'), str):
                try:
                    failed_label = json.loads(record['raw_content'])
                except ValueError:
                    pass
            if (record.get('id') is not None and isinstance(failed_label, dict)
                    and isinstance(failed_label.get('reasons'), list)):
                item_id = key(record['id'])
                states = [quote_status(reason.get('quote'), by_id[item_id]['text'])
                          for reason in failed_label['reasons'] if isinstance(reason, dict)]
                failed_quotes.update(states)
                if any(s != 'exact' for s in states):
                    failed_quote_ids.add(item_id)
            elif 'quote' in str(record.get('error', '')).lower():
                quote_errors_without_completion += 1
        seconds = [s for r in accepted.values() if (s := _seconds(r)) is not None]
        starts = [t for r in accepted.values() if (t := _timestamp(r.get('started_at'))) is not None]
        finishes = [t for r in accepted.values() if (t := _timestamp(r.get('completed_at'))) is not None]
        missing = len(corpus) - len(accepted)
        error_ids = {key(r['id']) for r in errors if r.get('id') is not None}
        attempted_ids = seen | error_ids
        exact_valid_ids = {item_id for item_id, record in accepted.items() if record['exact_quotes']}
        clean_streams = not (stream['partial_tail_lines'] or error_stream['partial_tail_lines']
                             or stream['malformed_lines'] or error_stream['malformed_lines'])
        model_stats[model] = {
            'successes': len(accepted), 'missing': missing,
            'valid_successes': len(exact_valid_ids), 'valid_missing': len(corpus) - len(exact_valid_ids),
            'attempted_unique': len(attempted_ids), 'unattempted': len(corpus) - len(attempted_ids),
            'quarantined_unique': len(attempted_ids - exact_valid_ids),
            'valid_complete': len(exact_valid_ids) == len(corpus) and clean_streams,
            'success_stream_records': len(records), 'invalid_success_records': invalid,
            'error_attempts': len(errors),
            'ids_with_error_attempts': len({key(r['id']) for r in errors if r.get('id') is not None}),
            'unrecovered_error_ids': len({key(r['id']) for r in errors if r.get('id') is not None} - set(accepted)),
            'partial_tail_lines': stream['partial_tail_lines'] + error_stream['partial_tail_lines'],
            'malformed_lines': stream['malformed_lines'] + error_stream['malformed_lines'],
            'missing_success_file': stream['missing_file'],
            'quotes': {k: quotes[k] for k in ('exact', 'normalized_only', 'mismatch')},
            'failed_attempt_quotes': {k: failed_quotes[k] for k in ('exact', 'normalized_only', 'mismatch')},
            'quote_error_attempts_without_archived_completion': quote_errors_without_completion,
            'reviews_without_exact_quotes': sum(not r['exact_quotes'] for r in accepted.values()),
            'observed_success_seconds': sum(seconds),
            'observed_wall_span_seconds': max(0, (max(finishes) - min(starts)).total_seconds()) if starts and finishes else None,
            'eta_seconds': (sum(seconds) / len(seconds) * missing) if len(seconds) >= ETA_MIN_SUCCESSES and missing else (0 if not missing else None),
            'eta_source': 'successful_request_duration_serial_extrapolation',
            'eta_basis': 'conditional extrapolation of successful request durations only; excludes retries, loading, pauses and future model changes',
        }
        events = [r for r in runtime if r.get('model') == model
                  and r.get('event') in ('progress', 'server_starting', 'server_stopped', 'deadline_stop')]
        state = events[-1]['event'] if events else 'not_observed'
        model_stats[model]['runtime_state'] = state
        resumed = finished_results.get(model)
        resumed_complete = isinstance(resumed, dict) and resumed.get('resumed_complete') is True
        stopped = owned_run_finished and (state == 'server_stopped' or resumed_complete)
        model_stats[model]['processing_stopped'] = stopped
        model_stats[model]['processing_complete'] = stopped and len(attempted_ids) == len(corpus)
        if state in ('server_stopped', 'deadline_stop') and missing:
            model_stats[model]['eta_seconds'] = None
            model_stats[model]['eta_source'] = 'model_stopped_incomplete'
        elif state == 'progress':
            progress = events[-1]
            at = _timestamp(progress.get('at'))
            eta = progress.get('estimated_remaining_seconds')
            if (len(accepted) >= ETA_MIN_SUCCESSES and at is not None and 0 <= (now - at).total_seconds() <= 120
                    and progress.get('successful') == len(accepted) and progress.get('total') == len(corpus)
                    and type(eta) in (int, float) and math.isfinite(eta) and eta >= 0):
                model_stats[model]['eta_seconds'] = eta
                model_stats[model]['eta_source'] = 'fresh_runtime_effective_rate'
                model_stats[model]['runtime_progress_at'] = at.isoformat()
        model_data[model] = accepted
        source_hashes[f'{model}.labels.jsonl'] = stream
        source_hashes[f'{model}.errors.jsonl'] = error_stream

    agreement = Counter()
    silver, queue = [], []
    for row in corpus:
        item_id = row['id']
        pair = [model_data[m].get(item_id) for m in MODELS]
        labels = [r['label'] for r in pair if r is not None]
        hints = []
        if not independent_manifest:
            hints.append('independence_manifest_not_verified')
        if item_id in failed_quote_ids:
            hints.append('failed_attempt_quote_not_exact')
        if len(labels) == 2:
            agreement['paired_valid'] += 1
            a, b = map(_signature, labels)
            for field, left, right in zip(('need', 'reasons', 'sentiment'), a, b):
                agreement[field + '_agree'] += left == right
                if left != right:
                    hints.append(field + '_disagreement')
            agree = a == b
            agreement['all_targets_agree'] += agree
            if pair[0]['prompt_sha256'] != pair[1]['prompt_sha256']:
                hints.append('prompt_provenance_disagreement')
                agree = False
        else:
            agree = False
            hints.append('missing_or_invalid_model_label')
        for record in pair:
            if record is None:
                continue
            label = record['label']
            hints.extend(policy_flags(label))
            if not record['exact_quotes']:
                hints.append('quote_not_exact')
            if 'mismatch' in record['quote_statuses']:
                hints.append('quote_mismatch')
            if not label['needs_human']['need']:
                hints.append('need_false_review')
            if any(r['reason'] == 'safety' for r in label['reasons']):
                hints.append('safety_review')
        if (independent_manifest and agree and all(r['exact_quotes'] for r in pair)
                and not any(policy_flags(label) for label in labels)):
            need, reasons, sentiment = _signature(labels[0])
            silver.append({'id': item_id, 'nmId': row['nmId'], 'text': row['text'],
                           'need': need, 'reasons': [r for r in REASONS if r in reasons],
                           'sentiment': sentiment, 'unsure': False,
                           'source': 'machine_agreement_silver_not_gold',
                           'independent_human_truth': False,
                           'evidence': {m: model_data[m][item_id]['label'] for m in MODELS}})
        if hints:
            hints = sorted(set(hints))
            priority = (100 if any('disagreement' in h for h in hints) else 0)
            priority += 90 if 'quote_mismatch' in hints else 0
            priority += 80 if 'safety_review' in hints else 0
            priority += 70 if any('conflict' in h or 'uncertain' in h for h in hints) else 0
            priority += 60 if 'quote_not_exact' in hints else 0
            priority += 50 if 'missing_or_invalid_model_label' in hints else 0
            queue.append({'id': item_id, 'nmId': row['nmId'], 'text': row['text'],
                          'priority': priority, 'review_flags': hints,
                          'models': {m: model_data[m].get(item_id) for m in MODELS}})
    queue.sort(key=lambda r: (-r['priority'], r['id']))
    valid_complete = all(s['valid_complete'] for s in model_stats.values())
    processing_complete = all(s['processing_complete'] for s in model_stats.values())
    status = ('complete' if valid_complete else 'complete_with_quarantine') if processing_complete else 'partial'
    totals = {k: agreement[k] for k in ('paired_valid', 'need_agree', 'reasons_agree',
                                      'sentiment_agree', 'all_targets_agree')}
    totals.update(accuracy_claim=False, definition='Machine target agreement among schema/provenance-valid pairs; not truth. Quote validity is counted separately.')
    counts = Counter(flag for row in queue for flag in row['review_flags'])
    summary = {'generated_at': now.isoformat().replace('+00:00', 'Z'),
               'hard_deadline_utc': hard_deadline.isoformat().replace('+00:00', 'Z'),
               'original_hard_deadline_utc': original_deadline.isoformat().replace('+00:00', 'Z'),
               'seconds_until_deadline': max(0, (hard_deadline - now).total_seconds()),
               'deadline_reached': now >= hard_deadline,
               'status': status, 'processing_complete': processing_complete,
               'valid_complete': valid_complete, 'owned_run_finished': owned_run_finished,
               'snapshot_mode': 'partial' if partial else 'final',
               'corpus_reviews': len(corpus), 'corpus_unique_ids': len(by_id),
               'corpus_unique_products': len({key(r['nmId']) for r in corpus}),
               'corpus_unique_normalized_texts': len({normalize(r['text']) for r in corpus}),
               'strata': dict(Counter(r.get('selection_group', r.get('stratum'))
                                     if r.get('selection_group', r.get('stratum')) in ('random', 'hard') else 'other'
                                     for r in corpus)),
               'models': model_stats, 'agreement': totals,
               'silver_count': len(silver), 'human_review_queue_count': len(queue),
               'human_review_flags': dict(counts), 'gold_labels_written': 0,
               'independent_benchmark': False, 'machine_agreement_is_truth': False,
               'manifest_declares_independent_labellers': independent_manifest,
               'limitations': ['No accuracy or quality gain can be inferred from machine agreement.',
                               'Silver remains model supervision; human policy and evidence require manual checking.',
                               'Random stratum is sampled from a deduplicated eligible pool with one representative per product, not all original WB reviews.',
                               'Hard stratum uses heuristic difficulty signals, not verified model errors.',
                               'Source isolation covers recorded teacher/probe/demo pools and held-out WB text; other pretraining exposure is unknown.',
                               'ETA is conditional and does not promise completion before a deadline.'],
               'sources': {'corpus.jsonl': corpus_info, **source_hashes,
                           'manifest.json': {'sha256': digest(manifest_path.read_bytes()) if manifest else None}}}
    summary['sources']['runtime.jsonl'] = runtime_info
    return {'summary': summary, 'silver': silver, 'queue': queue, 'issues': private_issues,
            'review_items': {'note': 'Blind human review: text only; decisions start blank.',
                             'items': [{'id': r['id'], 'text': r['text']} for r in queue]}}


def render_public(summary):
    lines = ['# Ночная разметка: состояние и машинное согласие', '',
             f"Снимок UTC: {summary['generated_at']}. Состояние: {summary['status']}.", '',
             f"Текущий предел остановки UTC: {summary['hard_deadline_utc']}; "
             f"исходный предел: {summary['original_hard_deadline_utc']}.", '',
             f"Корпус: {summary['corpus_reviews']} отзывов; уникальных ID, товаров и нормализованных текстов: "
             f"{summary['corpus_unique_ids']}, {summary['corpus_unique_products']}, {summary['corpus_unique_normalized_texts']}.", '',
             f"Части корпуса: случайная — {summary['strata'].get('random', 0)}, "
             f"сложная — {summary['strata'].get('hard', 0)}, другая — {summary['strata'].get('other', 0)}.", '',
             'Статусы: complete — обработка закончена и все результаты прошли проверку; '
             'complete_with_quarantine — обе модели попытались обработать весь корпус и запуск завершён, '
             'но часть результатов не прошла проверку; partial — запуск ещё идёт, завершение не подтверждено '
             'или остались неотправленные отзывы. Невалидные результаты не считаются валидными метками.', '',
             f"Обработка обеими моделями завершена: {'да' if summary['processing_complete'] else 'нет'}. "
             f"Все записи валидны: {'да' if summary['valid_complete'] else 'нет'}.", '',
             '| Модель | Отзывов с попыткой | Валидных с точными цитатами | В карантине | Без попытки |',
             '|---|---:|---:|---:|---:|']
    for model in MODELS:
        m = summary['models'][model]
        lines.append(f"| {model} | {m['attempted_unique']} | {m['valid_successes']} | "
                     f"{m['quarantined_unique']} | {m['unattempted']} |")
    lines += ['', 'Попытки считаются по уникальным ID из файлов результатов и ошибок; повторный запрос '
              'не увеличивает число обработанных отзывов. Карантин здесь — отзывы с попыткой, '
              'но без результата, прошедшего проверку схемы, происхождения и дословности цитат. '
              'Завершение обработки подтверждается событиями остановки серверов и окончания запуска; '
              'одних попыток для этого недостаточно.', '',
             'Это согласие машин, не истинность разметки и не независимая проверка качества. '
             'Случайная часть выбрана из очищенного допустимого пула с одним представителем каждого товара, '
             'а не равномерно из всех исходных отзывов WB. Сложная часть отобрана по эвристикам '
             'неопределённости и высокого балла, а не по известным ошибкам. '
             'Доли не оценивают естественную частоту проблем.', '',
             'Изоляция касается записанных выборок учителя/студента need, разметки, проб, демо и '
             'нормализованных текстов WB val/test. Другие предобучения и дополнительные исторические '
             'данные обучения тональности проверены не полностью.', '',
             '| Модель | Успешных записей | Ещё нет валидной записи | Ошибок запросов | Невалидных записей |',
             '|---|---:|---:|---:|---:|']
    for model in MODELS:
        m = summary['models'][model]
        lines.append(f"| {model} | {m['successes']} | {m['missing']} | {m['error_attempts']} | {m['invalid_success_records']} |")
    lines += ['', 'Успех здесь означает проверенную схему и происхождение записи. Точность цитат проверяется отдельно.', '',
              '| Модель | Точных цитат | Только после нормализации | Не найдены | Повреждённых строк | Незавершённых строк |',
              '|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        m, q = summary['models'][model], summary['models'][model]['quotes']
        lines.append(f"| {model} | {q['exact']} | {q['normalized_only']} | {q['mismatch']} | {m['malformed_lines']} | {m['partial_tail_lines']} |")
    lines += ['', 'Отдельная проверка цитат в архиве неудачных попыток (не прибавляются к успешным):', '',
              '| Модель | Точных | Только после нормализации | Не найдены | Ошибок цитат без доступного ответа |',
              '|---|---:|---:|---:|---:|']
    for model in MODELS:
        m, q = summary['models'][model], summary['models'][model]['failed_attempt_quotes']
        lines.append(f"| {model} | {q['exact']} | {q['normalized_only']} | {q['mismatch']} | "
                     f"{m['quote_error_attempts_without_archived_completion']} |")
    a = summary['agreement']
    lines += ['', f"Пар валидных записей: {a['paired_valid']}. Совпали: need — {a['need_agree']}, "
              f"набор причин — {a['reasons_agree']}, тональность — {a['sentiment_agree']}; "
              f"все три цели — {a['all_targets_agree']}.", '',
              f"Серебряных машинных меток: {summary['silver_count']}. Они не становятся человеческим эталоном. "
              'Допущены только полные согласия с дословными цитатами, без флага сомнения и возможного конфликта с правилами.', '',
              f"В частной очереди ручной проверки: {summary['human_review_queue_count']}. "
              'Тексты и индивидуальные решения моделей находятся только в data/nightly; скрипт не записывает исходные ручные метки и веса.', '',
              '## Время и условная оценка остатка', '']
    for model in MODELS:
        m = summary['models'][model]
        eta = f"{m['eta_seconds'] / 60:.1f} мин" if m['eta_seconds'] is not None else 'недостаточно данных'
        wall = f"{m['observed_wall_span_seconds'] / 60:.1f} мин" if m['observed_wall_span_seconds'] is not None else 'нет завершённых временных отметок'
        lines.append(f"- {model}: сумма успешных запросов {m['observed_success_seconds'] / 60:.1f} мин; "
                     f"наблюдаемый интервал {wall}; условный остаток {eta}.")
    lines += ['', f'Оценка остатка появляется после {ETA_MIN_SUCCESSES} успешных записей модели. '
              'При свежем совпадающем событии прогресса используется текущая фактическая скорость процесса; '
              'иначе — условная последовательная экстраполяция длительности успешных запросов. '
              'Экстраполяция не включает повторы, загрузку, паузы и будущие изменения скорости; это не обещание срока. '
              'Отсутствующая или незаконченная модель не считается завершённой.', '']
    return '\n'.join(lines)


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_outputs(report, run_dir, history_path, public_path):
    run_dir = Path(run_dir)
    for name, value in [('summary.json', report['summary']), ('silver_labels.json', {'labels': report['silver']}),
                        ('human_review_queue.json', {'items': report['queue'], 'issues': report['issues']}),
                        ('review_items.json', report['review_items'])]:
        atomic_write(run_dir / name, json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    public = render_public(report['summary'])
    atomic_write(public_path, public)
    atomic_write(history_path, public)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'data/nightly/20261111')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    allowed = (ROOT / 'data/nightly').resolve()
    if allowed not in run_dir.parents:
        parser.error('--run-dir must be a dedicated directory inside data/nightly')
    # Public/history destinations are fixed and never overlap human labels/model.
    report = aggregate(run_dir, args.partial)
    write_outputs(report, run_dir, ROOT.parent / 'claude-history/NIGHTLY-2026-11-11.txt',
                  ROOT / 'reports/nightly_2026-11-11.md')
    print(json.dumps({'status': report['summary']['status'], 'models': {
        name: {k: r[k] for k in ('attempted_unique', 'valid_successes', 'quarantined_unique', 'unattempted', 'processing_complete')}
        for name, r in report['summary']['models'].items()},
        'processing_complete': report['summary']['processing_complete'],
        'valid_complete': report['summary']['valid_complete'],
        'silver': report['summary']['silver_count'], 'human_review': report['summary']['human_review_queue_count']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

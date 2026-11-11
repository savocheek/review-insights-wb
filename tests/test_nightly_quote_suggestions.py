"""Quote repair is an explicit review aid, never a label repair path."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def helper():
    assert importlib.util.find_spec('scripts.nightly_quote_suggestions') is not None, 'quote suggestions helper missing'
    from scripts import nightly_quote_suggestions
    return nightly_quote_suggestions


def sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def test_case_punctuation_yo_maps_to_exact_raw_span(helper):
    text = 'Всё — СЛОМАЛОСЬ, ещё вчера!'
    found = helper.match_quote(text, 'все сломалось еще')
    assert found['status'] == 'unique'
    assert found['source_quote'] == 'Всё — СЛОМАЛОСЬ, ещё'
    assert text[found['start']:found['end']] == found['source_quote']
    assert helper.match_quote('Товар_сломался вчера', 'ТОВАР сломался')['source_quote'] == 'Товар_сломался'


def test_casefold_and_nfkc_expansions_keep_whole_raw_characters(helper):
    assert helper.match_quote('Straße ﬃ', 'STRASSE FFI')['source_quote'] == 'Straße ﬃ'
    assert helper.match_quote('ﬃ', 'FI')['status'] == 'no_match'
    assert helper.match_quote('ß', 'S')['status'] == 'no_match'
    assert helper.match_quote('ß', 'SS')['source_quote'] == 'ß'


def test_multiple_matches_typos_paraphrases_and_unsupported_combiners_rejected(helper):
    assert helper.match_quote('Ещё раз. ЕЩЕ раз!', 'еще раз')['status'] == 'ambiguous'
    assert helper.match_quote('Товар сломался вчера', 'товар сломалсь')['status'] == 'no_match'
    assert helper.match_quote('Товар сломался вчера', 'Товар неисправен')['status'] == 'no_match'
    assert helper.match_quote('е\u0308жик', 'ЕЖИК')['status'] == 'unsupported_unicode'
    assert helper.match_quote('가', '가')['status'] == 'unsupported_unicode'
    assert helper.match_quote('...', '!!!')['status'] == 'empty_normalized'


def setup_run(tmp_path):
    text = 'Всё — сломалось, ещё вчера!'
    corpus = (json.dumps({'id': '17', 'nmId': 10, 'text': text}, ensure_ascii=False) + '\n').encode('utf-8')
    (tmp_path / 'corpus.jsonl').write_bytes(corpus)
    provenance = {'model': 'qwen3-14b', 'model_sha256': 'weights',
                  'corpus_sha256': hashlib.sha256(corpus).hexdigest(),
                  'text_sha256': sha(text), 'prompt_sha256': 'prompt', 'generation_sha256': 'generation'}
    error = {**provenance, 'id': '17', 'attempt': 1, 'at': '2026-11-11T01:00:00Z',
             'error': 'ValueError: quote must be a nonempty exact raw substring of the review',
             'raw_content': 'original raw response',
             'parsed_label': {'reasons': [{'reason': 'defect', 'quote': 'все сломалось еще'}]}}
    return error


def test_only_invalid_quote_attempts_and_matching_provenance_suggested(helper, tmp_path):
    error = setup_run(tmp_path)
    unrelated = {**error, 'attempt': 2, 'error': 'TimeoutError: request timeout'}
    stale = {**error, 'attempt': 3, 'text_sha256': 'different'}
    exact = {**error, 'attempt': 4, 'parsed_label': {'reasons': [{'reason': 'defect', 'quote': 'сломалось'}]}}
    errors = tmp_path / 'qwen3-14b.errors.jsonl'
    errors.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in (error, unrelated, stale, exact)), encoding='utf-8')
    output = helper.generate_suggestions(tmp_path)
    assert output['status'] == 'suggestions_for_review'
    assert output['valid_labels'] is False and output['silver'] is False and output['ground_truth'] is False
    assert len(output['suggestions']) == 1
    row = output['suggestions'][0]
    assert row['id'] == '17' and row['model'] == 'qwen3-14b' and row['attempt'] == 1
    assert row['reason'] == 'defect'
    assert row['raw_quote'] == 'все сломалось еще'
    assert row['suggested_source_quote'] == 'Всё — сломалось, ещё'
    assert row['model_sha256'] == 'weights' and row['prompt_sha256'] == 'prompt'
    assert row['generation_sha256'] == 'generation' and row['corpus_sha256'] == error['corpus_sha256']
    assert row['status'] == 'suggestions_for_review'
    assert output['summary']['provenance_rejected'] == 1
    assert output['summary']['already_exact_quotes'] == 1


def test_readonly_inputs_rerunnable_output_isolated_and_torn_tail_deferred(helper, tmp_path):
    error = setup_run(tmp_path)
    errors = tmp_path / 'qwen3-14b.errors.jsonl'
    errors.write_bytes((json.dumps(error, ensure_ascii=False) + '\n').encode('utf-8') + b'{"unfinished":"\xd0')
    (tmp_path / 'manifest.json').write_text('{"protected":true}', encoding='utf-8')
    (tmp_path / 'qwen3-14b.labels.jsonl').write_text('{"untouched":true}\n', encoding='utf-8')
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    first = helper.generate_suggestions(tmp_path)
    second = helper.generate_suggestions(tmp_path)
    assert first['suggestions'] == second['suggestions']
    assert first['summary']['deferred_tail_bytes'] > 0
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert (tmp_path / 'quote_repair_suggestions.json').is_file()
    for target in (tmp_path/'corpus.jsonl', tmp_path/'manifest.json', tmp_path/'elsewhere.json'):
        with pytest.raises(ValueError, match='output'):
            helper.generate_suggestions(tmp_path, output_path=target)
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert not list(tmp_path.glob('*.tmp'))

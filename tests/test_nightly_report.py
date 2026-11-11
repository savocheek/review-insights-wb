"""Fixture-only nightly machine agreement and private/public report contracts."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module():
    spec = importlib.util.find_spec('scripts.nightly_report')
    assert spec is not None, 'nightly report aggregator is missing'
    from scripts import nightly_report
    return nightly_report


def sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def fixture(tmp_path):
    corpus = [{'id': str(i), 'nmId': i, 'text': f'Личный секрет {i}: Сломался товар',
               'stratum': 'random' if i % 2 else 'hard'} for i in range(1, 8)]
    raw = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in corpus)
    (tmp_path / 'corpus.jsonl').write_text(raw, encoding='utf-8')
    provenance = hashlib.sha256((tmp_path / 'corpus.jsonl').read_bytes()).hexdigest()
    records = []
    for row in corpus:
        records.append({'id': row['id'], 'model': 'qwen3-14b',
            'text_sha256': sha(row['text']), 'corpus_sha256': provenance,
            'model_sha256': 'a' * 64, 'prompt_sha256': 'b' * 64,
            'generation_sha256': 'c' * 64,
            'label': {'reasons': [{'reason': 'defect', 'quote': 'Сломался товар'}],
                      'needs_human': {'need': False, 'why': 'мелкая проблема'},
                      'sentiment': 'negative', 'uncertain': False, 'conflict_flags': []},
            'seconds': 2, 'started_at': '2026-11-11T01:00:00Z',
            'completed_at': '2026-11-11T01:00:02Z'})
    other = copy.deepcopy(records)
    for r in other:
        r['model'] = 'qwen3.8-27b'
        r['model_sha256'] = 'd' * 64
    # 1 silver; 2 normalized-only quote; 3 hallucinated quote; 4 disagreement;
    # 5 explicit uncertainty; 6 agreement but policy conflict; 7 absent peer.
    other[1]['label']['reasons'][0]['quote'] = 'сломался  товар'
    other[2]['label']['reasons'][0]['quote'] = 'Несуществующая цитата'
    other[3]['label']['needs_human']['need'] = True
    other[4]['label']['uncertain'] = True
    for r in [records[5], other[5]]:
        r['label']['reasons'][0]['reason'] = 'safety'
    other = other[:-1]
    for name, rows in [('qwen3-14b', records), ('qwen3.8-27b', other)]:
        (tmp_path / f'{name}.labels.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'corpus_sha256': provenance, 'independent_labellers': True, 'manual_gold_fewshot': False,
        'models': [{k: rows[0][k] for k in ('model', 'model_sha256', 'prompt_sha256',
                    'generation_sha256', 'corpus_sha256')} for rows in (records, other)]}), encoding='utf-8')
    return corpus, records, other


def test_silver_requires_exact_valid_agreement_and_never_becomes_gold(tmp_path):
    corpus, _, _ = fixture(tmp_path)
    report = module().aggregate(tmp_path, partial=True)
    assert report['summary']['status'] == 'partial'
    assert report['summary']['models']['qwen3-14b']['successes'] == 7
    assert report['summary']['models']['qwen3.8-27b']['successes'] == 6
    assert report['summary']['models']['qwen3.8-27b']['missing'] == 1
    quotes = report['summary']['models']['qwen3.8-27b']['quotes']
    assert quotes['normalized_only'] == 1 and quotes['mismatch'] == 1
    assert [r['id'] for r in report['silver']] == ['1']
    assert report['silver'][0]['source'] == 'machine_agreement_silver_not_gold'
    assert report['summary']['agreement']['paired_valid'] == 6
    assert report['summary']['agreement']['need_agree'] == 5
    assert report['summary']['agreement']['accuracy_claim'] is False
    assert {r['id'] for r in report['queue']} == {r['id'] for r in corpus}
    for item in report['review_items']['items']:
        assert set(item) == {'id', 'text'}
    assert report['summary']['gold_labels_written'] == 0


def test_duplicate_corpus_products_texts_ids_and_outside_labels_fail_closed(tmp_path):
    corpus, records, _ = fixture(tmp_path)
    original = (tmp_path / 'corpus.jsonl').read_bytes()
    for field, value in [('id', corpus[0]['id']), ('nmId', corpus[0]['nmId']),
                         ('text', '  ЛИЧНЫЙ СЕКРЕТ 1:  Сломался товар  ')]:
        bad = copy.deepcopy(corpus)
        bad[1][field] = value
        (tmp_path / 'corpus.jsonl').write_text('\n'.join(json.dumps(r) for r in bad), encoding='utf-8')
        with pytest.raises(ValueError, match='Duplicate'):
            module().aggregate(tmp_path, partial=True)
    (tmp_path / 'corpus.jsonl').write_bytes(original)
    records[0]['id'] = 'outside'
    (tmp_path / 'qwen3-14b.labels.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    with pytest.raises(ValueError, match='outside corpus'):
        module().aggregate(tmp_path, partial=True)


def test_missing_streams_partial_tail_and_stale_source_are_honest(tmp_path):
    _, records, _ = fixture(tmp_path)
    records[0]['text_sha256'] = 'f' * 64
    path = tmp_path / 'qwen3-14b.labels.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in records) + '\n{"id":', encoding='utf-8')
    peer = tmp_path / 'qwen3.8-27b.labels.jsonl'
    peer.unlink()
    (tmp_path / 'qwen3-14b.errors.jsonl').write_text(json.dumps(
        {'id': '1', 'error': 'PRIVATE ERROR TEXT', 'model': 'qwen3-14b', 'attempt': 1}) + '\n')
    report = module().aggregate(tmp_path, partial=True)
    stats = report['summary']['models']['qwen3-14b']
    assert stats['successes'] == 6 and stats['invalid_success_records'] == 1
    assert stats['partial_tail_lines'] == 1 and stats['error_attempts'] == 1
    assert report['summary']['models']['qwen3.8-27b']['successes'] == 0
    assert report['summary']['models']['qwen3.8-27b']['eta_seconds'] is None
    assert report['silver'] == []
    assert path.read_text(encoding='utf-8').endswith('{"id":')


def test_public_report_aggregates_only_and_atomic_regeneration_is_safe(tmp_path):
    corpus, _, _ = fixture(tmp_path)
    mod = module()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    report = mod.aggregate(tmp_path, partial=True)
    history = tmp_path / 'history' / 'NIGHTLY-2026-11-11.txt'
    public = tmp_path / 'reports' / 'nightly_2026-11-11.md'
    mod.write_outputs(report, tmp_path, history, public)
    mod.write_outputs(report, tmp_path, history, public)
    for path in [history, public]:
        text = path.read_text(encoding='utf-8')
        assert 'Личный секрет' not in text
        assert 'Сломался товар' not in text
        assert 'не истинность' in text
    assert all((tmp_path / name).read_bytes() == raw for name, raw in before.items())
    blind = json.loads((tmp_path / 'review_items.json').read_text(encoding='utf-8'))
    assert blind['items'] and set(blind['items'][0]) == {'id', 'text'}
    assert set(r['id'] for r in blind['items']) <= set(r['id'] for r in corpus)
    assert not list(tmp_path.rglob('*.tmp'))


def test_missing_or_wrong_manifest_prevents_silver_and_independent_claim(tmp_path):
    fixture(tmp_path)
    (tmp_path / 'manifest.json').unlink()
    report = module().aggregate(tmp_path, partial=True)
    assert report['silver'] == []
    assert report['summary']['manifest_declares_independent_labellers'] is False


def test_failed_completion_quotes_are_audited_separately_from_successes(tmp_path):
    _, records, _ = fixture(tmp_path)
    error = {**records[0], 'error': 'quote must be an exact substring',
             'parsed_label': copy.deepcopy(records[0]['label'])}
    error.pop('label')
    error['parsed_label']['reasons'][0]['quote'] = 'сломался  товар'
    (tmp_path / 'qwen3-14b.errors.jsonl').write_text(json.dumps(error) + '\n', encoding='utf-8')
    report = module().aggregate(tmp_path, partial=True)
    stats = report['summary']['models']['qwen3-14b']
    assert stats['successes'] == 7 and stats['error_attempts'] == 1
    assert stats['failed_attempt_quotes']['normalized_only'] == 1
    assert stats['quotes']['normalized_only'] == 0
    assert 'failed_attempt_quote_not_exact' in next(r for r in report['queue'] if r['id'] == '1')['review_flags']


def test_duplicate_successes_and_mixed_generation_fail_closed(tmp_path):
    _, records, _ = fixture(tmp_path)
    path = tmp_path / 'qwen3-14b.labels.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in records + [records[0]]), encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate model success'):
        module().aggregate(tmp_path, partial=True)
    records[0]['generation_sha256'] = 'e' * 64
    path.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
    report = module().aggregate(tmp_path, partial=True)
    assert report['summary']['models']['qwen3-14b']['invalid_success_records'] == 1
    assert '1' not in {r['id'] for r in report['silver']}


def test_fresh_runtime_rate_drives_eta_and_stopped_model_has_no_live_eta(tmp_path, monkeypatch):
    fixture(tmp_path)
    progress = {'at': datetime.now(timezone.utc).isoformat(), 'event': 'progress',
                'model': 'qwen3.8-27b', 'successful': 6, 'total': 7,
                'active': 1, 'reviews_per_minute': 3, 'estimated_remaining_seconds': 20}
    runtime = tmp_path / 'runtime.jsonl'
    runtime.write_text(json.dumps(progress) + '\n', encoding='utf-8')
    # Production deliberately suppresses warmup ETA until 100 successful rows.
    assert module().aggregate(tmp_path, partial=True)['summary']['models']['qwen3.8-27b']['eta_seconds'] is None
    monkeypatch.setattr(module(), 'ETA_MIN_SUCCESSES', 3)
    stats = module().aggregate(tmp_path, partial=True)['summary']['models']['qwen3.8-27b']
    assert stats['eta_seconds'] == 20 and stats['eta_source'] == 'fresh_runtime_effective_rate'
    with runtime.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'at': progress['at'], 'event': 'server_stopped', 'model': progress['model']}) + '\n')
    stats = module().aggregate(tmp_path, partial=True)['summary']['models']['qwen3.8-27b']
    assert stats['eta_seconds'] is None and stats['runtime_state'] == 'server_stopped'


def test_finished_processing_with_invalid_quotes_is_complete_with_quarantine(tmp_path):
    corpus, records, _ = fixture(tmp_path)
    # The final peer row was attempted twice and rejected, not successfully labelled.
    error = {'id': corpus[-1]['id'], 'model': 'qwen3.8-27b', 'error': 'nonexact quote', 'attempt': 1}
    (tmp_path / 'qwen3.8-27b.errors.jsonl').write_text(
        json.dumps(error) + '\n' + json.dumps({**error, 'attempt': 2}) + '\n', encoding='utf-8')
    report = module().aggregate(tmp_path, partial=True)
    stats = report['summary']['models']['qwen3.8-27b']
    assert stats['attempted_unique'] == 7 and stats['unattempted'] == 0
    assert stats['valid_successes'] == 4 and stats['quarantined_unique'] == 3
    assert not stats['processing_complete'] and report['summary']['status'] == 'partial'
    events = [{'event': 'server_stopped', 'model': model} for model in module().MODELS]
    events.append({'event': 'run_finished', 'results': {}})
    (tmp_path / 'runtime.jsonl').write_text('\n'.join(json.dumps(e) for e in events), encoding='utf-8')
    report = module().aggregate(tmp_path, partial=True)
    assert report['summary']['status'] == 'complete_with_quarantine'
    assert report['summary']['processing_complete'] is True
    assert report['summary']['valid_complete'] is False
    assert all(s['processing_complete'] for s in report['summary']['models'].values())
    assert report['summary']['models']['qwen3.8-27b']['error_attempts'] == 2
    assert report['summary']['gold_labels_written'] == 0


def test_valid_complete_still_running_or_unattempted_is_partial(tmp_path):
    _, records, _ = fixture(tmp_path)
    peer = copy.deepcopy(records)
    for r in peer:
        r['model'] = 'qwen3.8-27b'
        r['model_sha256'] = 'd' * 64
    (tmp_path / 'qwen3.8-27b.labels.jsonl').write_text('\n'.join(json.dumps(r) for r in peer), encoding='utf-8')
    report = module().aggregate(tmp_path, partial=True)
    assert report['summary']['valid_complete'] is True
    assert report['summary']['processing_complete'] is False
    assert report['summary']['status'] == 'partial'
    runtime = tmp_path / 'runtime.jsonl'
    events = [{'event': 'server_stopped', 'model': model} for model in module().MODELS]
    events.append({'event': 'run_finished', 'results': {}})
    runtime.write_text('\n'.join(json.dumps(e) for e in events) + '\n', encoding='utf-8')
    assert module().aggregate(tmp_path, partial=True)['summary']['status'] == 'complete'
    # A later execution invalidates an old run_finished marker.
    with runtime.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'event': 'server_starting', 'model': 'qwen3.8-27b'}) + '\n')
    assert module().aggregate(tmp_path, partial=True)['summary']['status'] == 'partial'
    # Missing a review is partial even if both servers stopped cleanly.
    runtime.write_text('\n'.join(json.dumps(e) for e in events) + '\n', encoding='utf-8')
    (tmp_path / 'qwen3.8-27b.labels.jsonl').write_text('\n'.join(json.dumps(r) for r in peer[:-1]), encoding='utf-8')
    report = module().aggregate(tmp_path, partial=True)
    assert report['summary']['status'] == 'partial'
    assert report['summary']['models']['qwen3.8-27b']['unattempted'] == 1


def test_resumed_execution_window_updates_deadline_and_invalidates_old_finish(tmp_path):
    fixture(tmp_path)
    runtime = tmp_path / 'runtime.jsonl'
    events = [{'event': 'run_finished', 'results': {}},
              {'event': 'execution_window', 'deadline': '2026-11-11T11:50:00Z',
               'hard_deadline': '2026-11-11T12:00:00Z',
               'original_manifest_deadline': '2026-11-11T06:00:00Z'}]
    runtime.write_text('\n'.join(json.dumps(e) for e in events) + '\n', encoding='utf-8')
    summary = module().aggregate(tmp_path, partial=True)['summary']
    assert summary['hard_deadline_utc'] == '2026-11-11T12:00:00Z'
    assert summary['original_hard_deadline_utc'] == '2026-11-11T06:00:00Z'
    assert summary['owned_run_finished'] is False
    assert '2026-11-11T12:00:00Z' in module().render_public(summary)

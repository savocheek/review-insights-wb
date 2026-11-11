import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module():
    assert importlib.util.find_spec('nightly_labeling') is not None, 'nightly runner missing'
    import nightly_labeling
    return nightly_labeling


def label():
    return {'reasons': [{'reason': 'defect', 'quote': 'сломался'}],
            'needs_human': {'need': False, 'why': 'Факт проблемы без открытого запроса'},
            'sentiment': 'negative', 'uncertain': True, 'conflict_flags': ['borderline_need']}


def provenance(mod):
    return {'model': 'qwen3-14b', 'model_sha256': 'm', 'prompt_sha256': mod.sha_text(mod.PROMPT),
            'generation_sha256': mod.generation_hash(), 'corpus_sha256': 'c'}


def test_exact_quotes_and_independent_reason_need_targets():
    mod = module()
    text = 'Товар сломался вчера.'
    assert mod.validate_label(label(), text) == label()
    wrong = label()
    wrong['reasons'][0]['quote'] = 'товар сломался'
    with pytest.raises(ValueError, match='quote'):
        mod.validate_label(wrong, text)
    wrong = label()
    wrong['needs_human']['need'] = 'false'
    with pytest.raises(ValueError):
        mod.validate_label(wrong, text)
    wrong = label()
    wrong['reasons'][0]['reason'] = 'unknown'
    with pytest.raises(ValueError):
        mod.validate_label(wrong, text)


def test_both_requests_are_blind_same_prompt_and_disable_thinking():
    mod = module()
    row = {'id': 'x', 'text': 'Товар сломался вчера.', 'label': {'secret': 'GOLD_NEVER_SEND'},
           'first_model_label': 'OTHER_MODEL_NEVER_SEND'}
    a = mod.build_request(row, 'qwen3-14b')
    b = mod.build_request(row, 'qwen3.8-27b')
    assert a['messages'] == b['messages']
    encoded = json.dumps(a, ensure_ascii=False)
    assert 'GOLD_NEVER_SEND' not in encoded and 'OTHER_MODEL_NEVER_SEND' not in encoded
    assert a['chat_template_kwargs'] == {'enable_thinking': False}
    assert a['stream'] is False and a['max_tokens'] <= 700
    assert '/no_think' in a['messages'][1]['content']
    assert a['response_format']['json_schema']['schema'] == mod.LABEL_SCHEMA


def test_valid_completion_durable_resume_does_not_request_again(tmp_path):
    mod = module()
    now = datetime(2026, 11, 11, 1, tzinfo=timezone.utc)
    row = {'id': 'x', 'text': 'Товар сломался вчера.'}
    calls = []
    def request(body, timeout):
        calls.append(body)
        return {'choices': [{'message': {'content': json.dumps(label(), ensure_ascii=False)}}], 'usage': {'completion_tokens': 50}}
    kwargs = dict(rows=[row], run_dir=tmp_path, provenance=provenance(mod),
                  deadline=now + timedelta(seconds=60), now_fn=lambda: now, workers=1)
    result = mod.label_stage(request_fn=request, **kwargs)
    assert result['successful'] == 1
    saved = tmp_path / 'qwen3-14b.labels.jsonl'
    before = saved.read_bytes()
    record = json.loads(before)
    assert record['label'] == label() and record['text_sha256'] == mod.sha_text(row['text'])
    result = mod.label_stage(request_fn=lambda *args: pytest.fail('resumed successful row'), **kwargs)
    assert result['successful'] == 1 and len(calls) == 1
    assert saved.read_bytes() == before


def test_invalid_quote_retries_twice_and_errors_stay_separate(tmp_path):
    mod = module()
    now = datetime(2026, 11, 11, 1, tzinfo=timezone.utc)
    calls = []
    bad = label()
    bad['reasons'][0]['quote'] = 'fabrication'
    def request(body, timeout):
        calls.append(timeout)
        return {'choices': [{'message': {'content': json.dumps(bad)}}]}
    result = mod.label_stage([{'id': 'x', 'text': 'Товар сломался вчера.'}], tmp_path, provenance(mod),
                             now + timedelta(seconds=60), request_fn=request, now_fn=lambda: now, workers=1)
    assert result['successful'] == 0 and len(calls) == 2
    assert len((tmp_path / 'qwen3-14b.errors.jsonl').read_text(encoding='utf8').splitlines()) == 2
    error = json.loads((tmp_path / 'qwen3-14b.errors.jsonl').read_text(encoding='utf8').splitlines()[0])
    assert 'fabrication' in error['raw_content']
    assert error['parsed_label'] == bad
    assert error['text_sha256'] == mod.sha_text('Товар сломался вчера.')
    assert error['corpus_sha256'] == 'c'
    assert not (tmp_path / 'qwen3-14b.labels.jsonl').exists()
    assert all(0 < timeout <= 52 for timeout in calls)


def test_deadline_prevents_any_request_and_server_process_is_owned(tmp_path):
    mod = module()
    now = datetime(2026, 11, 11, 6, tzinfo=timezone.utc)
    result = mod.label_stage([{'id': 'x', 'text': 'x'}], tmp_path, provenance(mod), now,
        request_fn=lambda *args: pytest.fail('request after deadline'), now_fn=lambda: now, workers=1)
    assert result['successful'] == 0 and result['deadline_reached'] is True
    calls = []
    class Process:
        pid = 123
        def poll(self): return None
        def terminate(self): calls.append('terminate_owned_123')
        def wait(self, timeout): calls.append(('wait', timeout)); return 0
        def kill(self): calls.append('kill_owned_123')
    server = mod.OwnedServer(Path('server.exe'), Path('model.gguf'), tmp_path / 'server.log', now,
                             now_fn=lambda: now)
    server.process = Process()
    server.stop()
    assert calls == ['terminate_owned_123', ('wait', 3)]


def test_resume_rejects_source_changes_and_repairs_only_torn_tail(tmp_path):
    mod = module()
    row = {'id': 'x', 'text': 'Товар сломался вчера.'}
    record = {**provenance(mod), 'id': 'x', 'text_sha256': mod.sha_text(row['text']), 'label': label()}
    path = tmp_path / 'qwen3-14b.labels.jsonl'
    path.write_text(json.dumps(record, ensure_ascii=False) + '\n{"partial":', encoding='utf8')
    successes = mod.load_successes(path, [row], provenance(mod))
    assert set(successes) == {'x'}
    assert path.read_text(encoding='utf8').endswith('\n')
    changed = dict(provenance(mod), corpus_sha256='changed')
    with pytest.raises(ValueError, match='provenance'):
        mod.load_successes(path, [row], changed)

def test_two_owned_servers_sequential_and_same_independent_input(tmp_path):
    mod = module()
    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text(json.dumps({'id': 'x', 'text': 'Товар сломался вчера.'}, ensure_ascii=False) + '\n', encoding='utf8')
    models = []
    for name in ['qwen3-14b', 'qwen3.8-27b']:
        path = tmp_path / f'{name}.gguf'
        path.write_bytes(b'fake model for orchestration test')
        models.append((name, path))
    exe = tmp_path / 'server.exe'
    exe.write_bytes(b'fake executable never spawned')
    trace = []
    class FakeServer:
        def __init__(self, executable, model, log, deadline, model_id, **kw):
            self.name = model_id
        def __enter__(self):
            trace.append(('start', self.name))
            return self
        def __exit__(self, *args):
            trace.append(('stop', self.name))
    payloads = []
    def request(body, timeout):
        payloads.append(body)
        return {'choices': [{'message': {'content': json.dumps(label(), ensure_ascii=False)}, 'finish_reason': 'stop'}]}
    now = datetime(2026, 11, 11, 1, tzinfo=timezone.utc)
    before = corpus.read_bytes()
    result = mod.execute(corpus, tmp_path, server_factory=FakeServer, request_fn=request,
                         now_fn=lambda: now, models=models, executable=exe, workers=1)
    assert trace == [('start', 'qwen3-14b'), ('stop', 'qwen3-14b'),
                     ('start', 'qwen3.8-27b'), ('stop', 'qwen3.8-27b')]
    assert payloads[0]['messages'] == payloads[1]['messages']
    assert all(value['successful'] == 1 for value in result.values())
    assert corpus.read_bytes() == before
    corpus.write_text(json.dumps({'id': 'changed', 'text': 'new input'}) + '\n')
    with pytest.raises(ValueError, match='manifest differs'):
        mod.prepare(corpus, tmp_path, models=models, executable=exe)


def test_empty_quotes_and_context_overflow_never_reach_completion(monkeypatch):
    mod = module()
    bad = label()
    bad['reasons'][0]['quote'] = ''
    with pytest.raises(ValueError, match='nonempty exact'):
        mod.validate_label(bad, 'Товар сломался вчера.')
    calls = []
    def fake_http(url, payload, timeout):
        calls.append(url)
        return {'tokens': [1] * 4000}
    monkeypatch.setattr(mod, 'http_json', fake_http)
    with pytest.raises(ValueError, match='4096-token'):
        mod.request_completion('http://localhost:18080', mod.build_request({'text': 'x'}, 'm'), 10)
    assert calls == ['http://localhost:18080/tokenize']

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nightly_labeling as mod


def test_explicit_new_window_preserves_original_manifest_and_resume(tmp_path):
    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text(json.dumps({'id':'one','text':'Хороший товар'})+'\n',encoding='utf-8')
    exe = tmp_path / 'server.exe'
    exe.write_bytes(b'fixture')
    weights = tmp_path / 'fixture.gguf'
    weights.write_bytes(b'model')
    models = [('qwen3-14b',weights)]
    mod.prepare(corpus,tmp_path,models,exe)
    original = (tmp_path/'manifest.json').read_bytes()
    trace = []
    class Server:
        def __init__(self,*args,**kwargs): trace.append(args[3])
        def __enter__(self): return self
        def __exit__(self,*args): pass
    def request(body,timeout):
        return {'choices':[{'message':{'content':json.dumps({'reasons':[],
            'needs_human':{'need':False,'why':'Положительное впечатление'},
            'sentiment':'positive','uncertain':False,'conflict_flags':[]})}}]}
    now = datetime(2026,11,11,8,tzinfo=timezone.utc)
    cutoff = datetime(2026,11,11,11,50,tzinfo=timezone.utc)
    hard = datetime(2026,11,11,12,tzinfo=timezone.utc)
    kwargs = dict(models=models,executable=exe,server_factory=Server,request_fn=request,
                  now_fn=lambda:now,workers=1)
    with pytest.raises(ValueError):
        mod.execute(corpus,tmp_path,deadline=cutoff,**kwargs)
    result = mod.execute(corpus,tmp_path,deadline=cutoff,hard_deadline=hard,**kwargs)
    assert result['qwen3-14b']['successful']==1
    assert trace==[cutoff]
    assert (tmp_path/'manifest.json').read_bytes()==original
    saved=(tmp_path/'qwen3-14b.labels.jsonl').read_bytes()
    result=mod.execute(corpus,tmp_path,deadline=cutoff,hard_deadline=hard,**kwargs)
    assert result['qwen3-14b']['resumed_complete']
    assert (tmp_path/'qwen3-14b.labels.jsonl').read_bytes()==saved
    events=[json.loads(x) for x in (tmp_path/'runtime.jsonl').read_text().splitlines()]
    windows=[x for x in events if x['event']=='execution_window']
    assert windows[-1]['hard_deadline']==mod.stamp(hard)


def test_new_window_must_still_bound_requests(tmp_path):
    now=datetime(2026,11,11,8,tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        mod.execute(tmp_path/'missing',tmp_path,deadline=datetime(2026,11,11,13,tzinfo=timezone.utc),
                    hard_deadline=datetime(2026,11,11,12,tzinfo=timezone.utc),now_fn=lambda:now)

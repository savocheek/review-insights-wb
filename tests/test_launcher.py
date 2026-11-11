import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def launcher():
    assert importlib.util.find_spec('run') is not None, 'one-command launcher missing'
    import run
    return run


def test_model_readiness_requires_encoder_tokenizer_config_and_heads(tmp_path):
    mod = launcher()
    errors = mod.check_model(tmp_path)
    assert any('config.json' in e for e in errors)
    assert any('tokenizer' in e for e in errors)
    assert any('heads.pt' in e for e in errors)
    assert any('model.safetensors' in e for e in errors)
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'bert', 'hidden_size': 8}))
    (tmp_path / 'tokenizer.json').write_text('{}')
    (tmp_path / 'tokenizer_config.json').write_text('{}')
    (tmp_path / 'heads.pt').write_bytes(b'heads')
    (tmp_path / 'model.safetensors').write_bytes(b'weights')
    assert mod.check_model(tmp_path) == []
    (tmp_path / 'config.json').write_text('broken')
    assert any('config.json' in e for e in mod.check_model(tmp_path))


def test_annotation_check_needs_no_model_and_missing_dependencies_are_actionable(monkeypatch, tmp_path):
    mod = launcher()
    monkeypatch.setattr(mod.importlib.util, 'find_spec', lambda name: None if name == 'torch' else object())
    monkeypatch.setattr(mod, 'ROOT', ROOT)
    annotation = mod.preflight('annotate', model=tmp_path / 'missing')
    assert annotation['ok']
    csv = mod.preflight('csv', model=tmp_path / 'missing')
    assert not csv['ok']
    assert any('torch' in error and 'requirements.txt' in error for error in csv['errors'])


def test_check_never_imports_or_launches_apps(monkeypatch, capsys):
    mod = launcher()
    monkeypatch.setattr(mod, 'preflight', lambda app, model=None: {'ok': True, 'errors': [], 'app': app})
    monkeypatch.setattr(mod.importlib, 'import_module', lambda name: pytest.fail('check imported UI'))
    assert mod.main(['--check']) == 0
    assert '127.0.0.1:7865' in capsys.readouterr().out


def test_launcher_uses_requested_app_and_loopback_without_sharing(monkeypatch):
    mod = launcher()
    calls = []
    class App:
        def launch(self, **kwargs):
            calls.append(kwargs)
    class Module:
        @staticmethod
        def build_app():
            return App()
    monkeypatch.setattr(mod, 'preflight', lambda app, model=None: {'ok': True, 'errors': [], 'app': app})
    monkeypatch.setattr(mod.importlib, 'import_module', lambda name: calls.append(name) or Module)
    assert mod.main(['--app', 'annotate']) == 0
    assert calls == ['needs_reply_label', {'server_name': '127.0.0.1', 'server_port': 7864, 'share': False}]
    calls.clear()
    assert mod.main(['--port', '7890']) == 0
    assert calls == ['csv_app', {'server_name': '127.0.0.1', 'server_port': 7890, 'share': False}]


def test_launcher_refuses_launch_if_checks_fail(monkeypatch):
    mod = launcher()
    monkeypatch.setattr(mod, 'preflight', lambda app, model=None: {'ok': False, 'errors': ['missing torch'], 'app': app})
    monkeypatch.setattr(mod.importlib, 'import_module', lambda name: pytest.fail('launched without requirements'))
    assert mod.main([]) == 1

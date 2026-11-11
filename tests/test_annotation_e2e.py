"""Real localhost HTTP/API E2E; no browser rendering or source labels are used."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from gradio_client import Client
import pytest

ROOT = Path(__file__).resolve().parents[1]
MALICIOUS = '<script>alert("review")</script> **literal** <img src=x onerror=alert(1)>'
NO = 'Хватит шаблона'


@contextmanager
def live_app(items, labels, log_path):
    """Use the production command entry point; always reap the actual server."""
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    env = dict(os.environ, GRADIO_ANALYTICS_ENABLED='False', HF_HUB_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', NO_PROXY='127.0.0.1,localhost',
               PYTHONDONTWRITEBYTECODE='1')
    base = f'http://127.0.0.1:{port}/'
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            [sys.executable, '-B', '-u', str(ROOT / 'needs_reply_label.py'),
             '--items', str(items), '--labels', str(labels), '--port', str(port)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 35
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            last_error = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(log_path.read_text(encoding='utf-8'))
                try:
                    with opener.open(base + 'config', timeout=.5) as response:
                        config = json.load(response)
                    break
                except (OSError, ValueError) as exc:
                    last_error = exc
                    time.sleep(.1)
            else:
                raise TimeoutError(f'Local annotation app did not become ready: {last_error}\n'
                                   + log_path.read_text(encoding='utf-8'))
            client = Client(base, verbose=False, analytics_enabled=False,
                            download_files=False, httpx_kwargs={'trust_env': False})
            try:
                yield client, config
            finally:
                client.close()
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def test_local_http_annotation_save_restart_and_error_recovery(tmp_path):
    items = tmp_path / 'items.json'
    labels = tmp_path / 'labels.json'
    items.write_text(json.dumps({'items': [
        {'id': 'one', 'text': MALICIOUS, 'draft': {'need': True},
         'answer': 'HIDDEN_SELLER_ANSWER', 'stars': 1},
        {'id': 'two', 'text': 'second review'},
    ]}), encoding='utf-8')
    original = json.dumps({'note': 'preserve metadata', 'labels': [
        {'id': 'outside-batch', 'need': True, 'reasons': [], 'unsure': False},
    ]}).encode('utf-8')
    labels.write_bytes(original)

    with live_app(items, labels, tmp_path / 'first-server.log') as (client, config):
        # Verify production UI wiring, literal-text component and blind defaults.
        components = {c['id']: c for c in config['components']}
        review = next(c for c in components.values()
                      if c.get('props', {}).get('label') == 'Текст отзыва')
        assert review['type'] == 'textbox'
        assert review['props']['interactive'] is False
        assert 'HIDDEN_SELLER_ANSWER' not in json.dumps(config)
        dependencies = config['dependencies']
        load = next(d for d in dependencies if d['api_name'] == 'show')
        save = next(d for d in dependencies if d['api_name'] == 'save')
        after_save = next(d for d in dependencies if d['trigger_after'] == save['id'])
        assert after_save['trigger_only_on_success'] is True
        assert review['id'] in load['outputs']
        first = client.predict(api_name='/show')
        assert first[1:] == (MALICIOUS, None, [], False, '')
        assert '0 / 2' in first[0]
        assert labels.read_bytes() == original

        # Rejected decision never writes; the very next valid request succeeds.
        with pytest.raises(Exception, match='Выберите'):
            client.predict(None, [], False, api_name='/save')
        assert labels.read_bytes() == original
        client.predict(NO, ['defect', 'wb_logistics'], True, api_name='/save')
        # HTTP clients call chained endpoints explicitly (a browser runs them).
        second = client.predict(api_name='/' + after_save['api_name'])
        assert second[1:] == ('second review', None, [], False, '')
        saved = labels.read_bytes()
        payload = json.loads(saved)
        assert payload['note'] == 'preserve metadata'
        assert payload['labels'][0]['id'] == 'outside-batch'
        one = payload['labels'][1]
        assert one['id'] == 'one' and one['need'] is False
        assert one['reasons'] == ['defect', 'wb_logistics'] and one['unsure'] is True
        backups = list((tmp_path / '.annotation-backups').glob('labels.json.*.bak'))
        assert len(backups) == 1 and backups[0].read_bytes() == original

        # Corruption while the server is live fails closed, then recovers once
        # the operator restores the intact file. The app never rewrites damage.
        labels.write_text('{broken JSON', encoding='utf-8')
        broken = labels.read_bytes()
        with pytest.raises(Exception):
            client.predict(NO, [], False, api_name='/save')
        assert labels.read_bytes() == broken
        assert len(list((tmp_path / '.annotation-backups').glob('*.bak'))) == 1
        labels.write_bytes(saved)
        client.predict(NO, [], False, api_name='/save')
        restored = json.loads(labels.read_bytes())
        assert restored['labels'][2]['id'] == 'two'

    # A new OS process and a new HTTP session reload durable human labels.
    with live_app(items, labels, tmp_path / 'restarted-server.log') as (client, _):
        reloaded = client.predict(api_name='/show')
        assert reloaded[1:] == (MALICIOUS, NO, ['defect', 'wb_logistics'], True, '')
        assert '2 / 2' in reloaded[0] and 'Всё размечено' in reloaded[0]

    labels.write_text('{broken JSON', encoding='utf-8')
    with pytest.raises(RuntimeError, match='JSONDecodeError'):
        with live_app(items, labels, tmp_path / 'corrupt-startup.log'):
            pytest.fail('Corrupt canonical labels must refuse application startup')
    assert labels.read_text(encoding='utf-8') == '{broken JSON'

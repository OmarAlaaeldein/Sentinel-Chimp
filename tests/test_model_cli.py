"""Local model CLI contracts; no Ollama installation or network required."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from core import model_config
from core.ollama import LocalModelError, OllamaClient, normalize_host
from main.cli import main, _write_csv, _json_safe


@pytest.fixture
def fake_ollama(monkeypatch):
    def request(self, endpoint, payload=None, timeout=None):
        if endpoint == '/api/tags':
            return {'models': [{'name': 'test:latest', 'size': 1000}]}
        if endpoint == '/api/show':
            return {'capabilities': ['completion'], 'parameters': 'temperature 0'}
        if endpoint == '/api/chat':
            assert payload['stream'] is False
            assert payload['options']['num_predict'] == 32
            return {'message': {'content': 'Hello'}, 'done': True, 'eval_count': 1}
        pytest.fail(endpoint)
    monkeypatch.setattr(OllamaClient, 'request', request)


def test_profile_round_trip_and_ask(tmp_path, capsys, fake_ollama):
    path = tmp_path / 'config.json'
    flags = ['--config', str(path), '--json']
    assert main(['models', 'select', 'test:latest', '--save-as', 'fast', '--num-predict', '32', *flags]) == 0
    selected = json.loads(capsys.readouterr().out)
    assert selected['data']['profile'] == 'fast'
    assert main(['ask', 'Hello', *flags]) == 0
    assert json.loads(capsys.readouterr().out)['data']['response'] == 'Hello'
    assert main(['models', 'select', 'test:latest', '--save-as', 'fast', *flags]) == 4
    assert json.loads(capsys.readouterr().out)['error']['code'] == 'PROFILE_EXISTS'
    assert model_config.load(path)['profiles']['fast']['options']['num_predict'] == 32
    assert main(['profiles', 'use', 'fast', *flags]) == 0
    capsys.readouterr()
    assert main(['models', 'current', *flags]) == 0
    assert json.loads(capsys.readouterr().out)['data']['profile'] == 'fast'


@pytest.mark.parametrize('tokens', [
    ['models', 'select'], ['unknown'], ['scan', 'SPY', '--max-expiries', '-1'],
    ['scan', 'SPY', '--limit', '-1'], ['scan', 'SPY', '--div', 'abc'],
])
def test_json_errors_are_single_objects(tokens, capsys):
    assert main([*tokens, '--json']) == 2
    output = capsys.readouterr()
    assert json.loads(output.out)['status'] == 'error'
    assert not output.err


@pytest.mark.parametrize('host', ['https://example.com', 'http://127.0.0.1@evil.com',
                                 'http://localhost/path', 'http://localhost?x=1'])
def test_remote_hosts_rejected(host):
    with pytest.raises(LocalModelError):
        normalize_host(host)


def test_local_hosts():
    assert normalize_host('127.0.0.1:11434') == 'http://127.0.0.1:11434'
    assert normalize_host('http://[::1]:11434/') == 'http://[::1]:11434'


def test_unknown_model_not_pulled(fake_ollama):
    with pytest.raises(LocalModelError, match='not installed'):
        OllamaClient().show('missing:latest')


def test_remote_model_rejected(monkeypatch):
    monkeypatch.setattr(OllamaClient, 'models', lambda self: [{'name': 'remote'}])
    monkeypatch.setattr(OllamaClient, 'request', lambda *a, **k: {'remote_host': 'cloud'})
    with pytest.raises(LocalModelError, match='remote backend'):
        OllamaClient().show('remote')


def test_invalid_config_preserved(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text('{broken')
    with pytest.raises(LocalModelError):
        model_config.update(path, lambda data: data.update(default_profile=None))
    assert path.read_text() == '{broken'


def test_concurrent_updates_keep_both_profiles(tmp_path):
    path = tmp_path / 'config.json'
    def save(name):
        def mutate(data):
            data['profiles'][name] = {'model': 'test:latest', 'options': {'temperature': 0, 'num_ctx': 4096, 'num_predict': 32}}
        model_config.update(path, mutate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(save, ['one', 'two']))
    assert set(model_config.load(path)['profiles']) == {'one', 'two'}


def test_bad_profile_does_not_contact_server(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(OllamaClient, 'show', lambda *a: pytest.fail('contacted server'))
    path = tmp_path / 'config.json'
    assert main(['models', 'select', 'test:latest', '--temperature', 'nan', '--config', str(path), '--json']) == 2
    assert not path.exists()
    assert json.loads(capsys.readouterr().out)['error']['code'] == 'INVALID_PROFILE'


def test_lightweight_imports():
    result = subprocess.run([sys.executable, '-c',
        "import main.cli, sentinel, sys; assert not set(['numpy', 'torch', 'tkinter', 'pandas']) & set(sys.modules)"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_empty_csv_has_headers(tmp_path, capsys):
    path = tmp_path / 'empty.csv'
    _write_csv(str(path), [])
    assert path.read_text().startswith('date,type,strike,')
    assert not capsys.readouterr().out
    assert _json_safe({'missing': float('nan')}) == {'missing': None}


def test_scan_csv_full_rows_and_partial_failure(tmp_path, monkeypatch, capsys):
    import core.data
    import core.scan_service
    monkeypatch.setattr(core.data, 'YFinanceProvider', lambda: None)
    row = core.scan_service.OptionScanRow
    import dataclasses
    values = {field.name: 0 for field in dataclasses.fields(row)}
    values.update(date='2030-01-01', type='call', verdict='Under', tag='')
    result = core.scan_service.ScanResult(rows=[row(**values), row(**values)],
        errors=[{'expiry': '2030-02-01', 'code': 'EXPIRY_FAILED', 'message': 'offline'}], requested_expiries=2)
    analysis = SimpleNamespace(to_dict=lambda: {'ticker': 'TEST'})
    monkeypatch.setattr(core.scan_service, 'run_ticker_scan', lambda *a, **k: (analysis, result))
    path = tmp_path / 'rows.csv'
    assert main(['scan', 'TEST', '--limit', '1', '--csv', str(path), '--json']) == 5
    output = json.loads(capsys.readouterr().out)
    assert output['count'] == 1 and output['status'] == 'partial'
    assert len(path.read_text().splitlines()) == 3
    result.requested_expiries = 1
    result.rows = []
    assert main(['scan', 'TEST', '--json']) == 4
    assert json.loads(capsys.readouterr().out)['status'] == 'error'


def test_http_error_retains_server_reason():
    from io import BytesIO
    from urllib.error import HTTPError
    client = OllamaClient()
    def fail(*a, **kw):
        raise HTTPError(client.host, 500, 'error', {}, BytesIO(b'{"error":"unknown model architecture"}'))
    client.opener = SimpleNamespace(open=fail)
    with pytest.raises(LocalModelError, match='unknown model architecture'):
        client.request('/api/chat', {})


def test_invalid_server_json():
    from io import BytesIO
    client = OllamaClient()
    client.opener = SimpleNamespace(open=lambda *a, **kw: BytesIO(b'not json'))
    with pytest.raises(LocalModelError, match='invalid JSON'):
        client.models()


def test_unavailable_server():
    from urllib.error import URLError
    client = OllamaClient()
    def fail(*a, **kw):
        raise URLError('connection refused')
    client.opener = SimpleNamespace(open=fail)
    with pytest.raises(LocalModelError, match='Cannot reach Ollama'):
        client.models()

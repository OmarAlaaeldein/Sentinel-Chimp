"""Small local-only Ollama client. Never pulls models or follows redirects."""
from __future__ import annotations

import ipaddress
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler


class LocalModelError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def normalize_host(host):
    if '://' not in host:
        host = 'http://' + host
    parts = urlsplit(host)
    try:
        local = parts.hostname == 'localhost' or ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        local = False
    if (not local or parts.scheme not in ('http', 'https') or parts.username
            or parts.password or parts.query or parts.fragment or parts.path not in ('', '/')):
        raise LocalModelError('INVALID_HOST', 'Ollama host must be a loopback HTTP(S) address without credentials or a path.')
    return host.rstrip('/')


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OllamaClient:
    def __init__(self, host='http://127.0.0.1:11434', timeout=5):
        self.host = normalize_host(host)
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def request(self, endpoint, payload=None, timeout=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(self.host + endpoint, data=data, headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=timeout or self.timeout) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise LocalModelError('INVALID_RESPONSE', 'Ollama response exceeds 8 MiB.')
            result = json.loads(raw)
            if not isinstance(result, dict) or result.get('error'):
                raise LocalModelError('OLLAMA_ERROR', str(result.get('error')) if isinstance(result, dict) else 'Expected a JSON object.')
            return result
        except HTTPError as exc:
            detail = ''
            try:
                body = json.loads(exc.read(4096))
                if isinstance(body, dict) and isinstance(body.get('error'), str):
                    detail = ' ' + body['error'][:1000]
            except (ValueError, OSError):
                pass
            finally:
                exc.close()
            raise LocalModelError('OLLAMA_HTTP_ERROR', f'Ollama returned HTTP {exc.code}.{detail}') from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise LocalModelError('OLLAMA_UNAVAILABLE', f'Cannot reach Ollama at {self.host}: {exc}') from exc
        except (ValueError, UnicodeError) as exc:
            raise LocalModelError('INVALID_RESPONSE', 'Ollama returned invalid JSON.') from exc

    def models(self):
        models = self.request('/api/tags').get('models')
        if not isinstance(models, list) or any(not isinstance(m, dict) or not isinstance(m.get('name'), str) for m in models):
            raise LocalModelError('INVALID_RESPONSE', 'Ollama did not return a valid model list.')
        return sorted(models, key=lambda m: m['name'])

    def show(self, name):
        if name not in {m['name'] for m in self.models()}:
            raise LocalModelError('MODEL_NOT_INSTALLED', f'Model {name!r} is not installed. Run models list for exact names.')
        result = self.request('/api/show', {'model': name})
        if result.get('remote_host') or result.get('remote_model'):
            raise LocalModelError('REMOTE_MODEL', 'This model uses a remote backend; choose a locally installed model.')
        return result

    def chat(self, name, prompt, options):
        details = self.show(name)
        if 'completion' not in details.get('capabilities', []):
            raise LocalModelError('UNSUPPORTED_MODEL', 'Selected model does not advertise completion support.')
        payload = {'model': name, 'messages': [{'role': 'user', 'content': prompt}],
                   'stream': False, 'options': options, 'keep_alive': '5m'}
        if 'thinking' in details.get('capabilities', []):
            payload['think'] = False
        result = self.request('/api/chat', payload, timeout=120)
        message = result.get('message')
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, str) or not result.get('done'):
            raise LocalModelError('INVALID_RESPONSE', 'Ollama did not return a completed text response.')
        return result

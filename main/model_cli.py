"""Human-readable and machine-readable commands for local model profiles."""
import json
import os

from core import model_config
from core.ollama import OllamaClient, LocalModelError


def add_commands(sub):
    models = sub.add_parser('models', help='Discover and select installed local Ollama models')
    commands = models.add_subparsers(dest='model_command', required=True)
    for name, help_text in [('list', 'List exact installed model names'), ('show', 'Inspect a model'),
                            ('select', 'Save a model profile and make it the default'),
                            ('current', 'Show the configured default profile')]:
        parser = commands.add_parser(name, help=help_text)
        common(parser)
        if name in ('show', 'select'):
            parser.add_argument('model')
        if name == 'select':
            parser.add_argument('--save-as', default='default', metavar='PROFILE')
            parser.add_argument('--temperature', type=float, default=0.0)
            parser.add_argument('--num-ctx', type=int, default=4096)
            parser.add_argument('--num-predict', type=int, default=512)
            parser.add_argument('--replace', action='store_true', help='Allow replacing an existing profile')
    profiles = sub.add_parser('profiles', help='Inspect saved profiles or change the default')
    commands = profiles.add_subparsers(dest='profile_command', required=True)
    common(commands.add_parser('list', help='List profiles without contacting Ollama'))
    use = commands.add_parser('use', help='Validate and select an existing profile')
    common(use)
    use.add_argument('name')
    ask = sub.add_parser('ask', help='Send a prompt to the selected local model')
    common(ask)
    ask.add_argument('prompt', help='Prompt text (maximum 16,000 characters)')
    ask.add_argument('--profile', help='Override default profile for this invocation')
    common(sub.add_parser('config', help='Show model configuration and its location'))


def common(parser):
    parser.add_argument('--json', action='store_true', help='Emit one versioned JSON object')
    parser.add_argument('--config', metavar='PATH', help='Config file; overrides SENTINEL_CONFIG')
    parser.add_argument('--host', help='Loopback Ollama URL; overrides SENTINEL_OLLAMA_HOST and OLLAMA_HOST')


def run(args):
    path = model_config.config_path(args.config)
    config = model_config.load(path)
    host = args.host or os.environ.get('SENTINEL_OLLAMA_HOST') or os.environ.get('OLLAMA_HOST') or 'http://127.0.0.1:11434'
    client = OllamaClient(host)
    if args.command == 'config':
        return {'path': str(path), 'host': client.host, **config}
    if args.command == 'profiles' and args.profile_command == 'list':
        return config
    if args.command == 'models':
        action = args.model_command
        if action == 'list':
            return {'host': client.host, 'models': client.models()}
        if action == 'show':
            return {'model': args.model, **client.show(args.model)}
        if action == 'current':
            name = config['default_profile']
            return {'profile': name, 'config': config['profiles'].get(name), 'host': client.host}
        profile = {'model': args.model, 'options': {'temperature': args.temperature,
                   'num_ctx': args.num_ctx, 'num_predict': args.num_predict}}
        # Reject bad input before contacting the server or touching configuration.
        try:
            model_config.validate({'schema_version': 1, 'default_profile': args.save_as,
                                   'profiles': {args.save_as: profile}})
        except ValueError as exc:
            raise LocalModelError('INVALID_PROFILE', str(exc)) from exc
        client.show(args.model)
        def save(data):
            if args.save_as in data['profiles'] and not args.replace:
                raise LocalModelError('PROFILE_EXISTS', 'Profile already exists; choose a new --save-as name or use --replace.')
            data['profiles'][args.save_as] = profile
            data['default_profile'] = args.save_as
        config = model_config.update(path, save)
        return {'profile': args.save_as, 'config': profile, 'path': str(path)}
    name = args.name if args.command == 'profiles' else args.profile or os.environ.get('SENTINEL_PROFILE') or config['default_profile']
    if name not in config['profiles']:
        raise LocalModelError('PROFILE_REQUIRED', 'Choose an existing profile with models select MODEL --save-as NAME.')
    profile = config['profiles'][name]
    if args.command == 'profiles':
        client.show(profile['model'])
        def select(data):
            if data['profiles'].get(name) != profile:
                raise LocalModelError('CONFIG_CHANGED', 'Profile changed during validation; retry.')
            data['default_profile'] = name
        model_config.update(path, select)
        return {'profile': name, 'config': profile, 'path': str(path)}
    if not args.prompt.strip() or len(args.prompt) > 16000:
        raise LocalModelError('INVALID_PROMPT', 'Prompt must contain 1–16,000 characters.')
    result = client.chat(profile['model'], args.prompt, profile['options'])
    return {'profile': name, 'model': profile['model'], 'response': result['message']['content'],
            'done_reason': result.get('done_reason'), 'eval_count': result.get('eval_count'),
            'eval_duration_ns': result.get('eval_duration')}


def render(data, as_json):
    if as_json:
        print(json.dumps({'schema_version': 1, 'status': 'ok', 'data': data}, indent=2, allow_nan=False))
    elif 'models' in data:
        models = data['models']
        if not models:
            print('No models installed in this Ollama server.')
        for model in models:
            details = model.get('details') or {}
            print(f"{model['name']}  {details.get('parameter_size', '?')}  {details.get('quantization_level', '?')}  {model.get('size', 0) / 1e9:.2f} GB")
    elif 'response' in data:
        print(data['response'])
    else:
        print(json.dumps(data, indent=2, allow_nan=False))

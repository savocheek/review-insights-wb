"""Start a local review app, or check prerequisites without loading the model.

python run.py --check
python run.py                     # CSV review app, localhost:7865
python run.py --app annotate      # manual annotation, localhost:7864
"""
import argparse
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
DEPENDENCIES = {
    'csv': ('gradio', 'torch', 'transformers'),
    'annotate': ('gradio',),
}


def check_model(model):
    """Check local files, not model execution or weight integrity; no downloads."""
    model = Path(model)
    errors = []
    for name in ['config.json', 'tokenizer_config.json', 'heads.pt']:
        path = model / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f'Missing or empty model file: {path}')
        elif name.endswith('.json'):
            try:
                config = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(config, dict):
                    raise ValueError('expected a JSON object')
                if name == 'config.json' and (not isinstance(config.get('model_type'), str)
                        or type(config.get('hidden_size')) is not int or config['hidden_size'] <= 0):
                    raise ValueError('model_type and positive hidden_size are required')
            except (ValueError, OSError) as exc:
                errors.append(f'Invalid {path}: {exc}')
    if not any((model / name).is_file() and (model / name).stat().st_size
               for name in ['tokenizer.json', 'vocab.txt']):
        errors.append(f'Missing tokenizer.json or vocab.txt in {model}')
    if not any((model / name).is_file() and (model / name).stat().st_size
               for name in ['model.safetensors', 'pytorch_model.bin']):
        errors.append(f'Missing local encoder weights model.safetensors or pytorch_model.bin in {model}')
    return errors


def preflight(app='csv', model=None):
    if app not in DEPENDENCIES:
        raise ValueError('app must be csv or annotate')
    errors = []
    if sys.version_info < (3, 10):
        errors.append('Python 3.10 or newer is required; choose a newer Python environment.')
    install_hint = f'Install the project dependencies in your chosen environment: "{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}"'
    for dependency in DEPENDENCIES[app]:
        if importlib.util.find_spec(dependency) is None:
            errors.append(f'Missing dependency {dependency}. {install_hint}')
    if importlib.util.find_spec('gradio') is not None:
        try:
            version = importlib.metadata.version('gradio')
            if int(version.split('.')[0]) != 6:
                errors.append(f'Gradio 6.x is required (found {version}). {install_hint}')
        except (importlib.metadata.PackageNotFoundError, ValueError):
            errors.append(f'Cannot verify the installed Gradio version. {install_hint}')
    if app == 'csv':
        model = ROOT / 'data/student/model' if model is None else Path(model)
        if not model.is_absolute():
            model = ROOT / model
        errors.extend(check_model(model))
    else:
        for name in ['labeling/needs_reply_items_enriched.json', 'labeling/needs_reply_labels.json']:
            if not (ROOT / name).is_file():
                errors.append(f'Missing annotation input: {ROOT / name}')
    return {'ok': not errors, 'app': app, 'python': sys.executable, 'errors': errors,
            'check_scope': 'Dependencies and required local file availability; no model load, inference, downloads, or installation.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', choices=['csv', 'annotate'], default='csv')
    parser.add_argument('--port', type=int)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    port = args.port if args.port is not None else (7865 if args.app == 'csv' else 7864)
    if not 1 <= port <= 65535:
        parser.error('--port must be between 1 and 65535')
    check = preflight(args.app)
    if not check['ok']:
        print('Cannot start the app:')
        for error in check['errors']:
            print(f'- {error}')
        print('Nothing was installed or downloaded.')
        if args.app == 'csv':
            print('Restore the saved model to data/student/model for CSV review.')
        return 1
    print(f'Ready: {args.app} at http://127.0.0.1:{port}')
    if args.app == 'csv':
        print(f'CSV example: upload {ROOT / "examples/demo_reviews.csv"} (fictional reviews), or your UTF-8 CSV with a text column.')
        print('Checks cover local file availability; the model loads only when the app runs.')
    else:
        print(f'Annotation input: {ROOT / "labeling/needs_reply_items_enriched.json"}; no model is required.')
    if args.check:
        return 0
    try:
        module = importlib.import_module('csv_app' if args.app == 'csv' else 'needs_reply_label')
        app = module.build_app()
        app.launch(server_name='127.0.0.1', server_port=port, share=False)
    except (ImportError, OSError) as exc:
        print(f'Could not launch {args.app}: {exc}. Run "{sys.executable}" "{ROOT / "run.py"}" --app {args.app} --check to review prerequisites.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

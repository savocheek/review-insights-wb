"""Opt-in continuation on manual need/reason labels, never sentiment.

Prepare: python train_manual.py --output data/manual_runs/run-001 --prepare-only
Train:   python train_manual.py --output data/manual_runs/run-002 --epochs 3

Outputs must be fresh. All model loading is local. Development scores on the
already inspected manual labels are not an independent benchmark. No deployment
or replacement of data/student/model is performed.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata

import torch
from torch.nn import functional as F

REASON_KEYS = ('defect', 'refund', 'delivery', 'question', 'safety', 'misinfo',
               'strong_negative', 'wb_logistics')
ROOT = Path(__file__).resolve().parent


def normalize_text(text):
    text = unicodedata.normalize('NFKC', text).casefold().replace('ё', 'е')
    return ' '.join(re.sub(r'[^\w\s]', ' ', text).split())


def provenance(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_manual(labels, item_paths, audit, parquet):
    """IDs in the original labeling corpus are WB parquet row indexes."""
    import pandas as pd
    inputs = [Path(labels), *map(Path, item_paths), Path(audit), Path(parquet)]
    hashes = [provenance(p) for p in inputs]
    items = {}
    for path in item_paths:
        for item in read_json(path)['items']:
            key = str(item['id'])
            if key in items and items[key]['text'] != item['text']:
                raise ValueError(f'conflicting item text for ID {key}')
            items[key] = item
    audit_rows = {}
    for item in read_json(audit)['rows']:
        key = str(item['id'])
        if key in audit_rows:
            raise ValueError(f'duplicate audit ID {key}')
        audit_rows[key] = item
    products = pd.read_parquet(parquet, columns=['nmId'])
    if not products.index.is_unique:
        raise ValueError('parquet ID index is not unique')
    rows, seen = [], set()
    for label in read_json(labels)['labels']:
        key = str(label['id'])
        if key in seen:
            raise ValueError(f'duplicate manual ID {key}')
        seen.add(key)
        if key not in items:
            raise ValueError(f'missing item ID {key}')
        try:
            index = int(key)
        except ValueError as exc:
            raise ValueError(f'invalid WB row ID {key}') from exc
        if index not in products.index:
            raise ValueError(f'missing parquet ID {key}')
        product = products.loc[index, 'nmId']
        if pd.isna(product):
            raise ValueError(f'missing product for ID {key}')
        if type(label.get('need')) is not bool or type(label.get('unsure', False)) is not bool:
            raise ValueError(f'invalid manual target for ID {key}')
        reasons = label.get('reasons')
        if not isinstance(reasons, list) or any(k not in REASON_KEYS for k in reasons):
            raise ValueError(f'invalid manual reasons for ID {key}')
        text = items[key]['text']
        if not isinstance(text, str) or not normalize_text(text):
            raise ValueError(f'empty text for ID {key}')
        a = audit_rows.get(key)
        # Audit decisions are trusted only for exactly the same reviewed label.
        if a and (a.get('text') != text or a.get('need') != label['need']
                  or set(a.get('reasons', [])) != set(reasons)):
            raise ValueError(f'stale audit for ID {key}; rerun audit before training')
        if a is None and not label.get('unsure', False):
            raise ValueError(f'missing audit ID {key}; review before training')
        rows.append({'id': key, 'nmId': int(product), 'text': text,
                     'need': label['need'], 'reasons': list(reasons),
                     'unsure': label.get('unsure', False),
                     'review_note': a.get('review_note') if a else None,
                     'used_in_teacher_fewshot': bool(a and a.get('used_in_teacher_fewshot'))})
    return rows, hashes


def select_rows(rows):
    eligible, excluded = [], []
    for source in rows:
        row = copy.deepcopy(source)
        row.update(sentiment=None, sentiment_loss_mask=False,
                   role='development_only',
                   exposure='seen_by_teacher' if row.get('used_in_teacher_fewshot')
                   else 'not_in_teacher_fewshot')
        if row.get('unsure') or row.get('review_note'):
            row['exclusion'] = 'uncertain' if row.get('unsure') else 'audit_conflict'
            excluded.append(row)
        else:
            eligible.append(row)
    return eligible, excluded


def group_split(rows, val_fraction=.25, seed=42):
    """Union by product OR normalized text; split connected components."""
    if not 0 < val_fraction < 1:
        raise ValueError('development fraction must be between zero and one')
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    ids = set()
    for i, row in enumerate(rows):
        if row['id'] in ids:
            raise ValueError('duplicate ID in split')
        ids.add(row['id'])
        for key in [('product', str(row['nmId'])), ('text', normalize_text(row['text']))]:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    components = {}
    for i, row in enumerate(rows):
        components.setdefault(find(i), []).append(row)
    groups = list(components.values())
    if len(groups) < 2:
        raise ValueError('need at least two independent product/text components')
    random.Random(seed).shuffle(groups)
    development, train = [], []
    target = max(1, round(len(rows) * val_fraction))
    for group in groups[:-1]:
        (development if len(development) < target else train).extend(group)
    train.extend(groups[-1])
    result = {'train': train, 'development': development}
    assert_no_overlap(result)
    return result


def assert_no_overlap(splits):
    train, dev = splits['train'], splits['development']
    for field, transform in [('id', str), ('nmId', str), ('text', normalize_text)]:
        if {transform(r[field]) for r in train} & {transform(r[field]) for r in dev}:
            raise ValueError(f'{field} overlap across training and development')


def reserve_output(output, protected):
    output = Path(output).resolve()
    for source in protected:
        source = Path(source).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f'output overlaps protected input: {source}')
    if output.exists():
        raise ValueError(f'output must be a fresh nonexistent directory: {output}')
    output.mkdir(parents=True, exist_ok=False)
    return output


class ManualStudent(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder
        self.drop = torch.nn.Dropout(.1)
        hidden = encoder.config.hidden_size
        self.need = torch.nn.Linear(hidden, 1)
        self.reasons = torch.nn.Linear(hidden, 8)
        self.sent = torch.nn.Linear(hidden, 4)
        self.sent.requires_grad_(False)

    def forward(self, **tokens):
        x = self.drop(self.enc(**tokens).last_hidden_state[:, 0])
        return self.need(x).squeeze(-1), self.reasons(x), self.sent(x)


def manual_loss(outputs, need, reasons):
    n, r, _ = outputs
    return F.binary_cross_entropy_with_logits(n, need) + F.binary_cross_entropy_with_logits(r, reasons)


def load_model(init):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(init), local_files_only=True)
    model = ManualStudent(AutoModel.from_pretrained(str(init), local_files_only=True))
    state = torch.load(Path(init) / 'heads.pt', map_location='cpu', weights_only=True)
    expected = {k for k in model.state_dict() if not k.startswith('enc.')}
    if set(state) != expected:
        raise ValueError('saved heads must exactly match need + 8 reasons + 4 sentiment')
    model.load_state_dict(state, strict=False)
    return model, tokenizer


def targets(rows, device):
    return (torch.tensor([float(r['need']) for r in rows], device=device),
            torch.tensor([[float(k in r['reasons']) for k in REASON_KEYS] for r in rows], device=device))


@torch.no_grad()
def development_metrics(model, tokenizer, rows, device, batch):
    """Fixed thresholds; development only, stratified by teacher fewshot exposure."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    model.eval()
    needs, reasons = [], []
    for start in range(0, len(rows), batch):
        subset = rows[start:start + batch]
        tokens = tokenizer([r['text'] for r in subset], truncation=True, max_length=256,
                           padding=True, return_tensors='pt').to(device)
        n, r, _ = model(**tokens)
        needs.extend((n.sigmoid() >= .5).cpu().tolist())
        reasons.extend((r.sigmoid() >= .5).cpu().tolist())
    result = {}
    for exposure in ['all', 'seen_by_teacher', 'not_in_teacher_fewshot']:
        idx = [i for i, r in enumerate(rows) if exposure == 'all' or r['exposure'] == exposure]
        gold = [int(rows[i]['need']) for i in idx]
        pred = [needs[i] for i in idx]
        g_reasons = [[k in rows[i]['reasons'] for k in REASON_KEYS] for i in idx]
        result[exposure] = {'n': len(idx), 'positives': sum(gold)}
        if idx:
            result[exposure].update(need_f1=float(f1_score(gold, pred, zero_division=0)),
                                   need_precision=float(precision_score(gold, pred, zero_division=0)),
                                   need_recall=float(recall_score(gold, pred, zero_division=0)),
                                   reasons_micro_f1=float(f1_score(g_reasons, [reasons[i] for i in idx],
                                                                average='micro', zero_division=0)))
    return result


def train(splits, args, output):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, tokenizer = load_model(args.init)
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(args.seed)
    history = []
    for epoch in range(args.epochs):
        rows = list(splits['train'])
        rng.shuffle(rows)
        model.train()
        total = 0.
        for start in range(0, len(rows), args.batch):
            subset = rows[start:start + args.batch]
            tokens = tokenizer([r['text'] for r in subset], truncation=True, max_length=256,
                               padding=True, return_tensors='pt').to(device)
            opt.zero_grad(set_to_none=True)
            loss = manual_loss(model(**tokens), *targets(subset, device))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite manual loss; refusing to save model')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            opt.step()
            total += float(loss.detach()) * len(subset)
        history.append({'epoch': epoch + 1, 'train_loss': total / len(rows),
                        'development': development_metrics(model, tokenizer, splits['development'], device, args.batch)})
        print(json.dumps(history[-1]), flush=True)
    destination = output / 'model'
    model.enc.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith('enc.')}, destination / 'heads.pt')
    return history


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT / 'labeling/needs_reply_labels.json')
    parser.add_argument('--items', type=Path, nargs='+', default=[ROOT / 'labeling/needs_reply_items.json', ROOT / 'labeling/needs_reply_items_enriched.json'])
    parser.add_argument('--audit', type=Path, default=ROOT / 'data/audit/human_training_candidates.json')
    parser.add_argument('--parquet', type=Path, default=ROOT / 'data/wb.parquet')
    parser.add_argument('--init', type=Path, default=ROOT / 'data/student/model')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--development-fraction', type=float, default=.25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch < 1 or args.threads < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('epochs, batch, threads and finite learning rate must be positive')
    if not args.init.is_dir():
        raise ValueError('init must be an existing local saved student directory')
    protected = [args.labels, *args.items, args.audit, args.parquet, args.init,
                 ROOT / 'data/student', ROOT / 'models', ROOT / 'labeling', ROOT / 'data/teacher']
    # Refuse unsafe paths before any expensive input/model operation.
    resolved = args.output.resolve()
    for source in protected:
        source = source.resolve()
        if resolved == source or resolved in source.parents or source in resolved.parents:
            raise ValueError(f'output overlaps protected input: {source}')
    if resolved.exists():
        raise ValueError('output must be a fresh nonexistent directory')
    rows, hashes = load_manual(args.labels, args.items, args.audit, args.parquet)
    hashes.extend(provenance(p) for p in sorted(args.init.rglob('*')) if p.is_file())
    eligible, excluded = select_rows(rows)
    splits = group_split(eligible, args.development_fraction, args.seed)
    output = reserve_output(args.output, protected)
    manifest = {'status': 'prepared_only' if args.prepare_only else 'training_started',
                'independent_benchmark': False,
                'limitations': 'Existing inspected manual labels: development metrics only. Sentiment head is frozen and unsupervised; encoder updates can change its predictions. No sentiment quality claim.',
                'normalization': 'NFKC, casefold, yo to ye, punctuation to spaces, collapsed whitespace',
                'supervision': {'need': True, 'reasons': list(REASON_KEYS), 'sentiment': False},
                'init': str(args.init.resolve()), 'provenance': hashes,
                'counts': {**{k: len(v) for k, v in splits.items()}, 'excluded': len(excluded)},
                'class_counts': {name: {'positive': sum(int(r['need']) for r in rs),
                                       'negative': sum(not r['need'] for r in rs)}
                                 for name, rs in {**splits, 'excluded': excluded}.items()},
                'warnings': [f'{name} has no positive need labels; recall/F1 cannot assess positive detection'
                             for name, rs in splits.items() if not any(r['need'] for r in rs)],
                'seed': args.seed, 'development_fraction': args.development_fraction,
                'thresholds': {'need': .5, 'reasons': .5},
                'training': {'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr, 'device': args.device}}
    for name, data in {**splits, 'excluded': excluded}.items():
        write_json(output / f'{name}.json', data)
    write_json(output / 'manifest.json', manifest)
    if not args.prepare_only:
        try:
            history = train(splits, args, output)
            write_json(output / 'development_metrics.json', {'independent_benchmark': False, 'history': history})
            manifest['status'] = 'completed_development_training'
        except Exception as exc:
            manifest.update(status='failed', error=str(exc))
            write_json(output / 'manifest.json', manifest)
            raise
        write_json(output / 'manifest.json', manifest)
    print(json.dumps({'output': str(output), 'status': manifest['status'], 'counts': manifest['counts']}), flush=True)
    return manifest


if __name__ == '__main__':
    main()

"""Prepare a private, leakage-isolated overnight WB review corpus; never train.

Default: 500 seeded random + 500 difficult reviews, current frozen student only.
No external requests. Existing corpus/manifest outputs are never overwritten.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261111
DEMO_PRODUCTS = (1017642, 11709626, 1169996, 10009941)
MARKERS = {
    'safety': re.compile(r'аллерг|сыпь|ожог|отрав|искрит|током|травм|порез|опасн', re.I),
    'substantive_negative': re.compile(r'перестал[аио]? работа|не работа|сломал|брак|отказ.{0,30}возврат|'
                                       r'возврат.{0,30}отказ|верните деньги|компенсац|подделк|протека|развалил', re.I),
}


def normalize_text(text):
    text = unicodedata.normalize('NFKC', text).casefold().replace('ё', 'е')
    return ' '.join(re.sub(r'[^\w\s]|_', ' ', text).split())


def _walk_refs(value, ids, products, texts):
    if isinstance(value, list):
        for item in value:
            _walk_refs(item, ids, products, texts)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in ('id', 'review_id', 'reviewId') and isinstance(item, (int, str)) and not isinstance(item, bool):
                ids.add(str(item).strip())
            elif key in ('ids', 'review_ids') and isinstance(item, list):
                ids.update(str(i).strip() for i in item if isinstance(i, (int, str)) and not isinstance(i, bool))
            elif key in ('nmId', 'nm_id', 'product_id') and isinstance(item, (int, str)):
                try:
                    products.add(int(item))
                except ValueError as error:
                    raise ValueError('Invalid prior product reference') from error
            elif key == 'text' and isinstance(item, str) and item.strip():
                texts.add(normalize_text(item))
            if isinstance(item, (dict, list)):
                _walk_refs(item, ids, products, texts)


def collect_exclusions(wb, paths, explicit_products=()):
    """Resolve all prior review IDs through the immutable WB row index."""
    ids, products, texts = set(), set(map(int, explicit_products)), set()
    for path in paths:
        path = Path(path)
        if path.suffix == '.jsonl':
            with path.open(encoding='utf-8-sig') as stream:
                for line_no, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            _walk_refs(json.loads(line), ids, products, texts)
                        except (ValueError, TypeError) as error:
                            raise ValueError(f'Invalid exclusion source {path.name}, line {line_no}') from error
        else:
            _walk_refs(json.loads(path.read_text(encoding='utf-8-sig')), ids, products, texts)
    integer_ids = {int(i) for i in ids if i.isdecimal()}
    resolved = wb.loc[wb.index.isin(integer_ids), ['nmId', 'text']]
    for product, text in resolved.itertuples(index=False, name=None):
        products.add(int(product))
        if isinstance(text, str) and text.strip():
            texts.add(normalize_text(text))
    unresolved = ids - {str(i) for i in resolved.index}
    return {'ids': ids, 'products': products, 'texts': texts, 'unresolved_ids': sorted(unresolved)}


def eligible_pool(wb, exclusions):
    """Keep first eligible WB row per normalized text and product, deterministically."""
    required = {'nmId', 'text', 'split'}
    if not required <= set(wb.columns) or not wb.index.is_unique:
        raise ValueError('WB must have a unique row index and nmId/text/split columns')
    held = wb.loc[wb['split'] != 'train', ['nmId', 'text']]
    held_texts = {normalize_text(t) for t in held['text'] if isinstance(t, str) and t.strip()}
    held_products = set(map(int, held['nmId']))
    banned_texts = exclusions['texts'] | held_texts
    banned_products = exclusions['products'] | held_products
    seen_texts, seen_products, rows = set(), set(), []
    for index, product, text, split in wb[['nmId', 'text', 'split']].itertuples(name=None):
        key, product = str(index), int(product)
        if split != 'train' or key in exclusions['ids'] or product in banned_products or product in seen_products:
            continue
        if not isinstance(text, str) or not 20 <= len(text.strip()) <= 1200 or len(text.split()) > 220:
            continue
        norm = normalize_text(text)
        if not norm or norm in banned_texts or norm in seen_texts:
            continue
        seen_texts.add(norm)
        seen_products.add(product)
        rows.append({'id': key, 'nmId': product, 'text': text.strip(), 'split': 'train'})
    return rows


def _draw_bins(rows, edges, target, rng):
    selected = []
    for bin_index, (lo, hi) in enumerate(zip(edges, edges[1:])):
        eligible = [r for r in rows if lo <= r['student_need_score'] < hi]
        quota = target // (len(edges) - 1) + int(bin_index < target % (len(edges) - 1))
        selected.extend(rng.sample(eligible, min(len(eligible), quota)))
    used = {r['id'] for r in selected}
    rest = [r for r in rows if r['id'] not in used]
    selected.extend(rng.sample(rest, min(len(rest), target - len(selected))))
    return selected


def _markers(text):
    return [name for name, pattern in MARKERS.items() if pattern.search(text)]


def select_reviews(pool, predictor, random_n=500, hard_n=500, candidate_limit=12000, seed=SEED):
    if min(random_n, hard_n, candidate_limit) < 1 or len(pool) < random_n + hard_n:
        raise ValueError('Not enough unique eligible reviews for requested random/hard samples')
    if len({r['id'] for r in pool}) != len(pool) or len({r['nmId'] for r in pool}) != len(pool):
        raise ValueError('Selection pool must have unique IDs and products')
    rng = random.Random(seed)
    random_rows = rng.sample(pool, random_n)
    random_ids = {r['id'] for r in random_rows}
    available = [r for r in pool if r['id'] not in random_ids]
    candidates = rng.sample(available, min(candidate_limit, len(available)))
    scored_rows = random_rows + candidates
    scores = predictor([r['text'] for r in scored_rows])
    if len(scores) != len(scored_rows) or any(not math.isfinite(float(s)) or not 0 <= float(s) <= 1 for s in scores):
        raise ValueError('Student returned invalid need scores or row count')
    scored = [{**row, 'student_need_score': float(score)} for row, score in zip(scored_rows, scores)]
    random_rows, candidates = scored[:random_n], scored[random_n:]
    uncertain = [{**r, 'selection_stratum': 'uncertain'} for r in candidates if .2 <= r['student_need_score'] < .85]
    high = [{**r, 'selection_stratum': 'high_score'} for r in candidates if r['student_need_score'] >= .85]
    uncertain_n, high_n = hard_n // 2, hard_n - hard_n // 2
    selected = (_draw_bins(uncertain, (.2, .33, .46, .59, .72, .85), uncertain_n, rng)
                + _draw_bins(high, (.85, .88, .91, .94, .97, 1.000001), high_n, rng))
    used = {r['id'] for r in selected}
    # Reallocate a deficient score family before falling back to textual markers.
    remaining_scores = [r for r in uncertain + high if r['id'] not in used]
    selected.extend(rng.sample(remaining_scores, min(len(remaining_scores), hard_n - len(selected))))
    used = {r['id'] for r in selected}
    fallback = [{**r, 'selection_stratum': 'marker_fallback', 'fallback_markers': _markers(r['text'])}
                for r in candidates if r['id'] not in used and _markers(r['text'])]
    selected.extend(rng.sample(fallback, min(len(fallback), hard_n - len(selected))))
    if len(selected) != hard_n:
        raise ValueError('Not enough hard candidates in score strata or substantive marker fallback')
    result = ([{**r, 'selection_group': 'random', 'selection_stratum': 'uniform_random'} for r in random_rows]
              + [{**r, 'selection_group': 'hard'} for r in selected])
    counts = Counter(r['selection_stratum'] for r in selected)
    return result, {'seed': seed, 'random_count': random_n, 'hard_count': hard_n,
                    'scored_candidate_count': len(candidates), 'scored_random_count': random_n,
                    'requested_hard_quotas': {'uncertain': uncertain_n, 'high_score': high_n},
                    'hard_strata_counts': dict(counts), 'fallback_count': counts['marker_fallback'],
                    'fallback_explanation': ('If score families cannot fill the target, redistribute within '
                                             'score-eligible strata, then use low-score reviews containing '
                                             'explicit safety or substantive defect/refund/negative markers. '
                                             'These are selection heuristics, never human or teacher labels.'),
                    'candidate_scores': [{'id': r['id'], 'student_need_score': r['student_need_score']}
                                         for r in candidates]}


def predict_gpu(texts):
    import torch
    sys.path.insert(0, str(ROOT))
    from train_manual import load_model
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for overnight preparation; no implicit slow CPU run')
    torch.set_num_threads(4)
    model, tokenizer = load_model(ROOT / 'data/student/model')
    model.to('cuda').eval().requires_grad_(False)
    scores = []
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), 32):
                tokens = tokenizer(texts[start:start + 32], truncation=True, max_length=256,
                                   padding=True, return_tensors='pt').to('cuda')
                need, _, _ = model(**tokens)
                scores.extend(need.sigmoid().cpu().tolist())
                if start % 1024 == 0:
                    print(f'scored {len(scores)}/{len(texts)} elapsed={time.monotonic()-started:.1f}s', flush=True)
        torch.cuda.synchronize()
    finally:
        del model, tokenizer
        torch.cuda.empty_cache()
    return scores


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_outputs(output, rows, manifest):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    corpus, info = output / 'corpus.jsonl', output / 'selection_manifest.json'
    if corpus.exists() or info.exists():
        raise FileExistsError('Refusing to overwrite existing overnight corpus or manifest')
    encoded = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode('utf-8')
    payload = {**manifest, 'corpus_sha256': hashlib.sha256(encoded).hexdigest()}
    for path, content in [(corpus, encoded), (info, (json.dumps(payload, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))]:
        with path.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())


def source_paths():
    required = [ROOT/'data/teacher/corpus.jsonl', ROOT/'data/needs_reply_items.jsonl',
                ROOT/'labeling/needs_reply_items.json', ROOT/'labeling/needs_reply_items_enriched.json',
                ROOT/'labeling/needs_reply_labels.json']
    if any(not p.is_file() for p in required):
        raise FileNotFoundError('A required prior teacher/annotation source is missing')
    optional = [ROOT/'data/demo_maternity/unreviewed_export.json', ROOT/'reports/demo_maternity.json']
    return sorted(set(required + list((ROOT/'data/llm_probe').glob('*.jsonl'))
                      + list((ROOT/'data/demo_maternity').glob('*.jsonl')) + [p for p in optional if p.is_file()]))


def main():
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/nightly/20261111')
    parser.add_argument('--candidate-pool', type=int, default=12000)
    args = parser.parse_args()
    if not 500 <= args.candidate_pool <= 12000:
        raise ValueError('Candidate pool must be between 500 and 12000')
    output = args.output.resolve()
    protected = [ROOT/'data/wb.parquet', *source_paths(), *sorted((ROOT/'data/student/model').glob('*'))]
    protected = [p for p in protected if p.is_file()]
    if any(output == p or output in p.parents for p in protected):
        raise ValueError('Output cannot overlap a protected source')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'corpus.jsonl').exists() or (output/'selection_manifest.json').exists():
        raise FileExistsError('Refusing to overwrite existing overnight corpus or manifest')
    lock = output/'.prepare.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        started = time.monotonic()
        hashes = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
        print('source hashes captured; loading immutable WB', flush=True)
        wb = pd.read_parquet(ROOT/'data/wb.parquet', columns=['nmId', 'text', 'split'])
        excluded = collect_exclusions(wb, source_paths(), DEMO_PRODUCTS)
        if excluded['unresolved_ids']:
            raise ValueError(f'Unresolved prior review ids: {len(excluded["unresolved_ids"])}; refusing weak isolation')
        print(f'exclusions ids={len(excluded["ids"])} products={len(excluded["products"])} texts={len(excluded["texts"])}', flush=True)
        pool = eligible_pool(wb, excluded)
        print(f'eligible unique train products/texts={len(pool)}; scoring bounded pool on current frozen student', flush=True)
        rows, selection = select_reviews(pool, predict_gpu, candidate_limit=args.candidate_pool)
        assert len(rows) == 1000 and len({r['id'] for r in rows}) == 1000
        assert len({r['nmId'] for r in rows}) == 1000
        assert len({normalize_text(r['text']) for r in rows}) == 1000
        if any(sha256(ROOT/p) != digest for p,digest in hashes.items()):
            raise RuntimeError('A protected source changed during preparation; no corpus saved')
        manifest = {**selection, 'created_at': datetime.now(timezone.utc).isoformat(),
                    'source_hashes': hashes, 'script_sha256': sha256(__file__),
                    'model': 'data/student/model (existing frozen current student)',
                    'score_kind': 'uncalibrated model score, not a label or calibrated probability',
                    'scoring': {'device': 'cuda', 'batch_size': 32, 'max_length_tokens': 256, 'amp': False},
                    'eligible_unique_pool': len(pool), 'source_wb_rows': len(wb),
                    'exclusions': {key: len(excluded[key]) for key in ('ids','products','texts')},
                    'explicit_demo_products': list(DEMO_PRODUCTS),
                    'isolation': 'WB train only; exclude prior IDs/products/normalized texts and every val/test product/text',
                    'normalization': 'NFKC, casefold, ё->е, punctuation to spaces, collapse whitespace',
                    'length_limits': {'min_chars':20,'max_chars':1200,'max_words':220},
                    'deduplication': 'first eligible WB row-index order per normalized text and product',
                    'source_hashes_unchanged': True, 'elapsed_seconds': round(time.monotonic()-started,2)}
        write_outputs(output, rows, manifest)
        print(json.dumps({'output':str(output),'count':len(rows),'strata':selection['hard_strata_counts'],
                          'fallback':selection['fallback_count'],'seconds':manifest['elapsed_seconds']}, ensure_ascii=False), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

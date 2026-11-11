"""Leakage, reproducibility and immutable-output contracts for overnight selection."""
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def prep():
    assert importlib.util.find_spec('scripts.nightly_prepare') is not None, 'nightly preparation script missing'
    from scripts import nightly_prepare
    return nightly_prepare


def test_normalization_matches_punctuation_case_and_unicode(prep):
    assert prep.normalize_text('  ЁЖИК — сломался!!! Ａ  ') == 'ежик сломался a'
    assert prep.normalize_text('То_же / самый отзыв') == 'то же самый отзыв'


def test_prior_nested_ids_resolve_product_and_text_and_crosssplit_isolation(prep, tmp_path):
    frame = pd.DataFrame([
        {'nmId': 10, 'split': 'train', 'text': 'Отзыв ранее показан модели полностью'},
        {'nmId': 10, 'split': 'train', 'text': 'Другой текст того же старого товара'},
        {'nmId': 20, 'split': 'train', 'text': 'ОТЗЫВ ранее показан модели полностью!!!'},
        {'nmId': 30, 'split': 'val', 'text': 'Уникальный отзыв из контрольного набора'},
        {'nmId': 40, 'split': 'train', 'text': 'Уникальный отзыв из контрольного набора!'},
        {'nmId': 50, 'split': 'train', 'text': 'Совершенно новый отзыв для ночного корпуса'},
        {'nmId': 60, 'split': 'train', 'text': 'Товар дополнительно запрещен явно по коду'},
    ], index=[100,101,102,103,104,105,106])
    prior = tmp_path / 'prior.jsonl'
    prior.write_text(json.dumps({'results': [{'review': {'id': '100'}}],
                                'payload': {'nmId': 60, 'ids': [106]}}) + '\n', encoding='utf-8')
    before = prior.read_bytes()
    excluded = prep.collect_exclusions(frame, [prior])
    assert {'100', '106'} <= excluded['ids']
    assert {10, 60} <= excluded['products']
    assert prep.normalize_text(frame.loc[100, 'text']) in excluded['texts']
    eligible = prep.eligible_pool(frame, excluded)
    assert [r['id'] for r in eligible] == ['105']
    assert prior.read_bytes() == before


def test_pool_bounds_and_dedupes_product_and_normalized_text(prep):
    rows = [
        {'nmId': 1, 'split': 'train', 'text': 'Длинный уникальный отзыв покупателя номер один'},
        {'nmId': 1, 'split': 'train', 'text': 'Ещё один допустимый отзыв того же товара'},
        {'nmId': 2, 'split': 'train', 'text': 'ДЛИННЫЙ уникальный отзыв покупателя номер один!'},
        {'nmId': 3, 'split': 'test', 'text': 'Другой отзыв в тестовом наборе данных'},
        {'nmId': 4, 'split': 'train', 'text': 'коротко'},
        {'nmId': 5, 'split': 'train', 'text': 'а' * 1201},
        {'nmId': 6, 'split': 'train', 'text': 'аа ' * 221},
        {'nmId': 7, 'split': 'train', 'text': 'Новый допустимый текст второго товара без повторов'},
    ]
    result = prep.eligible_pool(pd.DataFrame(rows), {'ids': set(), 'products': set(), 'texts': set()})
    assert [r['id'] for r in result] == ['0', '7']
    assert len({r['nmId'] for r in result}) == len(result)
    assert len({prep.normalize_text(r['text']) for r in result}) == len(result)


def test_fixed_seed_reproducible_disjoint_random_and_hard(prep):
    pool = [{'id': str(i), 'nmId': i, 'split': 'train', 'text': f'Новый отзыв номер {i} требует внимательной проверки'}
            for i in range(100)]
    def scores(texts):
        return [.96 if int(t.split()[3]) % 2 else .5 for t in texts]
    first, metadata = prep.select_reviews(pool, scores, random_n=4, hard_n=6, candidate_limit=40)
    second, other_metadata = prep.select_reviews(pool, scores, random_n=4, hard_n=6, candidate_limit=40)
    assert first == second
    assert metadata == other_metadata
    assert len(first) == 10
    assert len({r['id'] for r in first}) == 10
    assert len({r['nmId'] for r in first}) == 10
    assert sum(r['selection_group'] == 'random' for r in first) == 4
    assert sum(r['selection_group'] == 'hard' for r in first) == 6
    hard = [r for r in first if r['selection_group'] == 'hard']
    assert sum(r['student_need_score'] >= .85 for r in hard) == 3
    assert sum(.2 <= r['student_need_score'] < .85 for r in hard) == 3


def test_fallback_is_explicit_and_cannot_fill_with_unmarked_easy_reviews(prep):
    pool = [{'id': str(i), 'nmId': i, 'split': 'train',
             'text': f'После использования появился ожог кожи номер {i}'} for i in range(20)]
    selected, metadata = prep.select_reviews(pool, lambda ts: [.05] * len(ts), random_n=2, hard_n=4, candidate_limit=15)
    assert sum(r.get('selection_stratum') == 'marker_fallback' for r in selected) == 4
    assert metadata['fallback_count'] == 4
    assert metadata['fallback_explanation']
    plain = [{**r, 'text': f'Качественный товар покупателю понравился номер {i}'} for i,r in enumerate(pool)]
    with pytest.raises(ValueError, match='hard'):
        prep.select_reviews(plain, lambda ts: [.05] * len(ts), random_n=2, hard_n=4, candidate_limit=15)


def test_output_never_overwrites_existing_corpus(prep, tmp_path):
    output = tmp_path / 'nightly'
    rows = [{'id': '1', 'nmId': 1, 'text': 'Новый отзыв', 'selection_group': 'random'}]
    prep.write_outputs(output, rows, {'seed': 20261111})
    original = (output / 'corpus.jsonl').read_bytes()
    assert json.loads((output / 'selection_manifest.json').read_text(encoding='utf-8'))['seed'] == 20261111
    with pytest.raises(FileExistsError):
        prep.write_outputs(output, [{**rows[0], 'text': 'changed'}], {})
    assert (output / 'corpus.jsonl').read_bytes() == original

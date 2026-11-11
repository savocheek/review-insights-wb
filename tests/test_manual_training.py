"""Safety and supervision contracts for opt-in manual continuation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def manual():
    assert importlib.util.find_spec('train_manual') is not None, 'manual training runner is missing'
    import train_manual
    return train_manual


def row(i, product=None, text=None, **kw):
    return dict(id=str(i), nmId=product if product is not None else i,
                text=text or f'review {i}', need=False, reasons=[], unsure=False,
                review_note=None, used_in_teacher_fewshot=False, **kw)


def test_connected_product_and_normalized_text_never_cross_splits(manual):
    rows = [row(1, 10, ' SAME  text '), row(2, 20, 'same text'),
            row(3, 20), row(4, 30), row(5, 40), row(6, 50)]
    before = copy.deepcopy(rows)
    splits = manual.group_split(rows, val_fraction=.3, seed=42)
    side = {r['id']: k for k, rs in splits.items() for r in rs}
    assert side['1'] == side['2'] == side['3']
    assert splits['train'] and splits['development']
    assert len(side) == 6
    assert rows == before
    assert manual.group_split(rows, val_fraction=.3, seed=42) == splits
    with pytest.raises(ValueError, match='component'):
        manual.group_split(rows[:3], val_fraction=.3, seed=42)


def test_quarantine_unsure_and_teacher_exposure_preserve_labels(manual):
    rows = [row(1), row(2), row(3), row(4), row(5)]
    rows[0]['review_note'] = 'conflict; retain pending review'
    rows[1]['unsure'] = True
    rows[2]['used_in_teacher_fewshot'] = True
    before = copy.deepcopy(rows)
    eligible, excluded = manual.select_rows(rows)
    assert [r['id'] for r in eligible] == ['3', '4', '5']
    assert [r['exclusion'] for r in excluded] == ['audit_conflict', 'uncertain']
    assert eligible[0]['exposure'] == 'seen_by_teacher'
    assert eligible[1]['exposure'] == 'not_in_teacher_fewshot'
    assert all(r['sentiment'] is None and r['sentiment_loss_mask'] is False for r in eligible)
    assert rows == before


def test_fresh_output_rejects_existing_and_source_ancestors_descendants(manual, tmp_path):
    source = tmp_path / 'current' / 'model'
    source.mkdir(parents=True)
    (source / 'heads.pt').write_bytes(b'keep')
    for output in [source, source.parent, source / 'run', tmp_path]:
        with pytest.raises(ValueError):
            manual.reserve_output(output, [source])
    empty = tmp_path / 'existing-empty'
    empty.mkdir()
    with pytest.raises(ValueError):
        manual.reserve_output(empty, [source])
    fresh = tmp_path / 'new-run'
    manual.reserve_output(fresh, [source])
    assert fresh.is_dir()
    with pytest.raises(ValueError):
        manual.reserve_output(fresh, [source])
    assert (source / 'heads.pt').read_bytes() == b'keep'


def write_fixture(tmp_path):
    labels = tmp_path / 'labels.json'
    original = tmp_path / 'original.json'
    enriched = tmp_path / 'enriched.json'
    audit = tmp_path / 'candidates.json'
    parquet = tmp_path / 'wb.parquet'
    labels.write_text(json.dumps({'labels': [dict(id=str(i), need=i == 1,
                       reasons=['defect'] if i == 1 else [], unsure=i == 4) for i in range(1, 5)]}))
    original.write_text(json.dumps({'items': [dict(id=i, text=f'text {i}') for i in [1, 2]]}))
    enriched.write_text(json.dumps({'items': [dict(id=i, text=f'text {i}') for i in [3, 4]]}))
    audit.write_text(json.dumps({'rows': [dict(id=str(i), text=f'text {i}', need=i == 1,
        reasons=['defect'] if i == 1 else [], review_note='conflict' if i == 2 else None,
        used_in_teacher_fewshot=i == 2) for i in range(1, 5)]}))
    pd.DataFrame({'nmId': [100, 101, 102, 103, 104], 'text': ['x'] + [f'text {i}' for i in range(1, 5)]}).to_parquet(parquet)
    return labels, [original, enriched], audit, parquet


def test_local_loader_merges_enriched_and_strictly_resolves_ids(manual, tmp_path):
    labels, items, audit, parquet = write_fixture(tmp_path)
    inputs = [labels, *items, audit, parquet]
    before = [p.read_bytes() for p in inputs]
    rows, provenance = manual.load_manual(labels, items, audit, parquet)
    assert [r['nmId'] for r in rows] == [101, 102, 103, 104]
    assert rows[0]['need'] is True and rows[0]['reasons'] == ['defect']
    assert rows[1]['review_note'] == 'conflict'
    assert rows[1]['used_in_teacher_fewshot'] is True
    assert rows[3]['unsure'] is True
    assert all(p['sha256'] == hashlib.sha256(Path(p['path']).read_bytes()).hexdigest() for p in provenance)
    assert before == [p.read_bytes() for p in inputs]
    pd.DataFrame({'nmId': [100], 'text': ['x']}).to_parquet(parquet)
    with pytest.raises(ValueError, match='missing.*ID'):
        manual.load_manual(labels, items, audit, parquet)


def test_prepare_only_has_manifests_without_model_or_source_mutation(manual, tmp_path):
    labels, items, audit, parquet = write_fixture(tmp_path)
    model = tmp_path / 'source-model'
    model.mkdir()
    (model / 'heads.pt').write_bytes(b'no loading during preparation')
    output = tmp_path / 'prepared'
    argv = ['--labels', str(labels), '--items', *map(str, items), '--audit', str(audit),
            '--parquet', str(parquet), '--init', str(model), '--output', str(output), '--prepare-only']
    manual.main(argv)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'prepared_only'
    assert manifest['independent_benchmark'] is False
    assert manifest['counts'] == {'train': 1, 'development': 1, 'excluded': 2}
    assert manifest['class_counts']['development'] == {'positive': 0, 'negative': 1}
    assert any('development has no positive' in warning for warning in manifest['warnings'])
    assert len(manifest['provenance']) == 6
    assert not (output / 'model').exists()
    assert (model / 'heads.pt').read_bytes() == b'no loading during preparation'
    with pytest.raises(ValueError):
        manual.main(argv)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.config = type('Config', (), {'hidden_size': 4})()

    def forward(self, input_ids, **kw):
        return type('Output', (), {'last_hidden_state': self.embedding(input_ids)})()


def test_real_cpu_step_updates_supervised_heads_and_freezes_sentiment(manual):
    torch.manual_seed(7)
    model = manual.ManualStudent(TinyEncoder())
    original_sent = {k: v.detach().clone() for k, v in model.sent.state_dict().items()}
    old_need = model.need.weight.detach().clone()
    old_reasons = model.reasons.weight.detach().clone()
    outputs = model(input_ids=torch.tensor([[1, 2], [3, 4]]))
    # Sentinel NaNs prove no sentiment target/loss is consumed.
    loss = manual.manual_loss(outputs, torch.tensor([1., 0.]),
                              torch.tensor([[1.] + [0.] * 7, [0.] * 8]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert not torch.equal(old_need, model.need.weight)
    assert not torch.equal(old_reasons, model.reasons.weight)
    assert all(p.grad is None and not p.requires_grad for p in model.sent.parameters())
    assert all(torch.equal(v, model.sent.state_dict()[k]) for k, v in original_sent.items())
    n, r, s = outputs
    loss_with_nan = manual.manual_loss((n, r, torch.full_like(s, float('nan'))),
                                      torch.tensor([1., 0.]), torch.zeros(2, 8))
    assert torch.isfinite(loss_with_nan)

def test_manual_runner_is_available():
    assert importlib.util.find_spec('train_manual') is not None, 'manual training runner is missing'

def test_stale_or_missing_audit_fails_closed(manual, tmp_path):
    labels, items, audit, parquet = write_fixture(tmp_path)
    original = json.loads(audit.read_text())
    bad = copy.deepcopy(original)
    bad['rows'][0]['need'] = False
    audit.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='stale audit'):
        manual.load_manual(labels, items, audit, parquet)
    original['rows'] = original['rows'][1:]
    audit.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='missing audit ID'):
        manual.load_manual(labels, items, audit, parquet)


def test_local_cpu_runner_loads_saved_heads_and_writes_only_new_run(manual, tmp_path):
    from transformers import BertConfig, BertModel, BertTokenizerFast
    labels, items, audit, parquet = write_fixture(tmp_path)
    source = tmp_path / 'saved-student'
    source.mkdir()
    vocab = source / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ntext\n1\n2\n3\n4\n')
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(source)
    encoder = BertModel(BertConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=16))
    model = manual.ManualStudent(encoder)
    encoder.save_pretrained(source)
    heads = {k: v for k, v in model.state_dict().items() if not k.startswith('enc.')}
    torch.save(heads, source / 'heads.pt')
    input_files = [labels, *items, audit, parquet, *source.iterdir()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}
    output = tmp_path / 'trained-run'
    manual.main(['--labels', str(labels), '--items', *map(str, items), '--audit', str(audit),
                 '--parquet', str(parquet), '--init', str(source), '--output', str(output),
                 '--epochs', '1', '--batch', '2', '--threads', '1', '--device', 'cpu'])
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'completed_development_training'
    saved = torch.load(output / 'model/heads.pt', weights_only=True)
    assert torch.equal(saved['sent.weight'], heads['sent.weight'])
    assert torch.equal(saved['sent.bias'], heads['sent.bias'])
    assert not torch.equal(saved['need.weight'], heads['need.weight'])
    assert not torch.equal(saved['reasons.weight'], heads['reasons.weight'])
    metrics = json.loads((output / 'development_metrics.json').read_text())
    assert metrics['independent_benchmark'] is False
    development = metrics['history'][0]['development']
    assert development['all']['n'] == 1
    assert set(development) == {'all', 'seen_by_teacher', 'not_in_teacher_fewshot'}
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}

def test_normalization_groups_audit_punctuation_and_yo_variants(manual):
    rows = [row(1, 10, 'Корм норм. Собака ест.'), row(2, 20, 'Корм норм, собака ест!'),
            row(3, 30, 'Очень сильно большемерят (('), row(4, 40, 'очень сильно большемерят'),
            row(5, 50, 'ТЁПЛЫЙ'), row(6, 60, 'теплый'), row(7, 70, 'иное')]
    splits = manual.group_split(rows, val_fraction=.3, seed=42)
    side = {r['id']: name for name, rs in splits.items() for r in rs}
    assert side['1'] == side['2']
    assert side['3'] == side['4']
    assert side['5'] == side['6']

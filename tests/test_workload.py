import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def workload():
    assert importlib.util.find_spec('seller_workload') is not None, 'workload calculator missing'
    import seller_workload
    return seller_workload


def test_threshold_metrics_use_selected_count_and_keep_errors_visible():
    mod = workload()
    rows = [dict(id='1', gold_need=True, score=.9, fewshot=True),
            dict(id='2', gold_need=False, score=.8, fewshot=False),
            dict(id='3', gold_need=True, score=.2, fewshot=False),
            dict(id='4', gold_need=False, score=.1, fewshot=False)]
    result = mod.calculate(rows, [.5], minutes_per_review=2)
    all_rows = result['all']['thresholds'][0]
    assert all_rows['selected'] == 2 and all_rows['reduction_percent'] == 50
    assert all_rows['fp'] == 1 and all_rows['fn'] == 1
    assert all_rows['precision'] == all_rows['recall'] == all_rows['f1'] == .5
    assert all_rows['selected_per_100'] == 50
    assert all_rows['saved_minutes_scenario'] == 4
    excluded = result['excluding_teacher_fewshot']['thresholds'][0]
    assert excluded['selected'] == 1 and excluded['fp'] == excluded['fn'] == 1
    assert excluded['precision'] == excluded['recall'] == 0


@pytest.mark.parametrize('patch', [{'score': float('nan')}, {'score': 1.2}, {'score': True},
    {'gold_need': 'false'}, {'fewshot': 1}, {'id': ''}])
def test_invalid_saved_scores_and_labels_fail_closed(patch):
    mod = workload()
    row = dict(id='1', text='x', gold_need=False, score=.3, gold_reasons=[], fewshot=False,
               predicted_reasons=[], error_035='false_positive')
    row.update(patch)
    with pytest.raises(ValueError):
        mod.validate_rows([row])


def test_costs_are_only_explicit_scenarios_and_thresholds_are_validated():
    mod = workload()
    rows = [dict(id='1', gold_need=False, score=.5, fewshot=False)]
    record = mod.calculate(rows, [.5])['all']['thresholds'][0]
    assert record['selected'] == 1
    assert 'saved_minutes_scenario' not in record
    for thresholds in [[-1], [float('nan')], [], [True]]:
        with pytest.raises(ValueError):
            mod.calculate(rows, thresholds)
    with pytest.raises(ValueError):
        mod.calculate(rows, [.5], minutes_per_review=-1)


def test_saved_snapshot_rejects_hash_changes_and_matches_audit(tmp_path, monkeypatch):
    # Same population sizes as the historical contract, all scores synthetic.
    # This tests cross-file consistency without depending on ignored model/data.
    mod = workload()
    rows = []
    for i in range(101):
        positive = i in [0, 1, 8, 9, 10, 11, 12, 13, 14]
        rows.append(dict(id=str(i), text=f'fictional review {i}', gold_need=positive,
                         score=.9 if positive else .1, gold_reasons=[], fewshot=i < 8,
                         predicted_reasons=[], error_035=None))
    labels = [dict(id=r['id'], need=r['gold_need'], reasons=[], unsure=False) for r in rows]
    candidates = [dict(id=r['id'], text=r['text'], need=r['gold_need'], reasons=[],
                       used_in_teacher_fewshot=r['fewshot']) for r in rows]
    perfect = {'0.35': {'f1': 1., 'precision': 1., 'recall': 1., 'fp': 0, 'fn': 0},
               '0.85': {'f1': 1., 'precision': 1., 'recall': 1., 'fp': 0, 'fn': 0}}
    summary = {'human_rows': 101, 'fewshot_ids_in_human_evaluation': [str(i) for i in range(8)],
               'fresh_cpu_metrics': {'all': {'n': 101, 'positives': 9, 'thresholds': perfect},
                   'excluding_fewshot': {'n': 93, 'positives': 7, 'thresholds': perfect}}}
    files = {'data/audit/human_predictions.json': rows, 'data/audit/summary.json': summary,
             'data/audit/human_training_candidates.json': {'rows': candidates},
             'labeling/needs_reply_labels.json': {'labels': labels}}
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
    for name in ['data/audit/audit.py', 'student_curve.py', 'llm_probe.py']:
        (tmp_path / name).write_text('# synthetic supported source fixture')
    pinned = {name: mod.fingerprint(tmp_path / name) for name in
              ['data/audit/human_predictions.json', 'data/audit/audit.py', 'student_curve.py', 'llm_probe.py']}
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    monkeypatch.setattr(mod, 'KNOWN_HASHES', pinned)
    snapshot = mod.load_snapshot()
    result = mod.calculate(snapshot['rows'], [.35, .85])
    assert result['all']['n'] == 101
    assert result['all']['positives'] == 9
    assert result['all']['thresholds'][0]['selected'] == 9
    assert result['all']['thresholds'][1]['selected'] == 9
    assert result['excluding_teacher_fewshot']['n'] == 93
    assert snapshot['historical_model_hash_available'] is False
    assert all(len(item['sha256']) == 64 for item in snapshot['provenance'])
    prediction_path = tmp_path / 'data/audit/human_predictions.json'
    prediction_path.write_text('[]')
    with pytest.raises(ValueError, match='hash'):
        mod.load_snapshot()

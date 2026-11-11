"""Describe seller review workload from the immutable, already inspected audit.

python seller_workload.py
python seller_workload.py --minutes-per-review 2  # explicit hypothetical cost

This never runs inference, tunes a threshold, or changes model/labels.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REASONS = {'defect', 'refund', 'delivery', 'question', 'safety', 'misinfo', 'strong_negative', 'wb_logistics'}
# Pinned historical audit artifacts/code. A changed snapshot must be reviewed,
# not silently treated as having the same score meaning or evaluation history.
KNOWN_HASHES = {
    "data/audit/human_predictions.json": "55b90617084905f8ddae8cb22fee6c4d27d13de4a1e1cd91477afec7d5559eec",
    "data/audit/audit.py": "2b4497c5625309bcaf97d4dde1493992c2987773923fcf8680f66b1ecf53135c",
    "student_curve.py": "a3d30656098b0d7dd45feaef54dc5ac5354727c2ff10a26c55cbe6b821c27e3b",
    "llm_probe.py": "948c563fe1e7692980fbfc51c5a91d3dd077f66eb7ff57db443cb5d5a387a4e9"
}


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path, expected):
    actual = fingerprint(path)
    if actual != expected:
        raise ValueError(f'Unsupported source hash for {path}; review the audit snapshot before regenerating workload reports')
    return actual


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('saved predictions must be a nonempty list')
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('saved prediction must be an object')
        key = row.get('id')
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError('saved prediction IDs must be unique nonempty strings')
        seen.add(key)
        score = row.get('score')
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'invalid need score for ID {key}')
        if type(row.get('gold_need')) is not bool or type(row.get('fewshot')) is not bool:
            raise ValueError(f'gold_need and fewshot must be booleans for ID {key}')
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError(f'missing review text for ID {key}')
        for field in ['gold_reasons', 'predicted_reasons']:
            value = row.get(field)
            if not isinstance(value, list) or any(reason not in REASONS for reason in value):
                raise ValueError(f'invalid {field} for ID {key}')
        error = ('false_positive' if score >= .35 and not row['gold_need']
                 else 'false_negative' if score < .35 and row['gold_need'] else None)
        if row.get('error_035') != error:
            raise ValueError(f'inconsistent saved error_035 for ID {key}')
    return rows


def threshold_record(rows, threshold, minutes_per_review):
    n = len(rows)
    selected = sum(r['score'] >= threshold for r in rows)
    tp = sum(r['score'] >= threshold and r['gold_need'] for r in rows)
    fp = selected - tp
    fn = sum(r['score'] < threshold and r['gold_need'] for r in rows)
    precision = tp / selected if selected else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    result = {'threshold': threshold, 'selected': selected, 'not_selected': n - selected,
              'reduction_percent': 100 * (n - selected) / n if n else 0.,
              'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
              'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
              'selected_per_100': 100 * selected / n if n else 0.,
              'fp_per_100': 100 * fp / n if n else 0., 'fn_per_100': 100 * fn / n if n else 0.}
    if minutes_per_review is not None:
        result.update(baseline_minutes_scenario=n * minutes_per_review,
                      selected_minutes_scenario=selected * minutes_per_review,
                      saved_minutes_scenario=(n - selected) * minutes_per_review)
    return result


def calculate(rows, thresholds, minutes_per_review=None):
    if not thresholds or any(type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise ValueError('thresholds must be finite numbers between zero and one')
    if minutes_per_review is not None and (type(minutes_per_review) not in (int, float)
            or not math.isfinite(minutes_per_review) or minutes_per_review <= 0):
        raise ValueError('minutes per review must be a positive explicit scenario')
    result = {}
    for name, subset in [('all', rows), ('excluding_teacher_fewshot', [r for r in rows if not r['fewshot']])]:
        result[name] = {'n': len(subset), 'positives': sum(r['gold_need'] for r in subset),
                        'thresholds': [threshold_record(subset, float(t), minutes_per_review) for t in thresholds]}
    return result


def load_snapshot():
    for relative, digest in KNOWN_HASHES.items():
        verify_hash(ROOT / relative, digest)
    prediction_path = ROOT / 'data/audit/human_predictions.json'
    summary_path = ROOT / 'data/audit/summary.json'
    labels_path = ROOT / 'labeling/needs_reply_labels.json'
    candidates_path = ROOT / 'data/audit/human_training_candidates.json'
    rows = validate_rows(read_json(prediction_path))
    summary = read_json(summary_path)
    labels = read_json(labels_path)['labels']
    candidates = read_json(candidates_path)['rows']
    gold = {str(r['id']): r for r in labels}
    audited = {str(r['id']): r for r in candidates}
    if len(gold) != len(labels) or len(audited) != len(candidates):
        raise ValueError('duplicate source label/candidate IDs')
    if {r['id'] for r in rows} != set(gold) or set(gold) != set(audited):
        raise ValueError('saved predictions, audit candidates and current human label IDs differ')
    if len(rows) != 101 or summary.get('human_rows') != 101:
        raise ValueError('this saved audit describes exactly 101 inspected manual labels')
    fewshot = {r['id'] for r in rows if r['fewshot']}
    if len(fewshot) != 8 or fewshot != set(summary['fewshot_ids_in_human_evaluation']):
        raise ValueError('teacher fewshot exposure does not match the audit')
    for row in rows:
        label, candidate = gold[row['id']], audited[row['id']]
        if label.get('unsure') or type(label.get('need')) is not bool:
            raise ValueError('saved evaluation no longer matches certain human labels')
        if (row['gold_need'] != label['need'] or set(row['gold_reasons']) != set(label['reasons'])
                or candidate.get('text') != row['text'] or candidate.get('need') != row['gold_need']
                or set(candidate.get('reasons', [])) != set(row['gold_reasons'])
                or candidate.get('used_in_teacher_fewshot') != row['fewshot']):
            raise ValueError(f'current labels/audit disagree with saved prediction ID {row["id"]}')
    measured = calculate(rows, [.35, .85])
    for group, summary_group in [('all', 'all'), ('excluding_teacher_fewshot', 'excluding_fewshot')]:
        expected = summary['fresh_cpu_metrics'][summary_group]
        actual = measured[group]
        if actual['n'] != expected['n'] or actual['positives'] != expected['positives']:
            raise ValueError('saved audit population counts differ')
        for metric in actual['thresholds']:
            old = expected['thresholds'][str(metric['threshold'])]
            if any(round(metric[k], 3) != old[k] for k in ['f1', 'precision', 'recall', 'fp', 'fn']):
                raise ValueError('saved audit metrics do not match score >= threshold semantics')
    sources = [prediction_path, summary_path, labels_path, candidates_path,
               ROOT / 'data/audit/audit.py', ROOT / 'student_curve.py', ROOT / 'llm_probe.py']
    sources.extend(p for p in sorted((ROOT / 'data/student/model').rglob('*')) if p.is_file())
    return {'rows': rows,
            'provenance': [{'path': str(p.relative_to(ROOT)), 'sha256': fingerprint(p)} for p in sources],
            'historical_model_hash_available': False,
            'source_validation': 'Pinned saved predictions and inference code; IDs, targets, fewshot flags and 0.35/0.85 aggregate metrics cross-checked. Model hashes describe current files, not an independently verified historical inference checkpoint.',
            'quarantined_label_count': sum(bool(r.get('review_note')) for r in candidates)}


def render_report(report):
    lines = ['# Нагрузка продавца: сценарии ручного просмотра', '',
        'Источник — сохранённые оценки модели на 101 вручную отобранном и уже просмотренном отзыве (9 положительных меток «нужен человек»). Это описательный расчёт на разработческой выборке, не независимый тест и не калибровка вероятностей.', '',
        'score — sigmoid выхода головы need сохранённого студента, до порога. Отзыв выбирается при score ≥ порога; score не означает подтверждённую вероятность. Причины и тональность здесь не влияют на отбор.', '',
        'Снижение нагрузки = доля отзывов, не выбранных для ручного просмотра, относительно просмотра всей этой выборки. Лишние срабатывания (FP) и пропуски нужного человека (FN) показаны рядом. Нулевые P/R/F1 при отсутствии соответствующего знаменателя обозначаются 0.', '',
        'Пять спорных меток аудита сохранены без исправлений и остаются в этом расчёте. В подготовке обучения они помещены в карантин.', '',
        'Пересчёт «на 100» — только арифметическое масштабирование того же вручную отобранного набора. Он не оценивает естественную частоту проблем, будущую нагрузку магазина или экономию в рабочем потоке.', '']
    for name, title in [('all', 'Все 101 отзыва'), ('excluding_teacher_fewshot', 'Без 8 примеров, показанных учителю')]:
        group = report['groups'][name]
        lines += [f'## {title}', '', f"Отзывов: {group['n']}; меток «нужен человек»: {group['positives']}.", '',
                  '| Порог | На просмотр | Снижение | FP | FN | P | R | F1 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in group['thresholds']:
            lines.append(f"| {r['threshold']:.2f} | {r['selected']} | {r['reduction_percent']:.1f}% | {r['fp']} | {r['fn']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
        lines += ['', 'Арифметический пересчёт на 100 отзывов с тем же составом:', '',
                  '| Порог | На просмотр / 100 | FP / 100 | FN / 100 |', '|---|---:|---:|---:|']
        for r in group['thresholds']:
            lines.append(f"| {r['threshold']:.2f} | {r['selected_per_100']:.1f} | {r['fp_per_100']:.1f} | {r['fn_per_100']:.1f} |")
        lines.append('')
    minutes = report['minutes_per_review_scenario']
    if minutes is None:
        lines += ['Время и деньги не оценивались: измеренных затрат на просмотр нет. Для явно заданного сценария времени используйте `--minutes-per-review N`.', '']
    else:
        lines += [f'## Явно заданный сценарий времени: {minutes:g} мин/отзыв', '',
            'Это предположение пользователя: каждый просмотр занимает одинаковое время, сравнение — с просмотром всех отзывов. Затраты на исправление ошибок и обработку пропусков не включены. Измерений времени и оценки денег нет.', '',
            '| Группа | Порог | Базовое время, мин | Отобранные, мин | Разница, мин |', '|---|---:|---:|---:|---:|']
        for name, group in report['groups'].items():
            for r in group['thresholds']:
                lines.append(f"| {name} | {r['threshold']:.2f} | {r['baseline_minutes_scenario']:g} | {r['selected_minutes_scenario']:g} | {r['saved_minutes_scenario']:g} |")
        lines.append('')
    lines += ['## Происхождение данных', '',
        'Проверены закреплённые SHA-256 файла оценок и кода аудита/предсказания, соответствие ID и меток текущей ручной разметке, признаки few-shot и исходные метрики при 0.35/0.85. Хеши входов записаны в seller_workload.json.', '',
        'Исходный аудит не сохранил хеш модели в момент предсказания. Хеши модели в JSON относятся к текущим локальным файлам и не доказывают задним числом происхождение исторических оценок. Повторное предсказание не выполнялось.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[.05, .1, .2, .35, .5, .65, .85, .95])
    parser.add_argument('--minutes-per-review', type=float)
    args = parser.parse_args(argv)
    snapshot = load_snapshot()
    report = {key: value for key, value in snapshot.items() if key != 'rows'}
    report.update(independent_benchmark=False, calibrated_probability=False,
                  score_semantics='sigmoid need logit; selected when score >= threshold',
                  population='Manually selected, already inspected 101 reviews; no inference about natural prevalence.',
                  per_100='Arithmetic rescaling of this same selected set only.',
                  minutes_per_review_scenario=args.minutes_per_review,
                  groups=calculate(snapshot['rows'], args.thresholds, args.minutes_per_review))
    output = ROOT / 'reports'
    (output / 'seller_workload.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'seller_workload.md').write_text(render_report(report), encoding='utf-8')
    print(f'Wrote {output / "seller_workload.md"} and aggregate JSON; no model inference or label changes.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

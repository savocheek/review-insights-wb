"""Bounded offline CSV analysis; frozen local student, no remote calls or training."""
import csv
from collections import Counter
from functools import lru_cache
from html import escape
import io
import math
from pathlib import Path
from threading import Lock
from uuid import uuid4
from annotation_store import REASONS

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / 'data/csv_runs'
MODEL_DIR = ROOT / 'data/student/model'
MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 5000
REASON_KEYS = ('defect', 'refund', 'delivery', 'question', 'safety', 'misinfo',
               'strong_negative', 'wb_logistics')
SENTIMENTS = ('негатив', 'нейтрально', 'позитив', 'смешанный')
OUTPUT_COLUMNS = ('model_status', 'model_need_score', 'model_candidate', 'model_reasons',
                  'model_sentiment', 'model_need_threshold', 'model_reason_threshold',
                  *('model_score_' + key for key in REASON_KEYS))
_INFERENCE_LOCK = Lock()


def _read_csv(path):
    try:
        source = Path(path)
        if source.stat().st_size > MAX_BYTES:
            raise ValueError('CSV превышает ограничение 20 МБ')
        raw = source.read_bytes()
    except (OSError, TypeError) as error:
        raise ValueError('Не удалось прочитать CSV: загрузите файл') from error
    if not raw or len(raw) > MAX_BYTES or b'\x00' in raw:
        raise ValueError('Пустой, слишком большой или бинарный файл CSV')
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        try:
            text = raw.decode('cp1251')
        except UnicodeDecodeError as error:
            raise ValueError('Нужна кодировка UTF-8 или Windows-1251') from error
    try:
        first = text.splitlines()[0]
        try:
            delimiter = csv.Sniffer().sniff(first, delimiters=',;\t').delimiter
        except csv.Error:
            delimiter = ','
        csv.field_size_limit(MAX_BYTES)
        reader = csv.reader(io.StringIO(text, newline=''), delimiter=delimiter, strict=True)
        headers = next(reader)
        if not headers or any(not h.strip() for h in headers) or len(set(headers)) != len(headers):
            raise ValueError('Нужны непустые уникальные заголовки столбцов')
        if any(h in OUTPUT_COLUMNS for h in headers):
            raise ValueError('Вход уже содержит служебные model_* столбцы; загрузите исходный CSV')
        rows = []
        for row in reader:
            if len(rows) >= MAX_ROWS:
                raise ValueError('Разрешено не более 5000 строк данных')
            if not row:
                row = [''] * len(headers)
            if len(row) != len(headers):
                raise ValueError(f'Строка данных {len(rows) + 1}: число столбцов отличается от заголовка')
            rows.append(row)
        if not rows:
            raise ValueError('CSV содержит заголовок, но нет строк данных')
        return headers, rows
    except (csv.Error, StopIteration, IndexError) as error:
        raise ValueError('Некорректный CSV: проверьте разделители и кавычки') from error


def inspect_csv(path):
    """Return validated original column names without loading the model."""
    return _read_csv(path)[0]


@lru_cache(maxsize=1)
def _load_local_model():
    import torch
    from transformers import AutoModel, AutoTokenizer
    from train_manual import ManualStudent
    torch.set_num_threads(min(8, max(4, __import__('os').cpu_count() or 4)))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = ManualStudent(AutoModel.from_pretrained(str(MODEL_DIR), local_files_only=True))
    state = torch.load(MODEL_DIR / 'heads.pt', map_location='cpu', weights_only=True)
    expected = {f'{head}.{part}' for head in ('need', 'reasons', 'sent') for part in ('weight', 'bias')}
    if not isinstance(state, dict) or set(state) != expected:
        raise ValueError('Локальная модель: несовместимые сохранённые головы')
    for head in ('need', 'reasons', 'sent'):
        getattr(model, head).load_state_dict({part: state[f'{head}.{part}'] for part in ('weight', 'bias')}, strict=True)
    model.to('cpu').eval().requires_grad_(False)
    return model, tokenizer


def predict_local(texts):
    """CPU batches with deterministic eval, local-only weights, all three heads."""
    import torch
    results = []
    with _INFERENCE_LOCK:
        model, tokenizer = _load_local_model()
        with torch.inference_mode():
            for start in range(0, len(texts), 16):
                tokens = tokenizer(texts[start:start + 16], truncation=True, max_length=256,
                                   padding=True, return_tensors='pt')
                need, reasons, sentiment = model(**tokens)
                for n, r, s in zip(need.sigmoid().tolist(), reasons.sigmoid().tolist(), sentiment.argmax(-1).tolist()):
                    results.append({'need_score': n, 'reason_scores': dict(zip(REASON_KEYS, r)),
                                    'sentiment': SENTIMENTS[s]})
    return results


def _score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Модель вернула недопустимый балл')
    return float(value)


def _spreadsheet_safe(value):
    # Quoting CSV cells alone does not prevent spreadsheet formula execution.
    return "'" + value if value.lstrip().startswith(('=', '+', '-', '@')) or value.startswith(('\t', '\r', '\n')) else value


def run_analysis(path, text_column, threshold=.85, predictor=None):
    """Return fresh (enriched UTF8-BOM CSV, escaped standalone HTML) output paths."""
    headers, rows = _read_csv(path)
    if text_column not in headers:
        raise ValueError('Выберите существующий столбец с текстом отзыва')
    threshold = _score(threshold)
    column = headers.index(text_column)
    positions = [i for i, row in enumerate(rows) if row[column].strip()]
    texts = [rows[i][column] for i in positions]
    predictions = (predictor or predict_local)(texts) if texts else []
    if len(predictions) != len(texts):
        raise ValueError('Количество результатов модели не совпадает с количеством отзывов')
    analyzed = {}
    for i, prediction in zip(positions, predictions):
        n = _score(prediction.get('need_score'))
        r = prediction.get('reason_scores')
        if not isinstance(r, dict) or set(r) != set(REASON_KEYS) or prediction.get('sentiment') not in SENTIMENTS:
            raise ValueError('Модель вернула несовместимые причины или тональность')
        scores = [_score(r[key]) for key in REASON_KEYS]
        analyzed[i] = ['analyzed', f'{n:.6f}', str(n >= threshold).lower(),
                       ';'.join(k for k, score in zip(REASON_KEYS, scores) if score >= .5),
                       prediction['sentiment'], f'{threshold:.2f}', '0.50', *[f'{s:.6f}' for s in scores]]
    run = RUNS_DIR / uuid4().hex
    run.mkdir(parents=True, exist_ok=False)
    csv_path, html_path = run / 'reviews_enriched.csv', run / 'report.html'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([_spreadsheet_safe(h) for h in headers] + list(OUTPUT_COLUMNS))
        for i, row in enumerate(rows):
            output = analyzed.get(i, ['blank_text'] + [''] * (len(OUTPUT_COLUMNS) - 1))
            writer.writerow([_spreadsheet_safe(cell) for cell in row] + output)
    candidates = sum(output[2] == 'true' for output in analyzed.values())
    intro = (f'Строк: {len(rows)}. Проанализировано: {len(analyzed)}. Пустых: {len(rows) - len(analyzed)}. '
             f'Кандидатов на проверку человеком: {candidates}. Порог отбора: {threshold:.2f}; причины: 0.50. '
             'Баллы модели — не калиброванные вероятности. Кандидат модели не является решением человека '
             'о необходимости ответа. Причины и тональность — прогнозы модели. '
             'Тексты длиннее 256 токенов обрезаются для модели. Темы и цитаты не проверялись и не выводятся.')
    entries = []
    for i, row in enumerate(rows):
        output = analyzed.get(i)
        titles = ', '.join(REASONS[k] for k in output[3].split(';') if k) if output else ''
        selected = 'да' if output and output[2] == 'true' else 'нет'
        details = (f'Балл модели: {output[1]}; кандидат: {selected}; '
                   f'причины модели: {titles or "нет"}; тональность модели: {output[4]}') if output else 'Пустой текст: анализ пропущен'
        entries.append(f'<article><h2>Строка {i + 1}</h2><pre>{escape(row[column])}</pre><p>{escape(details)}</p></article>')
    sentiment_counts = Counter(output[4] for output in analyzed.values())
    reason_counts = Counter(reason for output in analyzed.values() for reason in output[3].split(';') if reason)

    def count_table(title, counts, keys):
        table_rows = ''.join(f'<tr><td>{escape(REASONS.get(key, key))}</td><td>{counts[key]}</td>'
                             f'<td>{100 * counts[key] / len(analyzed) if analyzed else 0:.1f}%</td></tr>' for key in keys)
        return (f'<h2>{title}</h2><table><thead><tr><th>Прогноз модели</th><th>Отзывов</th>'
                '<th>Доля проанализированных</th></tr></thead><tbody>' + table_rows + '</tbody></table>')

    aggregates = ('<section><p>Доли рассчитаны среди непустых проанализированных отзывов: '
                  f'{len(analyzed)}. Причин у одного отзыва может быть несколько, доли причин пересекаются.</p>'
                  + count_table('Тональность: прогноз модели', sentiment_counts, SENTIMENTS)
                  + count_table('Причины: прогноз модели', reason_counts, REASON_KEYS) + '</section>')
    prioritized = sorted((i for i in analyzed if analyzed[i][2] == 'true'),
                         key=lambda i: (-float(analyzed[i][1]), i))
    candidate_section = ('<section id="candidates"><h2>Приоритетная очередь на проверку человеком</h2>'
                         '<p>Кандидаты модели по убыванию балла; номер соответствует исходной строке данных CSV.</p>'
                         + (''.join(entries[i] for i in prioritized) if prioritized else '<p>Кандидатов выше порога нет.</p>')
                         + '</section>')
    html_path.write_text('<!doctype html><html lang="ru"><meta charset="utf-8">'
                        '<meta name="viewport" content="width=device-width,initial-scale=1">'
                        '<title>Анализ отзывов CSV</title><style>body{font:16px/1.5 system-ui;max-width:1000px;'
                        'margin:32px auto;padding:0 20px;color:#17233b;background:#f5f7fa}'
                        'article{padding:20px;margin:20px 0;background:white;border:1px solid #ddd;border-radius:12px}'
                        'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}'
                        'table{border-collapse:collapse;background:white;width:100%;margin:12px 0}'
                        'td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}</style>'
                        '<h1>Анализ отзывов CSV</h1><p>' + escape(intro) + '</p>' + aggregates + candidate_section
                        + '<details><summary>Все строки в исходном порядке</summary>' + ''.join(entries)
                        + '</details></html>', encoding='utf-8')
    return str(csv_path), str(html_path)

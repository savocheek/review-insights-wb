# Анализ отзывов для продавцов маркетплейсов

Сервис для продавца на Wildberries: по отзывам на его товары показывает **сводку частых проблем** и выделяет
**отзывы, на которые стоит ответить человеку**. Автоответов нет — продавец отвечает сам, сервис только
подсказывает, где это важно.

Статус: этап 1 — данные. Модель тональности берётся из проекта
[sentiment-analysis-ru](https://github.com/savocheek/sentiment-analysis-ru) (ruBert-base, дистиллированная
из ruRoberta-large на отзывах Яндекс Карт) и дообучается на товарах.

## Данные

- **Wildberries:** [nyuuzyou/wb-feedbacks](https://huggingface.co/datasets/nyuuzyou/wb-feedbacks) (CC0): товар
  (nmId), оценка 1–5★, текст, ответ продавца. Берутся два шарда parquet-конвертации HF (~2.6 млн отзывов,
  ~320 МБ), не весь датасет (10 ГБ).
- **Независимый тест:** [ai-forever/ru-reviews-classification](https://huggingface.co/datasets/ai-forever/ru-reviews-classification)
  (RuReviews, Apache-2.0): отзывы на товары с тремя классами тональности, англоязычные отзывы убраны.

После очистки — 2.29 млн отзывов на 204 тыс. товаров. 84% отзывов позитивные (4–5★), нейтральных (3★) 6.6%.
Продавец ответил на 43.5% отзывов, и 64% этих ответов — дословно повторяющиеся шаблоны («Спасибо за отзыв!»).
Подробно: [reports/data_stats.md](reports/data_stats.md).

**Сплиты по товарам.** Каждый товар (nmId) целиком лежит в одном сплите (80/10/10 по хэшу nmId). Так модель не
видит в обучении отзывов на товары из теста — иначе она запоминала бы сами товары, и оценка была бы завышена.

## Запуск

```bash
pip install -r requirements.txt
python prepare_data.py   # → data/wb.parquet, data/rureviews_test.parquet, reports/data_stats.md
```

## Лицензии данных

wb-feedbacks — CC0 1.0, RuReviews — Apache-2.0. Сами данные в репозиторий не кладутся, скрипт скачивает их с
Hugging Face.

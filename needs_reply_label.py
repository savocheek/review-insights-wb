"""Blind local annotation UI for labeling/*.json; run python needs_reply_label.py.

Need and problem reasons are independent human decisions. New items have no
preselected decision or reasons. Human labels resume from canonical JSON.
"""
import argparse
from pathlib import Path

import gradio as gr

from annotation_store import AnnotationSession, REASONS

ROOT = Path(__file__).resolve().parent
ITEMS = ROOT / 'labeling/needs_reply_items_enriched.json'
OUT = ROOT / 'labeling/needs_reply_labels.json'
YES = 'Нужен человек'
NO = 'Хватит шаблона'


def build_app(items_path=ITEMS, labels_path=OUT):
    session = AnnotationSession(items_path, labels_path)

    def show(index):
        review = session.show(index)
        need = None if review['need'] is None else YES if review['need'] else NO
        status = f"{index + 1} / {review['total']} · размечено в этой партии: {review['completed']} / {review['total']}"
        if review['completed'] == review['total']:
            status += ' · Всё размечено'
        return index, status, review['text'], need, review['reasons'], review['unsure'], ''

    def save(index, decision, reasons, unsure):
        if decision not in (YES, NO):
            raise gr.Error('Выберите: нужен человек или хватит шаблона')
        try:
            session.save(index, decision == YES, reasons, unsure)
        except (ValueError, OSError) as error:
            raise gr.Error(f'Не удалось сохранить: {error}') from error
        return session.first_open(index + 1)

    with gr.Blocks(title='Нужен ответ человека') as demo:
        gr.Markdown('## Нужен ли продавцу ответить лично?\n'
                    'Решайте только по тексту. Причины и необходимость человека — независимые решения. '
                    'Причины можно отметить и при ответе «хватит шаблона». Инструкция: docs/needs_reply.md.')
        index = gr.State(session.first_open())
        heading = gr.Textbox(label='Прогресс', interactive=False)
        # Textbox renders literal review text, including any Markdown or HTML.
        body = gr.Textbox(label='Текст отзыва', interactive=False, lines=10)
        need = gr.Radio([YES, NO], value=None, label='Нужен ли живой человек сверх шаблона?')
        reasons = gr.CheckboxGroup([(title, key) for key, title in REASONS.items()], value=[],
                                   label='Проблемы в отзыве (независимо от ответа выше)')
        unsure = gr.Checkbox(value=False, label='Сомневаюсь — требуется повторная проверка')
        message = gr.Textbox(label='Сообщение', interactive=False)
        with gr.Row():
            back = gr.Button('← Назад')
            go = gr.Button('Сохранить и дальше', variant='primary')
            forward = gr.Button('Дальше без сохранения →')
            resume = gr.Button('К первому неразмеченному')
        outputs = [index, heading, body, need, reasons, unsure, message]
        demo.load(show, index, outputs)
        go.click(save, [index, need, reasons, unsure], index).success(show, index, outputs)
        back.click(lambda i: max(i - 1, 0), index, index).then(show, index, outputs)
        forward.click(lambda i: min(i + 1, len(session.items) - 1), index, index).then(show, index, outputs)
        resume.click(session.first_open, None, index).then(show, index, outputs)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7864)
    parser.add_argument('--items', type=Path, default=ITEMS, help='Batch JSON with an items array')
    parser.add_argument('--labels', type=Path, default=OUT, help='Canonical JSON with a labels array')
    args = parser.parse_args()
    build_app(args.items, args.labels).launch(server_name='127.0.0.1', server_port=args.port, share=False)


if __name__ == '__main__':
    main()

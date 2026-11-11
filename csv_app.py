"""Local CSV upload UI. Run python csv_app.py --port 7865."""
import argparse
import gradio as gr
from csv_pipeline import inspect_csv, run_analysis


def build_app():
    def inspect(path):
        if not path:
            return gr.update(choices=[], value=None), 'Загрузите CSV', None, None
        try:
            columns = inspect_csv(path)
        except ValueError as error:
            return gr.update(choices=[], value=None), str(error), None, None
        return gr.update(choices=columns, value=None), 'Выберите столбец с текстом, затем нажмите «Анализировать»', None, None

    def analyze(path, column, threshold):
        try:
            csv_path, html_path = run_analysis(path, column, threshold)
            return 'Готово. Скачайте CSV с прогнозами и отчёт HTML.', csv_path, html_path
        except (ValueError, OSError, RuntimeError) as error:
            raise gr.Error(f'Анализ не выполнен: {error}') from error

    with gr.Blocks(title='Анализ отзывов CSV') as app:
        gr.Markdown('# Анализ отзывов из CSV\nЗагрузите файл, выберите столбец с текстом и запустите анализ. '
                    'До 5000 строк, 20 МБ. UTF-8 или Windows-1251; запятая, точка с запятой или табуляция.\n\n'
                    'Прогнозы локальной модели: кандидат на проверку человеком, восемь причин и тональность. '
                    'Баллы модели не являются калиброванными вероятностями. Решение об ответе принимает человек. '
                    'Порог причин — 0.50. Длинные отзывы обрезаются моделью до 256 токенов. '
                    'Исходный файл сохраняется без изменений.')
        upload = gr.File(label='CSV с отзывами', file_types=['.csv', '.tsv'], type='filepath')
        column = gr.Dropdown(choices=[], value=None, label='Столбец с текстом отзыва')
        threshold = gr.Slider(minimum=0, maximum=1, step=.01, value=.85, label='Порог кандидата на проверку человеком')
        go = gr.Button('Анализировать', variant='primary')
        status = gr.Textbox(label='Статус', interactive=False)
        csv_file = gr.File(label='Скачать CSV с прогнозами', interactive=False)
        html_file = gr.File(label='Скачать самостоятельный HTML-отчёт', interactive=False)
        upload.change(inspect, upload, [column, status, csv_file, html_file])
        go.click(lambda: ('Анализ выполняется…', None, None), None, [status, csv_file, html_file]).then(
            analyze, [upload, column, threshold], [status, csv_file, html_file], concurrency_limit=1)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7865)
    args = parser.parse_args()
    build_app().launch(server_name='127.0.0.1', server_port=args.port, share=False)


if __name__ == '__main__':
    main()

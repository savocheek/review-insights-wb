import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


class CsvPipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('csv_pipeline'), 'CSV pipeline must exist')
        import csv_pipeline
        self.pipeline = csv_pipeline
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'input.csv'
        self.old_output = csv_pipeline.RUNS_DIR
        csv_pipeline.RUNS_DIR = self.root / 'runs'
        self.addCleanup(setattr, csv_pipeline, 'RUNS_DIR', self.old_output)

    def predict(self, texts):
        self.seen = texts
        return [{'need_score': .9, 'reason_scores': {k: .6 if k == 'defect' else .1
                for k in self.pipeline.REASON_KEYS}, 'sentiment': 'негатив'} for _ in texts]

    def test_encodings_and_delimiters(self):
        for encoding, delimiter in [('utf-8-sig', ','), ('cp1251', ';'), ('utf-8', '\t')]:
            with self.subTest(encoding=encoding, delimiter=delimiter):
                self.path.write_text(f'id{delimiter}отзыв\n1{delimiter}Сломался\n', encoding=encoding)
                self.assertEqual(self.pipeline.inspect_csv(self.path), ['id', 'отзыв'])
                out, _ = self.pipeline.run_analysis(self.path, 'отзыв', predictor=self.predict)
                self.assertEqual(self.seen, ['Сломался'])
                self.assertTrue(Path(out).read_bytes().startswith(b'\xef\xbb\xbf'))

    def test_blank_rows_alignment_source_unchanged_and_fresh_runs(self):
        self.path.write_text('id,text\n1,first\n2,  \n\n3,last\n', encoding='utf-8')
        before = self.path.read_bytes()
        out, report = self.pipeline.run_analysis(self.path, 'text', predictor=self.predict)
        self.assertEqual(self.seen, ['first', 'last'])
        with open(out, encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r['id'] for r in rows], ['1', '2', '', '3'])
        self.assertEqual([r['model_status'] for r in rows], ['analyzed', 'blank_text', 'blank_text', 'analyzed'])
        self.assertEqual(rows[1]['model_need_score'], '')
        self.assertEqual(rows[0]['model_candidate'], 'true')
        self.assertEqual(rows[0]['model_reasons'], 'defect')
        self.assertEqual(rows[0]['model_sentiment'], 'негатив')
        second, _ = self.pipeline.run_analysis(self.path, 'text', predictor=self.predict)
        self.assertNotEqual(out, second)
        self.assertTrue(Path(report).is_file())
        self.assertEqual(self.path.read_bytes(), before)

    def test_html_and_csv_injection_are_escaped(self):
        self.path.write_text('=header,text\n=SUM(1),<script>alert(1)</script>\n@cmd,  =HYPERLINK(x)\n', encoding='utf-8')
        out, report = self.pipeline.run_analysis(self.path, 'text', predictor=self.predict)
        with open(out, encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0][0], "'=header")
        self.assertEqual(rows[1][0], "'=SUM(1)")
        self.assertEqual(rows[2][1], "'  =HYPERLINK(x)")
        html = Path(report).read_text(encoding='utf-8')
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('0.85', html)
        self.assertIn('0.50', html)
        self.assertIn('не калиброванные вероятности', html)

    def test_bad_files_selection_and_threshold_rejected_before_prediction(self):
        for text in ['', 'a,a\n1,2\n', 'a,b\n1,2,3\n', 'a\n\x00\n']:
            self.path.write_text(text, encoding='utf-8')
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.pipeline.inspect_csv(self.path)
        self.path.write_text('text\nhello\n', encoding='utf-8')
        for column, threshold in [('missing', .85), ('text', float('nan')), ('text', 1.1)]:
            with self.subTest(column=column, threshold=threshold), self.assertRaises(ValueError):
                self.pipeline.run_analysis(self.path, column, threshold, predictor=self.predict)
        self.assertFalse((self.root / 'runs').exists())

    def test_limits_and_prediction_count_validation(self):
        self.path.write_text('text\n' + 'x\n' * 5001, encoding='utf-8')
        with self.assertRaises(ValueError):
            self.pipeline.inspect_csv(self.path)
        self.path.write_text('text\nhello\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.pipeline.run_analysis(self.path, 'text', predictor=lambda texts: [])
        self.assertFalse((self.root / 'runs').exists())

    def test_report_aggregates_use_analyzed_denominator_and_candidates_sorted(self):
        self.path.write_text('text\nfirst\nsecond\n\nthird\n', encoding='utf-8')
        def predictions(texts):
            return [{'need_score': n, 'reason_scores': {key: .8 if key in reasons else .1
                     for key in self.pipeline.REASON_KEYS}, 'sentiment': sentiment}
                    for n, reasons, sentiment in [(.9, ['defect'], 'негатив'),
                                                  (.95, ['defect'], 'негатив'),
                                                  (.1, ['question'], 'позитив')]]
        _, report = self.pipeline.run_analysis(self.path, 'text', predictor=predictions)
        html = Path(report).read_text(encoding='utf-8')
        self.assertIn('негатив</td><td>2</td><td>66.7%', html)
        self.assertIn('Брак / поломка</td><td>2</td><td>66.7%', html)
        self.assertIn('Вопрос к продавцу</td><td>1</td><td>33.3%', html)
        candidates = html.split('<section id="candidates">')[1].split('</section>')[0]
        self.assertLess(candidates.index('Строка 2'), candidates.index('Строка 1'))
        self.assertNotIn('Строка 4', candidates)
        self.assertNotIn('Строка 3', candidates)


if __name__ == '__main__':
    unittest.main()

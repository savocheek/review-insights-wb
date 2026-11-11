import copy
import unittest

from demo_export import reviewed_export, render_demo


class DemoExportTests(unittest.TestCase):
    def sample(self):
        return {'meta': {}, 'products': [{'product': '<script>alert(1)</script>', 'nmId': 1, 'n': 2,
            'stars': {'5': 2}, 'avg_stars': 5, 'sentiment': {'позитив': 1},
            'themes': {'reviews_read': 2, 'liked': [], 'questions': [
                {'theme': 'не тянется', 'share': .5, 'reviews': 1, 'quotes': ['сорочка тянется']}]},
            'reasons': [{'title': 'брак', 'share': .5, 'reviews': 1, 'quotes': ['хороший товар']}],
            'seller_answered': .5, 'need_human': {'threshold': .85, 'count': 1, 'share': .5,
                'reviews': [{'text': '<img src=x>', 'score': .9, 'reasons': []}]}}]}

    def test_unverified_claims_suppressed_without_changing_input(self):
        source = self.sample()
        original = copy.deepcopy(source)
        out = reviewed_export(source)
        self.assertEqual(source, original)
        self.assertEqual(out['products'][0]['themes']['questions'], [])
        self.assertEqual(out['products'][0]['reasons'][0]['quotes'], [])
        self.assertEqual(out['products'][0]['need_human']['count'], 1)
        self.assertEqual(out, reviewed_export(out))

    def test_html_escapes_reviews_and_marks_predictions(self):
        html = render_demo(reviewed_export(self.sample()))
        self.assertNotIn('<script>', html)
        self.assertNotIn('<img src=x>', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('Предсказания модели', html)
        self.assertNotIn('сорочка тянется', html)
        self.assertIn('Товаров: 1 · отзывов: 2', html)

    def test_invalid_counts_rejected(self):
        data = self.sample()
        data['products'][0]['n'] = 3
        with self.assertRaises(ValueError):
            reviewed_export(data)


if __name__ == '__main__':
    unittest.main()

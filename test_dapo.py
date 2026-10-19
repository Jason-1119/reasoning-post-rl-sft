import unittest
from train_dapo import soft_penalty, informative_group, summarize


class DAPOTests(unittest.TestCase):
    def test_soft_penalty_boundaries(self):
        self.assertEqual(soft_penalty(100, 2048, 512, 1.), 0.)
        self.assertEqual(soft_penalty(1536, 2048, 512, 1.), 0.)
        self.assertEqual(soft_penalty(1792, 2048, 512, 1.), -.5)
        self.assertEqual(soft_penalty(2048, 2048, 512, 1.), -1.)
        self.assertEqual(soft_penalty(3000, 2048, 512, 1.), -1.)

    def test_dynamic_sampler_rejects_zero_variance_groups(self):
        self.assertFalse(informative_group([1, 1, 1, 1], [True]*4))
        self.assertFalse(informative_group([0, 0, 0, 0], [True]*4))
        self.assertTrue(informative_group([1, 0, 1, 0], [True]*4))

    def test_truncation_can_remove_the_only_correct_completion(self):
        self.assertFalse(informative_group([1, 0, 0, 0], [False, True, True, True]))
        self.assertFalse(informative_group([1, 0, 0, 0], [True, False, False, False]))
        self.assertTrue(informative_group([1, 0, 0, 0], [True, True, False, False]))

    def test_accuracy_recomputed_from_text_not_stale_labels(self):
        rows = [dict(text='#### 2', answer='2', tokens=10, seconds=1., budget_hit=False, correct=False),
                dict(text='The answer is 2.', answer='2', tokens=20, seconds=2., budget_hit=False, correct=True),
                dict(text='#### 3', answer='2', tokens=30, seconds=3., budget_hit=True, correct=True)]
        s = summarize(rows, 'dapo', 2048)
        self.assertEqual(s['correct_count'], 1)
        self.assertAlmostEqual(s['accuracy'], 1/3)
        self.assertAlmostEqual(s['answer_parse_rate'], 2/3)
        self.assertEqual(s['mean_tokens'], 20.)
        self.assertEqual(s['p95_tokens'], 30)


if __name__ == '__main__':
    unittest.main()

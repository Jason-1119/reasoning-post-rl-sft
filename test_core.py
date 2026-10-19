import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from pipeline import numeric, extract, correct, boxed, prompt, reward, uid
from monitoring import Progress, duration


class CoreTests(unittest.TestCase):
    def test_gsm_numeric_equivalence(self):
        for output in ['#### 1,200', '#### 1200.0', r'Answer: \boxed{1200}']:
            self.assertTrue(correct(output, '1200'))
        self.assertTrue(correct(r'\boxed{\frac{1}{2}}', '0.5'))
        self.assertFalse(correct('#### 12', '1200'))

    def test_no_guessing_or_execution(self):
        for text in ['The problem mentions 1200 apples.', '#### 1200 apples',
                     '#### __import__("os").system("echo nope")', '#### 1/0']:
            self.assertFalse(correct(text, '1200'))
        self.assertIsNone(numeric('nan'))

    def test_final_answer_not_internal_thought(self):
        self.assertIsNone(extract('#### 42\n</think>\nI cannot solve this.'))
        self.assertTrue(correct('#### 0\n</think>\n#### 42', '42'))
        self.assertFalse(correct('#### 42\n#### 13', '42'))

    def test_nested_box_and_malformed_box(self):
        self.assertEqual(boxed(r'x \boxed{\frac{1}{2}}'), r'\frac{1}{2}')
        self.assertIsNone(boxed(r'\boxed{\frac{1}{2}'))

    def test_rewards_and_identity(self):
        self.assertEqual(reward(['#### 2', '#### 0'], ['2', '2']), [1.0, 0.0])
        self.assertEqual(reward([[{'role':'assistant','content':'#### 2'}]], ['2']), [1.0])
        self.assertEqual(uid('A  question'), uid('A\nquestion'))
        self.assertNotIn('gold', prompt('Find 1+1.'))

    def test_resumed_progress_eta_excludes_previous_updates(self):
        with tempfile.TemporaryDirectory() as root, patch('monitoring.time.monotonic', side_effect=[100., 120.]):
            progress = Progress(root, 'grpo', 64, initial=16)
            with contextlib.redirect_stdout(io.StringIO()):
                record = progress.update(20, reward=.5)
            self.assertEqual(record['eta_seconds'], 220.)
            saved = json.loads((Path(root)/'logs/grpo_metrics.jsonl').read_text())
            self.assertEqual(saved['current'], 20)
            self.assertEqual(saved['reward'], .5)

    def test_progress_completion_and_unknown_eta(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            p = Progress(root, 'teacher', 128)
            self.assertIsNone(p.update(0)['eta_seconds'])
            self.assertEqual(p.update(128)['eta_seconds'], 0)
        self.assertEqual(duration(None), 'estimating')
        self.assertEqual(duration(3661), '01:01:01')


if __name__ == '__main__':
    unittest.main()

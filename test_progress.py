import tempfile
import unittest
from pathlib import Path
from progress import parse_line, describe

INFO = {'name': 'subject01', 'steps': 2000, 'photos': 28}


class ProgressTests(unittest.TestCase):
    def test_training_metrics_and_sample_counts(self):
        fields = parse_line('subject01: 50%|#####| 1000/2000 [42:00<42:00, 2.52s/it, lr: 1e-04 loss: 4.2e-01]', INFO)
        self.assertEqual(fields['step'], 1000)
        self.assertEqual(fields['remaining'], '42:00')
        self.assertEqual(fields['loss'], '4.2e-01')
        self.assertNotIn('step', parse_line('Generating Samples: 100%|##| 2/2 [00:26<00:00, 13s/it]', INFO))

    def test_complete_never_shows_remaining_time(self):
        with tempfile.TemporaryDirectory() as folder:
            job = Path(folder)
            (job / 'training.log').write_text('subject01: 99%|##| 1999/2000 [1:26:00<00:02, 2.1s/it]\n')
            result = describe(job, INFO, {'stage': 'complete', 'step': 2000, 'installed': 'personal.safetensors'})
            self.assertIn('Training finished', result)
            self.assertIn('2,000 / 2,000', result)
            self.assertIn('Active in Image Studio', result)
            self.assertNotIn('remaining', result)


if __name__ == '__main__':
    unittest.main()

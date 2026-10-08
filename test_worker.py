import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import worker


class WorkerTests(unittest.TestCase):
    def test_services_are_restored_after_training_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            job = state / 'jobs' / 'example'
            job.mkdir(parents=True)
            (job / 'job.json').write_text(json.dumps({'name': 'example', 'trigger': 'example_person', 'steps': 1000, 'install': False}))
            runner = worker.Runner(job)
            def pause():
                runner.paused = True
                runner.paused_ollama = True
            runner.release_gpu = pause
            runner.container = Mock(side_effect=RuntimeError('test failure'))
            with patch.object(worker, 'STATE', state), patch.object(worker, 'resource_errors', return_value=[]), patch.object(worker, 'runtime_ready', return_value=True), patch.object(worker.subprocess, 'run', return_value=Mock(returncode=0)) as run:
                runner.run()
            self.assertEqual(json.loads((job / 'status.json').read_text())['stage'], 'failed')
            names = [call.args[0][2] for call in run.call_args_list]
            self.assertIn('ollama', names)
            self.assertIn('ollama-pocket-comfyui', names)
            data = json.loads((job / 'status.json').read_text())
            self.assertFalse(data['paused_ollama'])
            self.assertFalse(data['paused_comfy'])
            self.assertIsNone(data['pid'])

    def test_gpu_memory_error_is_reported_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = worker.Runner(Path(directory))
            runner.info = {'name': 'example', 'steps': 1000}
            process = Mock(stdout=iter(['torch.OutOfMemoryError: CUDA out of memory\n']))
            process.wait.return_value = 1
            with patch.object(worker.subprocess, 'Popen', return_value=process):
                with self.assertRaisesRegex(RuntimeError, 'GPU memory ran out'):
                    runner.container(['run.py'], 'training')


if __name__ == '__main__':
    unittest.main()

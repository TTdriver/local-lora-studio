import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import core


class Tests(unittest.TestCase):
    def test_dataset_deduplicates_and_keeps_originals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for i in range(5):
                p = root / f'{i}.png'
                Image.new('RGB', (600, 400), (i * 40, 70, 100)).save(p)
                paths.append(p)
            with patch.object(core, 'STATE', root / 'private'):
                settings = {'learning_rate': 0.00005, 'rank': 16}
                job = core.prepare([*paths, paths[0]], 'subject01', settings=settings, trigger='short01')
                settings['rank'] = 64
                info = json.loads((job / 'job.json').read_text())
                self.assertEqual(info['photos'], 5)
                self.assertEqual(info['trigger'], 'short01')
                preview = core.training_config(info)['config']['process'][0]['sample']['prompts'][0]
                self.assertIn('short01', preview)
                self.assertNotIn('subject01_person', preview)
                self.assertEqual(info['training_settings']['rank'], 16)
                self.assertEqual(info['training_settings']['learning_rate'], 0.00005)
                self.assertEqual(len(info['rejected']), 1)
                self.assertTrue(all(path.exists() for path in paths))
                self.assertEqual(job.stat().st_mode & 0o777, 0o700)
                for p in (job / 'photos').glob('*.jpg'):
                    with Image.open(p) as photo:
                        self.assertFalse(photo.getexif())

    def test_invalid_alias_cannot_escape_job_folder(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(core, 'STATE', Path(temporary)):
            with self.assertRaises(ValueError):
                core.prepare([], '../../other')

    def test_edited_captions_are_kept_even_without_trigger(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'one.jpg').write_bytes(b'photo')
            (folder / 'two.jpg').write_bytes(b'photo')
            caption = folder / 'one.txt'
            caption.write_text('My carefully edited caption without a trigger.')
            self.assertEqual(core.caption_plan(folder), [folder / 'two.jpg'])
            self.assertEqual(caption.read_text(), 'My carefully edited caption without a trigger.')
            caption.write_text('')
            with self.assertRaisesRegex(ValueError, 'empty'):
                core.caption_plan(folder)

    def test_fresh_run_copies_captions_and_excludes_old_training_cache(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(core, 'STATE', Path(temporary)):
            job = core.STATE / 'jobs' / 'subject02_old'
            photos = job / 'photos'
            photos.mkdir(parents=True)
            info = {'name': 'subject02', 'trigger': 'subject02_person', 'steps': 1500,
                    'photos': 5, 'install': True, 'training_settings': {'rank': 16}}
            (job / 'job.json').write_text(json.dumps(info))
            for index in range(5):
                (photos / f'{index}.jpg').write_bytes(b'prepared image')
                (photos / f'{index}.txt').write_text(f'Edited caption {index}')
            (photos / '_t_e_cache').mkdir()
            (photos / '_t_e_cache' / 'old.pt').write_bytes(b'stale cache')
            (job / 'output').mkdir()
            new = core.copy_project(job)
            saved = json.loads((new / 'job.json').read_text())
            self.assertEqual(saved['trigger'], info['trigger'])
            self.assertEqual(saved['training_settings']['rank'], 16)
            self.assertEqual((new / 'photos' / '0.txt').read_text(), 'Edited caption 0')
            self.assertFalse((new / 'photos' / '_t_e_cache').exists())
            self.assertFalse((new / 'output').exists())
            (new / 'photos' / '0.txt').write_text('new edit')
            self.assertEqual((photos / '0.txt').read_text(), 'Edited caption 0')

    def test_training_adapter_and_existing_addon_are_separate(self):
        config = core.training_config({'name': 'subject01', 'trigger': 'subject01_person', 'steps': 2000})
        process = config['config']['process'][0]
        self.assertIn('training_adapter_v2', process['model']['assistant_lora_path'])
        self.assertNotIn('NSFW', json.dumps(config))
        self.assertTrue(process['train']['unload_text_encoder'])
        self.assertTrue(process['datasets'][0]['cache_text_embeddings'])

    def test_cancelled_project_settings_create_fresh_run(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(core, 'STATE', Path(temporary)):
            job = core.STATE / 'jobs' / 'old'
            photos = job / 'photos'
            photos.mkdir(parents=True)
            original = {'name': 'subject03', 'trigger': 'subject03_person', 'steps': 2000,
                        'photos': 5, 'install': True}
            (job / 'job.json').write_text(json.dumps(original))
            (job / 'status.json').write_text(json.dumps({'stage': 'cancelled', 'pid': None, 'started': 123}))
            for index in range(5):
                (photos / f'{index}.jpg').write_bytes(b'prepared')
                (photos / f'{index}.txt').write_text('A portrait of short01')
            new = core.edit_project_settings(job, {'learning_rate': 0.0001}, 'short01')
            self.assertNotEqual(new, job)
            info = json.loads((new / 'job.json').read_text())
            self.assertEqual(info['trigger'], 'short01')
            self.assertEqual(info['training_settings']['learning_rate'], 0.0001)
            self.assertEqual(json.loads((job / 'job.json').read_text()), original)
            self.assertEqual((new / 'photos' / '0.txt').read_text(), 'A portrait of short01')
            self.assertEqual(json.loads((new / 'status.json').read_text())['stage'], 'ready')
            (job / 'status.json').write_text(json.dumps({'stage': 'training', 'pid': 123}))
            with self.assertRaises(ValueError):
                core.edit_project_settings(job, {}, 'short01')

    def test_custom_settings_reach_training_config(self):
        info = {'name': 'subject02', 'trigger': 'subject02_person', 'steps': 1500,
                'training_settings': {'learning_rate': 0.00005, 'rank': 16, 'alpha': 8,
                'resolutions': [768], 'batch_size': 2, 'caption_dropout': 0.1,
                'checkpoint_interval': 100, 'keep_checkpoints': 2, 'preview_seed': 123,
                'preview_prompts': ['Portrait of {trigger} wearing a suit']}}
        process = core.training_config(info)['config']['process'][0]
        self.assertEqual(process['network']['linear'], 16)
        self.assertEqual(process['network']['linear_alpha'], 8)
        self.assertEqual(process['train']['lr'], 0.00005)
        self.assertEqual(process['train']['batch_size'], 2)
        self.assertEqual(process['datasets'][0]['resolution'], [768])
        self.assertEqual(process['datasets'][0]['caption_dropout_rate'], 0.1)
        self.assertEqual(process['save']['save_every'], 100)
        self.assertEqual(process['save']['max_step_saves_to_keep'], 2)
        self.assertEqual(process['sample']['sample_every'], 100)
        self.assertEqual(process['sample']['seed'], 123)
        self.assertEqual(process['sample']['prompts'], ['Portrait of subject02_person wearing a suit'])

    def test_old_projects_use_original_defaults(self):
        process = core.training_config({'name': 'old', 'trigger': 'old_person', 'steps': 2000})['config']['process'][0]
        self.assertEqual(process['train']['lr'], 0.0001)
        self.assertEqual(process['network']['linear'], 32)
        self.assertEqual(process['save']['save_every'], 250)

    def test_invalid_training_settings_rejected_before_creating_project(self):
        for settings in [{'learning_rate': float('nan')}, {'rank': 12.5}, {'batch_size': 0},
                         {'resolutions': []}, {'caption_dropout': 5}, {'preview_seed': -1},
                         {'preview_prompts': []}, {'checkpoint_interval': 0}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                core.prepare([], 'subject02', settings=settings)

    def test_truncated_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / 'test.safetensors'
            header = json.dumps({'lora_A.weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
            p.write_bytes(struct.pack('<Q', len(header)) + header + b'\x00' * 4)
            self.assertTrue(core.valid_lora(p))
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                core.valid_lora(p)


if __name__ == '__main__':
    unittest.main()

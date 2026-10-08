"""Private dataset preparation, trainer configuration, and resource checks."""
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path
from PIL import Image, ImageOps
import psutil
from training_settings import validate_settings

STATE = Path.home() / '.local/share/local-lora'
IMAGE = 'local-lora-trainer:1.0'
COMFY_MODELS = Path(os.environ.get('COMFY_MODELS_PATH', str(Path(__file__).resolve().parents[3] / 'work/deployment/images/models')))


def ensure_state():
    for folder in (STATE, STATE / 'jobs', STATE / 'cache'):
        folder.mkdir(parents=True, exist_ok=True)
        folder.chmod(0o700)


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.chmod(0o600)
    os.replace(tmp, path)


def status(job, stage, message, **fields):
    path = job / 'status.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    save(path, {**previous, **fields, 'stage': stage, 'message': message, 'updated': time.time()})


def validate_trigger(trigger):
    if not isinstance(trigger, str) or (trigger and not re.fullmatch(r'[A-Za-z0-9_]{1,64}', trigger)):
        raise ValueError('Use 1–64 letters, numbers or underscores for the trigger, or leave it blank for the project default.')
    return trigger


def prepare(paths, name, steps=2000, install=True, settings=None, trigger=''):
    settings = validate_settings(settings)
    trigger = validate_trigger(trigger.strip())
    ensure_state()
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{1,39}', name):
        raise ValueError('Use 2–40 letters/numbers/underscores, starting with a letter.')
    if not 500 <= int(steps) <= 4000:
        raise ValueError('Training steps must be between 500 and 4000.')
    job = STATE / 'jobs' / f'{name}_{time.time_ns()}'
    dataset = job / 'photos'
    dataset.mkdir(parents=True, mode=0o700)
    job.chmod(0o700)
    seen, rejected, count = set(), [], 0
    try:
        for source in paths:
            try:
                with Image.open(source) as raw:
                    photo = ImageOps.exif_transpose(raw).convert('RGB')
                    if min(photo.size) < 256:
                        raise ValueError('smaller than 256 pixels')
                    fingerprint = hashlib.sha256(str(photo.size).encode() + photo.tobytes()).hexdigest()
                    if fingerprint in seen:
                        rejected.append(f'{Path(source).name}: duplicate')
                        continue
                    seen.add(fingerprint)
                    photo.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
                    count += 1
                    photo.save(dataset / f'photo_{count:03}.jpg', quality=95, exif=b'')
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                rejected.append(f'{Path(source).name}: {error}')
        if count < 5:
            raise ValueError(f'Only {count} usable photos. Add at least five; 20–30 is recommended.')
        save(job / 'job.json', {'name': name, 'trigger': trigger or f'{name}_person', 'steps': int(steps),
                                'install': bool(install), 'photos': count, 'rejected': rejected,
                                'training_settings': settings})
        status(job, 'ready', f'{count} photos prepared. Originals unchanged; location metadata removed.')
        return job
    except Exception:
        shutil.rmtree(job)
        raise


def training_config(info):
    name, trigger = info['name'], info['trigger']
    options = validate_settings(info.get('training_settings'))
    return {'job': 'extension', 'config': {'name': name, 'process': [{
        'type': 'sd_trainer', 'training_folder': '/job/output', 'device': 'cuda:0',
        'trigger_word': trigger,
        'network': {'type': 'lora', 'linear': options['rank'], 'linear_alpha': options['alpha']},
        'save': {'dtype': 'bf16', 'save_every': options['checkpoint_interval'], 'max_step_saves_to_keep': options['keep_checkpoints']},
        'datasets': [{'folder_path': '/job/photos', 'caption_ext': 'txt',
                      'caption_dropout_rate': options['caption_dropout'], 'cache_latents_to_disk': True,
                      'cache_text_embeddings': True, 'resolution': options['resolutions']}],
        'train': {'batch_size': options['batch_size'], 'steps': info['steps'], 'gradient_checkpointing': True,
                  'train_unet': True, 'train_text_encoder': False,
                  'noise_scheduler': 'flowmatch', 'timestep_type': 'weighted',
                  'optimizer': 'adamw8bit', 'lr': options['learning_rate'], 'dtype': 'bf16',
                  'cache_text_embeddings': True, 'unload_text_encoder': True,
                  'skip_first_sample': True},
        # Keep quantized weights on the 24 GB GPU during loading. CPU parking
        # overlaps transformer and text encoder weights and exhausted 24 GB RAM.
        'model': {'name_or_path': 'Tongyi-MAI/Z-Image-Turbo', 'arch': 'zimage:turbo',
                  'assistant_lora_path': 'ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors',
                  'quantize': True, 'qtype': 'qfloat8', 'quantize_te': True, 'low_vram': False},
        'sample': {'sampler': 'flowmatch', 'sample_every': options['checkpoint_interval'], 'width': 1024, 'height': 1024,
                   'guidance_scale': 1, 'sample_steps': 9, 'seed': options['preview_seed'],
                   'prompts': [p.replace('{trigger}', trigger) for p in options['preview_prompts']]},
    }]}}


def caption_plan(folder):
    missing = []
    for photo in sorted(folder.glob('*.jpg')):
        caption = photo.with_suffix('.txt')
        if not caption.exists():
            missing.append(photo)
        elif not caption.read_text().strip():
            raise ValueError(f'{caption.name} is empty. Fill it in before training; existing captions are never overwritten.')
    return missing


def copy_project(job):
    """Fresh training run with edited captions; exclude checkpoints and stale caches."""
    ensure_state()
    info = json.loads((job / 'job.json').read_text())
    settings = validate_settings(info.get('training_settings'))
    caption_plan(job / 'photos')
    photos = sorted((job / 'photos').glob('*.jpg'))
    if len(photos) < 5:
        raise ValueError('The project needs at least five prepared photos.')
    destination = STATE / 'jobs' / f'{info["name"]}_{time.time_ns()}'
    folder = destination / 'photos'
    folder.mkdir(parents=True, mode=0o700)
    destination.chmod(0o700)
    try:
        for photo in photos:
            shutil.copyfile(photo, folder / photo.name)
            caption = photo.with_suffix('.txt')
            if caption.exists():
                shutil.copyfile(caption, folder / caption.name)
        save(destination / 'job.json', {**info, 'training_settings': settings, 'copied_from': job.name})
        status(destination, 'ready', 'Fresh run created with saved photos and captions. Training has not started.')
        return destination
    except Exception:
        shutil.rmtree(destination)
        raise


def edit_project_settings(job, settings, trigger):
    settings = validate_settings(settings)
    trigger = validate_trigger(trigger)
    data = json.loads((job / 'status.json').read_text())
    if data.get('pid') or data['stage'] not in ('ready', 'cancelled', 'failed'):
        raise ValueError('Wait for training and service restoration to finish before changing settings.')
    if data['stage'] == 'ready' and data.get('started'):
        raise ValueError('This project has already started. Create a fresh run first.')
    target = copy_project(job) if data['stage'] in ('cancelled', 'failed') else job
    info = json.loads((target / 'job.json').read_text())
    info['training_settings'] = settings
    info['trigger'] = trigger or info['name'] + '_person'
    save(target / 'job.json', info)
    return target


def runtime_ready():
    return subprocess.run(['docker', 'image', 'inspect', IMAGE], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def resource_errors(training=True):
    ensure_state()
    errors = []
    disk = shutil.disk_usage(STATE).free / 1024**3
    ram = psutil.virtual_memory().total / 1024**3
    required_disk = 45 if training else 10
    if disk < required_disk:
        errors.append(f'{disk:.1f} GB disk free; allow at least {required_disk} GB for this operation.')
    if training and ram < 23:
        errors.append(f'{ram:.1f} GB RAM assigned. Increase the VM to at least 24 GB; 32 GB is recommended.')
    if training:
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total',
                                              '--format=csv,noheader,nounits'], text=True, timeout=5)
            if float(result.splitlines()[0]) < 22000:
                errors.append('This training preset requires a GPU with about 24 GB VRAM.')
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            errors.append('NVIDIA GPU is unavailable.')
    return errors


def valid_lora(path):
    """Reject empty/truncated files before copying them into ComfyUI."""
    with path.open('rb') as file:
        size = path.stat().st_size
        header_size = struct.unpack('<Q', file.read(8))[0]
        if not 0 < header_size < min(size - 8, 32 * 1024**2):
            raise ValueError('Invalid safetensors header.')
        header = json.loads(file.read(header_size))
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    if not tensors or not any('lora' in key.lower() for key in tensors):
        raise ValueError('The output is not a LoRA checkpoint.')
    for item in tensors.values():
        start, end = item['data_offsets']
        if not 0 <= start <= end <= size - header_size - 8:
            raise ValueError('Truncated safetensors checkpoint.')
    return True

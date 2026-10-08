"""Background job runner. Closing the desktop window does not interrupt training."""
import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from progress import parse_line
from core import STATE, IMAGE, status, resource_errors, runtime_ready, training_config, valid_lora, COMFY_MODELS, caption_plan

ACTIVATE = r'''
import os,json,asyncio
from pathlib import Path
os.environ['WEBUI_SECRET_KEY']=Path('/app/backend/.webui_secret_key').read_text().strip()
from open_webui.models.config import Config
async def main():
    data=json.loads(os.environ['POCKET_LORA'])
    values=await Config.get_many('image_generation.comfyui.workflow')
    workflow=json.loads(values['image_generation.comfyui.workflow'])
    if workflow['4']['class_type']!='UNETLoader' or not workflow['4']['inputs']['unet_name'].startswith('z_image_turbo'):
        raise ValueError('Image Studio is no longer configured for Z-Image Turbo.')
    # Preserve the existing add-on. Replace only this application's identity node.
    node='900'
    if node in workflow and workflow[node].get('_meta',{}).get('title')!='Personal LoRA':
        raise ValueError('Workflow node 900 is already in use.')
    upstream=workflow['11']['inputs']['model']
    if upstream[0]==node:
        upstream=workflow[node]['inputs']['model']
    workflow[node]={'class_type':'LoraLoaderModelOnly','inputs':{'model':upstream,'lora_name':data['filename'],'strength_model':0.8},'_meta':{'title':'Personal LoRA'}}
    workflow['11']['inputs']['model']=[node,0]
    backup=Path('/app/backend/data/pocket-personal-lora-backup.json')
    if not backup.exists():
        backup.write_text(json.dumps(values));backup.chmod(0o600)
    await Config.upsert({'image_generation.comfyui.workflow':json.dumps(workflow)})
    print('Personal LoRA activated at strength 0.8')
asyncio.run(main())
'''


def install(job, checkpoint):
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to((job / 'output').resolve()):
        raise ValueError('Choose a checkpoint from this training job.')
    valid_lora(checkpoint)
    folder = COMFY_MODELS / 'loras'
    folder.mkdir(parents=True, exist_ok=True)
    filename = f'personal_{job.name}_{checkpoint.name}'
    destination = folder / filename
    import shutil
    temp = destination.with_suffix('.part')
    shutil.copyfile(checkpoint, temp)
    temp.chmod(0o644)
    os.replace(temp, destination)
    result = subprocess.run(['docker', 'exec', '-i', '-e', 'PYTHONPATH=/app/backend',
                             '-e', 'POCKET_LORA=' + json.dumps({'filename': filename}),
                             'open-webui', 'python', '-'], input=ACTIVATE, text=True,
                            capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Checkpoint copied, but activation failed: ' + result.stderr[-1500:])
    return filename


class Runner:
    def __init__(self, job):
        self.job = job
        self.name = 'local-lora-' + job.name.lower()
        self.cancelled = False
        self.paused = False
        self.paused_ollama = False
        self.process = None
        self.lock = None
        signal.signal(signal.SIGTERM, self.cancel)
        signal.signal(signal.SIGINT, self.cancel)

    def cancel(self, *_):
        self.cancelled = True
        subprocess.run(['docker', 'stop', '-t', '20', self.name], capture_output=True, timeout=30)
        if self.process and self.process.poll() is None:
            self.process.terminate()

    def container(self, command, stage):
        if self.cancelled:
            raise InterruptedError('Cancelled before starting the next stage.')
        uid, gid = os.getuid(), os.getgid()
        args = ['docker', 'run', '--rm', '--name', self.name, '--user', f'{uid}:{gid}',
                '--gpus', 'all', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
                '--memory', '24g', '--shm-size', '1g', '-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility',
                '-v', f'{STATE}:/state', '-v', f'{self.job}:/job',
                '-v', f'{Path(__file__).with_name("caption.py")}:/opt/caption.py:ro', IMAGE, *command]
        self.process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
        gpu_oom = False
        with (self.job / 'training.log').open('a') as log:
            for line in self.process.stdout:
                log.write(line); log.flush()
                gpu_oom = gpu_oom or 'CUDA out of memory' in line
                if self.cancelled:
                    continue
                fields = parse_line(line, self.info)
                if fields:
                    status(self.job, stage, fields.get('phase', 'Training your likeness'), **fields)
        code = self.process.wait()
        if self.cancelled:
            raise InterruptedError('Cancelled. Saved checkpoints remain available.')
        if code and gpu_oom:
            raise RuntimeError('GPU memory ran out while training. Another model may have occupied the GPU. '
                               'Retry with chat and image services paused. Saved photos and checkpoints remain.')
        if code == 137:
            raise RuntimeError(
                f'{stage} was killed (exit 137), usually because system RAM ran out. '
                'Use the GPU-resident loading preset; if it persists, assign more RAM to the VM. '
                f'Log: {self.job / "training.log"}')
        if code:
            raise RuntimeError(f'{stage} failed (exit {code}). See the job log.')

    def release_gpu(self):
        # Do not unload an actively working chat model or interrupt an image job.
        with urllib.request.urlopen('http://127.0.0.1:11434/api/ps', timeout=5) as response:
            models = json.load(response).get('models', [])
        utilization = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True, timeout=5)
        if float(utilization.splitlines()[0]) > 10:
            raise RuntimeError('The GPU is busy. Wait for chat/image generation to finish, then retry.')
        running = subprocess.check_output(['docker', 'inspect', '--format', '{{.State.Running}}',
                                           'ollama-pocket-comfyui'], text=True).strip() == 'true'
        if running:
            result = subprocess.run(['docker', 'exec', 'ollama-pocket-comfyui', 'python', '-c',
                "import json,urllib.request; q=json.load(urllib.request.urlopen('http://127.0.0.1:8188/queue')); assert not(q['queue_running'] or q['queue_pending']), 'Images are still running'"],
                capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('Image backend is busy or not ready. Wait and retry.')
            self.paused = True
            status(self.job, 'preparing', 'Pausing image generation while training', paused_comfy=True)
            subprocess.run(['docker', 'stop', 'ollama-pocket-comfyui'], check=True, capture_output=True)
        for model in models:
            request = urllib.request.Request('http://127.0.0.1:11434/api/generate',
                      data=json.dumps({'model': model['name'], 'keep_alive': 0, 'stream': False}).encode(),
                      headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=60) as response:
                response.read()
        # Unloading alone races with chat requests during CPU captioning/model loading.
        # Stop this managed backend for the job, then restore it in finally.
        running = subprocess.check_output(['docker', 'inspect', '--format', '{{.State.Running}}',
                                           'ollama'], text=True).strip() == 'true'
        if running:
            self.paused_ollama = True
            status(self.job, 'preparing', 'Pausing chat and image generation to reserve GPU memory',
                   paused_ollama=True, phase='Reserving the GPU for training')
            subprocess.run(['docker', 'stop', '-t', '20', 'ollama'], check=True, capture_output=True, timeout=30)
        free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True, timeout=5)
        if float(free.splitlines()[0]) < 18000:
            raise RuntimeError('Less than 18 GB GPU memory is free after pausing services. '
                               'Close other GPU workloads before retrying.')

    def run(self):
        self.info = json.loads((self.job / 'job.json').read_text())
        status(self.job, 'checking', 'Checking resources', pid=os.getpid(), step=0, started=time.time(), finished=None, phase='Checking resources')
        try:
            self.lock = (STATE / 'training.lock').open('a')
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another training job is already running. Wait for it to finish.')
            issues = resource_errors()
            if issues:
                raise RuntimeError('\n'.join(issues))
            if not runtime_ready():
                raise RuntimeError('Trainer runtime is not installed. Click Prepare trainer first.')
            missing_captions = caption_plan(self.job / 'photos')
            self.release_gpu()
            save_config = self.job / 'config.json'
            save_config.write_text(json.dumps(training_config(self.info), indent=2))
            status(self.job, 'captioning', 'Creating local photo captions', phase='Creating captions on CPU')
            photos = list((self.job / 'photos').glob('*.jpg'))
            captions_ready = bool(photos) and not missing_captions
            if captions_ready:
                status(self.job, 'captioning', f'Using {len(photos)} saved photo captions', phase='Saved captions ready')
            else:
                self.container(['/opt/caption.py', '/job/photos', self.info['trigger']], 'captioning')
            status(self.job, 'training', 'Loading model and training adapter; first use downloads weights', phase='Loading model on GPU')
            self.container(['run.py', '/job/config.json'], 'training')
            checkpoints = sorted((self.job / 'output').rglob('*.safetensors'), key=lambda p: p.stat().st_mtime)
            if not checkpoints:
                raise RuntimeError('Trainer finished without a checkpoint.')
            final = self.job / 'output' / self.info['name'] / f'{self.info["name"]}.safetensors'
            if not final.exists():
                final = checkpoints[-1]
            valid_lora(final)
            status(self.job, 'installing', 'Validating and activating final checkpoint', phase='Activating finished LoRA')
            installed = install(self.job, final) if self.info['install'] else None
            status(self.job, 'complete', f'Finished. Use {self.info["trigger"]} in image prompts.',
                   checkpoint=str(final), installed=installed, step=self.info['steps'], finished=time.time())
        except InterruptedError as error:
            status(self.job, 'cancelled', str(error), finished=time.time())
        except Exception as error:
            status(self.job, 'failed', str(error), finished=time.time())
        finally:
            restore = {}
            for paused, name, field in [(self.paused, 'ollama-pocket-comfyui', 'paused_comfy'),
                                         (self.paused_ollama, 'ollama', 'paused_ollama')]:
                if paused:
                    try:
                        result = subprocess.run(['docker', 'start', name], capture_output=True, timeout=30)
                        restore[field] = result.returncode != 0
                    except (OSError, subprocess.SubprocessError):
                        restore[field] = True
            data = json.loads((self.job / 'status.json').read_text())
            status(self.job, data['stage'], data['message'], **restore, pid=None)

            if self.lock:
                self.lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    parser.add_argument('--install', type=Path)
    args = parser.parse_args()
    job = args.job.resolve()
    if not job.is_relative_to((STATE / 'jobs').resolve()):
        raise SystemExit('Job must be in the private application folder.')
    if args.install:
        print(install(job, args.install))
    else:
        Runner(job).run()

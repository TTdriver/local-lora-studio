"""Read bounded log tails and interpret trainer progress without confusing sample counts."""
import re
import time


def log_tail(job, limit=65536):
    try:
        with (job / 'training.log').open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours}h {minutes:02}m' if hours else f'{minutes}m {seconds:02}s'


def parse_line(line, info):
    line = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line).strip()
    fields = {}
    if line.startswith(info['name'] + ':'):
        match = re.search(r'(\d+)/(\d+)\s*\[([^<\]]+)(?:<([^,\]]+))?', line)
        if match and int(match[2]) == info['steps']:
            fields.update(phase='Training your likeness', step=int(match[1]), training_elapsed=match[3])
            if match[4] and '?' not in match[4]:
                fields['remaining'] = match[4]
            speed = re.search(r'([\d.]+)(s/it|it/s)', line)
            if speed:
                fields['seconds_per_step'] = float(speed[1]) if speed[2] == 's/it' else 1 / max(float(speed[1]), 0.001)
            loss = re.search(r'loss:\s*([\d.eE+-]+)', line)
            if loss:
                fields['loss'] = loss[1]
    elif 'Captioned photo' in line:
        fields['phase'] = line
    else:
        phases = [('Downloading', 'Downloading model weights'), ('Fetching', 'Downloading model files'),
                  ('Loading transformer', 'Loading image model'), ('Text Encoder', 'Loading text encoder'),
                  ('Quantizing', 'Compressing model weights for the GPU'),
                  ('Caching latents', 'Preparing image cache'), ('Caching text', 'Preparing text cache'),
                  ('UNLOADING TEXT', 'Releasing text encoder memory'),
                  ('Saved checkpoint', 'Saving checkpoint'), ('Generating Samples', 'Generating test pictures')]
        for marker, phase in phases:
            if marker in line:
                fields['phase'] = phase
                break
    return fields


def describe(job, info, data):
    metrics = {}
    for line in log_tail(job).splitlines():
        metrics.update(parse_line(line, info))
    stage = data.get('stage', 'ready')
    step = data.get('step', 0)
    lines = [f"Training steps: {step:,} / {info['steps']:,} ({100 * step / info['steps']:.1f}%) · {info['photos']} photos"]
    if stage == 'complete':
        lines.insert(0, 'Training finished — final checkpoint saved.')
        lines.append('Active in Image Studio.' if data.get('installed') else 'Not activated. Choose a checkpoint and click Activate checkpoint.')
    elif stage in ('failed', 'cancelled'):
        lines.insert(0, 'Training stopped. Saved checkpoints remain available.')
    else:
        lines.insert(0, data.get('phase') or metrics.get('phase') or data.get('message', 'Waiting'))
    if step > 0 and metrics.get('training_elapsed'):
        lines.append(f"Time spent training: {metrics['training_elapsed']}")
    if stage == 'training' and step > 0 and metrics.get('remaining') and step < info['steps']:
        lines.append(f"Estimated training time remaining: {metrics['remaining']} (previews/saving can add time)")
    if step > 0 and metrics.get('seconds_per_step') and stage == 'training':
        lines.append(f"Recent speed: {metrics['seconds_per_step']:.2f} seconds/step · Loss: {metrics.get('loss', '—')}")
    if stage == 'training' and 0 < step < info['steps']:
        interval = info.get('training_settings', {}).get('checkpoint_interval', 250)
        next_save = min(info['steps'], (step // interval + 1) * interval)
        lines.append(f'Next checkpoint and test pictures: step {next_save:,}')
    if data.get('started'):
        end = data.get('finished') or time.time()
        lines.append(f"Total job time: {duration(end - data['started'])}")
    if stage not in ('complete', 'failed', 'cancelled', 'ready'):
        if data.get('paused_ollama'):
            lines.append('Chat backend is paused to reserve the GPU.')
        lines.append('Image generation is paused.' if data.get('paused_comfy') else 'Image generation service is not paused.')
        lines.append(f"Last status update: {duration(time.time() - data.get('updated', time.time()))} ago")
    if stage in ('complete', 'failed', 'cancelled') and (data.get('paused_ollama') or data.get('paused_comfy')):
        lines.append('A service still needs restarting. Click Restore AI services.')
    return '\n'.join(lines)

"""Validated, per-project training options with backwards-compatible defaults."""
import copy
import math

DEFAULTS = {
    'learning_rate': 0.0001, 'rank': 32, 'alpha': 32,
    'resolutions': [512, 768, 1024], 'batch_size': 1,
    'caption_dropout': 0.05, 'checkpoint_interval': 250, 'keep_checkpoints': 4,
    'preview_seed': 42,
    'preview_prompts': [
        'A photo of {trigger}, head and shoulders, natural daylight, neutral background',
        'A photo of {trigger}, standing outdoors in a park, casual clothing'],
}


def validate_settings(values=None):
    options = copy.deepcopy(DEFAULTS)
    if values is not None:
        if not isinstance(values, dict) or set(values) - set(DEFAULTS):
            raise ValueError('Unrecognized training settings.')
        options.update(copy.deepcopy(values))
    for key, low, high in [('learning_rate', 0.000001, 0.001), ('caption_dropout', 0, 1)]:
        try:
            value = float(options[key])
        except (TypeError, ValueError):
            raise ValueError(f'{key.replace("_", " ").capitalize()} must be a number.') from None
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key.replace("_", " ").capitalize()} must be between {low} and {high}.')
        options[key] = value
    for key, low, high in [('rank', 8, 128), ('alpha', 1, 128), ('batch_size', 1, 4),
                           ('checkpoint_interval', 25, 4000), ('keep_checkpoints', 1, 20),
                           ('preview_seed', 0, 4294967295)]:
        try:
            value = int(options[key])
            exact = float(options[key]) == value
        except (TypeError, ValueError, OverflowError):
            exact = False
        if not exact or not low <= value <= high:
            raise ValueError(f'{key.replace("_", " ").capitalize()} must be a whole number between {low} and {high}.')
        options[key] = value
    sizes = options['resolutions']
    if not isinstance(sizes, list) or not sizes or any(size not in (512, 768, 1024) for size in sizes):
        raise ValueError('Choose at least one training resolution: 512, 768 or 1024.')
    options['resolutions'] = sorted(set(sizes))
    prompts = options['preview_prompts']
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= 8 or any(
            not isinstance(p, str) or not p.strip() or len(p) > 1000 for p in prompts):
        raise ValueError('Enter 1–8 preview prompts, each up to 1,000 characters.')
    options['preview_prompts'] = [p.strip() for p in prompts]
    return options

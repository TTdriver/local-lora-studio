"""Caption local photos. Only model weights are downloaded; photos never leave Docker."""
import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument('folder')
parser.add_argument('trigger')
args = parser.parse_args()
photos = sorted(Path(args.folder).glob('*.jpg'))
missing = [photo for photo in photos if not photo.with_suffix('.txt').exists()]
if not missing:
    print('Using existing captions; nothing to generate.', flush=True)
    raise SystemExit(0)
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
model.eval()
for index, path in enumerate(missing, 1):
    with Image.open(path) as photo:
        inputs = processor(images=photo.convert('RGB'), return_tensors='pt')
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=65)
    caption = processor.decode(tokens[0], skip_special_tokens=True)
    # Exclusive creation also protects an edit made while caption generation runs.
    try:
        with path.with_suffix('.txt').open('x') as output:
            output.write(f'A photo of {args.trigger}. {caption}\n')
    except FileExistsError:
        print(f'Keeping existing caption for {path.name}', flush=True)
    print(f'Captioned photo {index}', flush=True)

# Local LoRA Studio

Current release: **0.1.1**. A local Linux desktop application.

## Installation

Download the repository source ZIP from GitHub, extract it, and open a terminal in that folder. On Ubuntu/Zorin install Python prerequisites:

```bash
sudo apt install python3-venv python3-tk python3-pil python3-psutil
python3 install.py
```

The installer installs app files and a desktop/application-menu launcher. Existing user data is preserved. Local WebUI creates its isolated Qt runtime when needed.

## Update notifications

The app checks exactly once at launch in a background thread, using a five-second timeout and GitHub's public Contents API for the root `VERSION` file. There are no credentials in the request. Equal, older, malformed, and unavailable versions are silent. A newer version adds a muted bottom-right **Update available · vX.Y.Z ↗** link to this installation page. There are no automatic downloads or installations.

`APP_VERSION`, `VERSION`, and the installer describe the same release. Update checks use the Contents API rather than the raw endpoint to avoid stale raw-file responses. Results pass through a queue to the GUI thread; normal status updates cannot replace the link.

## Tests

```bash
python3 -m unittest test_update_check
```

## Training prerequisites

Requires the existing local NVIDIA/Docker training and ComfyUI deployment. Set `COMFY_MODELS_PATH` to the ComfyUI models directory when launching the app. The app expects containers named `open-webui`, `ollama`, and `ollama-pocket-comfyui`, plus a compatible Z-Image workflow. Install/build those services before training; this repository ships the desktop trainer controls and runtime recipe, not a complete image-server deployment. Photos, captions, and checkpoints remain in `~/.local/share/local-lora` and are not distributed.

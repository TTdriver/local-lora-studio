FROM ollama-pocket-comfyui:1.0
USER root
WORKDIR /opt
RUN git clone https://github.com/ostris/ai-toolkit.git ai-toolkit && \
    cd ai-toolkit && git checkout ecee894ed2b1f3716d9d7326693061ec1a3105bb && \
    printf 'torch==2.9.1\ntorchvision==0.24.1\ntorchaudio==2.9.1\n' > /tmp/trainer-constraints.txt && \
    pip install --no-cache-dir -c /tmp/trainer-constraints.txt -r requirements.txt
COPY caption.py /opt/caption.py
ENV HF_HOME=/state/cache/huggingface MODELS_PATH=/state/cache/models \
    XDG_CACHE_HOME=/state/cache TORCH_HOME=/state/cache/torch \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 DO_NOT_TRACK=1 WANDB_MODE=disabled
WORKDIR /opt/ai-toolkit
ENTRYPOINT ["python"]

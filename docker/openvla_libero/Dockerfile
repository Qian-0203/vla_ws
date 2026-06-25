FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV LIBERO_CONFIG_PATH=/workspace/.libero
ENV PYTHONPATH=/workspace/openvla:/workspace/LIBERO

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    libosmesa6-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel packaging

RUN python -m pip install \
    "accelerate>=0.25.0" \
    "draccus==0.8.0" \
    "einops" \
    "huggingface_hub" \
    "json-numpy" \
    "jsonlines" \
    "matplotlib" \
    "peft==0.11.1" \
    "protobuf" \
    "rich" \
    "sentencepiece==0.1.99" \
    "timm==0.9.10" \
    "tokenizers==0.19.1" \
    "transformers==4.40.1" \
    "wandb" \
    "tensorflow==2.15.0" \
    "tensorflow_datasets==4.9.3" \
    "tensorflow_graphics==2021.12.3" \
    "imageio[ffmpeg]" \
    "mujoco==2.3.2" \
    "robosuite==1.4.1" \
    "bddl==1.0.1" \
    "easydict" \
    "cloudpickle" \
    "future==0.18.2" \
    "gym==0.25.2" \
    "opencv-python-headless" \
    "pyyaml" \
    "dlimp @ git+https://github.com/moojink/dlimp_openvla"

RUN python -m pip install "flash-attn==2.5.5" --no-build-isolation

RUN python -m pip install --no-deps "tensorflow-metadata==1.16.1"

# The base image ships only the Mesa (software) GLVND EGL vendor ICD, so libEGL
# falls back to swrast and headless MuJoCo rendering fails. Register the NVIDIA
# EGL ICD (libEGL_nvidia.so.0 is injected by the nvidia container runtime when
# the graphics driver capability is enabled) so GLVND uses the GPU.
RUN printf '%s\n' \
    '{' \
    '    "file_format_version" : "1.0.0",' \
    '    "ICD" : {' \
    '        "library_path" : "libEGL_nvidia.so.0"' \
    '    }' \
    '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY docker/openvla_libero/entrypoint.sh /usr/local/bin/openvla-libero-entrypoint
RUN chmod +x /usr/local/bin/openvla-libero-entrypoint

WORKDIR /workspace/openvla
ENTRYPOINT ["openvla-libero-entrypoint"]
CMD ["bash"]

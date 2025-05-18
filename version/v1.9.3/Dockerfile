# Dockerfile for FramePack-eichi
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    build-essential \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip and install build dependencies
RUN pip install --upgrade pip
RUN pip install packaging wheel setuptools ninja

# Install PyTorch with CUDA 12.6 support
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Install acceleration libraries one by one to better diagnose any issues
RUN pip install xformers==0.0.29.post3 --no-deps --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir
RUN pip install triton==2.2.0 --no-cache-dir
RUN pip install -U "huggingface_hub[cli]"

# Install flash-attention with the closest available version
RUN pip install flash-attn==2.7.4.post1

# Try installing sageattention
# Option 1: Install from git repository directly
RUN pip install sageattention

# Install remaining packages
RUN pip install pynvml "jinja2>=3.1.2" peft

# Create working directory
WORKDIR /app

# Clone original FramePack repository (needed for the base functionality)
RUN git clone https://github.com/lllyasviel/FramePack.git /app/framepack

# Install dependencies
WORKDIR /app/framepack
RUN pip install -r requirements.txt

# Clone FramePack-eichi repository
RUN git clone https://github.com/git-ai-code/FramePack-eichi /tmp/framepack-eichi

# Copy FramePack-eichi files
RUN cp -rf /tmp/framepack-eichi/webui/* /app/framepack/

# Create a simple startup script with better error handling
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'cd /app/framepack' >> /app/start.sh && \
    echo 'echo "Starting demo_gradio.py..."' >> /app/start.sh && \
    echo 'python demo_gradio.py --server 0.0.0.0 --port 7860 &' >> /app/start.sh && \
    echo 'SERVER_PID=$!' >> /app/start.sh && \
    echo 'sleep 5' >> /app/start.sh && \
    echo 'if [ -f "endframe_ichi.py" ]; then' >> /app/start.sh && \
    echo '  echo "Starting endframe_ichi.py..."' >> /app/start.sh && \
    echo '  python endframe_ichi.py --server 0.0.0.0 --port 7861 "$@"' >> /app/start.sh && \
    echo 'fi' >> /app/start.sh && \
    chmod +x /app/start.sh

# Set working directory for when container starts
WORKDIR /app/framepack

# Command to run when container starts
ENTRYPOINT ["/app/start.sh"]

# Default arguments (can be overridden)
CMD ["--lang", "en", "--listen"]
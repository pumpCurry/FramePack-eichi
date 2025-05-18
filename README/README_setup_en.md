# FramePack-eichi Setup Guide: Comprehensive Installation Manual for All Environments | [日本語](README_setup.md) | [繁體中文](README_setup_zh.md) | [Русский](README_setup_ru.md)

> **Disclaimer**: This document is a compilation of information collected from the internet and does not guarantee functionality in all environments. The procedures described may not work properly due to differences in environments and versions. Please adjust them according to your specific environment as needed. It is also recommended to always refer to the latest information in the official repository.

FramePack-eichi is an AI video generation system that creates short videos from a single image using text prompts. It is a fork of the original FramePack developed by Lvmin Zhang and Maneesh Agrawala at Stanford University, with numerous additional features and enhancements. This guide provides accurate setup procedures for each environment, system requirements, and troubleshooting tips.

## System Requirements

### RAM Requirements
- **Minimum**: 16GB (will work but with performance limitations)
- **Recommended**: 32GB (sufficient for standard operations)
- **Optimal**: 64GB (ideal for longer videos, LoRA usage, and high-resolution processing)
- If insufficient RAM is available, the system will use SSD swap space, which may reduce the lifespan of your SSD

### VRAM Requirements
- **Minimum**: 8GB VRAM (recommended minimum for FramePack-eichi)
- **Low VRAM Mode**: Automatically activated and efficiently manages memory
  - Adjustable via the `gpu_memory_preservation` setting (default: 10GB)
  - Lower value = More VRAM for processing = Faster but riskier
  - Higher value = Less VRAM for processing = Slower but more stable
- **High VRAM Mode**: Automatically activated when more than 100GB of free VRAM is detected
  - Models remain resident in GPU memory (approximately 20% faster)
  - No need for periodic model loading/unloading

### CPU Requirements
- No explicit minimum CPU model is specified
- **Recommended**: Modern multi-core CPU with 8+ cores
- CPU performance affects loading times and pre/post-processing
- Most of the actual generation processing runs on the GPU

### Storage Requirements
- **Application Code**: Typically 1-2GB
- **Models**: About 30GB (automatically downloaded on first launch)
- **Output and Temporary Files**: Depends on video length, resolution, and compression settings
- **Total Recommended Capacity**: 150GB or more
- SSD is recommended for frequent read/write operations

### Supported GPU Models
- **Officially Supported**: NVIDIA RTX 30XX, 40XX, 50XX series (supporting fp16 and bf16 data formats)
- **Minimum Recommended**: RTX 3060 (or equivalent with 8GB+ VRAM)
- **Confirmed Working**: RTX 3060, 3070Ti, 4060Ti, 4090
- **Unofficial/Untested**: GTX 10XX/20XX series
- **AMD GPUs**: No explicit support mentioned
- **Intel GPUs**: No explicit support mentioned

## Windows Setup Instructions

### Prerequisites
- Windows 10/11
- NVIDIA GPU with drivers supporting CUDA 12.6
- Python 3.10.x
- 7-Zip (for extracting installation packages)

### Step-by-Step Instructions
1. **Installing the Base FramePack**:
   - Go to the [official FramePack repository](https://github.com/lllyasviel/FramePack)
   - Click "Download One-Click Package (CUDA 12.6 + PyTorch 2.6)"
   - Download and extract the 7z package to a location of your choice
   - Run `update.bat` (important for getting the latest bug fixes)
   - Run `run.bat` to launch FramePack for the first time
   - Required models (about 30GB) will be automatically downloaded during first run

2. **Installing FramePack-eichi**:
   - Clone or download the [FramePack-eichi repository](https://github.com/git-ai-code/FramePack-eichi)
   - Copy the appropriate language batch file (`run_endframe_ichi.bat` for Japanese, `run_endframe_ichi_en.bat` for English, `run_endframe_ichi_zh-tw.bat` for Traditional Chinese, `run_endframe_ichi_ru.bat` for Russian) to the FramePack root directory
   - For F1 version, use `run_endframe_ichi_f1.bat` (or language-specific `run_endframe_ichi_en_f1.bat`, `run_endframe_ichi_zh-tw_f1.bat`, `run_endframe_ichi_f1_ru.bat`)
   - For single-frame inference, use `run_oneframe_ichi.bat` (or language-specific `run_oneframe_ichi_en.bat`, `run_oneframe_ichi_zh-tw.bat`, `run_oneframe_ichi_ru.bat`)
   - Copy the following files/folders from FramePack-eichi to the `webui` folder in FramePack:
     - `endframe_ichi.py`
     - `eichi_utils` folder (includes `lora_preset_manager.py`, `model_downloader.py`, `vae_settings.py`, etc. - added in v1.9.3)
     - `lora_utils` folder
     - `diffusers_helper` folder
     - `locales` folder (includes `ru.json` Russian translation file - added in v1.9.3)

3. **Installing Acceleration Libraries (Optional but Recommended)**:
   - Download the acceleration package installer from [FramePack Issue #138](https://github.com/lllyasviel/FramePack/issues/138)
   - Extract the `package_installer.zip` file to the FramePack root directory
   - Run `package_installer.bat` and follow the on-screen instructions (usually just press Enter)
   - Restart FramePack and confirm the following messages in the console:
     ```
     Xformers is installed!
     Flash Attn is not installed! (This is normal)
     Sage Attn is installed!
     ```

4. **Launching FramePack-eichi**:
   - Run `run_endframe_ichi.bat` (or the appropriate language version) from the FramePack root directory
   - The WebUI will open in your default browser

5. **Verification**:
   - Upload an image to the WebUI
   - Enter a prompt describing the desired movement
   - Click "Start Generation" to confirm video generation is working

## Linux Setup Instructions

### Supported Linux Distributions
- Ubuntu 22.04 LTS and newer (primary support)
- Other distributions supporting Python 3.10 should also work

### Required Packages and Dependencies
- NVIDIA GPU drivers supporting CUDA 12.6
- Python 3.10.x
- CUDA Toolkit 12.6
- PyTorch 2.6 with CUDA support

### Installation Steps

1. **Setting Up Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Installing PyTorch with CUDA Support**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

3. **Cloning and Setting Up FramePack**:
   ```bash
   git clone https://github.com/lllyasviel/FramePack.git
   cd FramePack
   pip install -r requirements.txt
   ```

4. **Cloning and Setting Up FramePack-eichi**:
   ```bash
   git clone https://github.com/git-ai-code/FramePack-eichi.git
   # Copy necessary files
   cp FramePack-eichi/webui/endframe_ichi.py FramePack/
   cp FramePack-eichi/webui/endframe_ichi_f1.py FramePack/
   cp -r FramePack-eichi/webui/eichi_utils FramePack/
   cp -r FramePack-eichi/webui/lora_utils FramePack/
   cp -r FramePack-eichi/webui/diffusers_helper FramePack/
   cp -r FramePack-eichi/webui/locales FramePack/
   ```

5. **Installing Acceleration Libraries (Optional)**:
   ```bash
   # sage-attention (recommended)
   pip install sageattention==1.0.6
   
   # xformers (if supported)
   pip install xformers
   ```

6. **Launching FramePack-eichi**:
   ```bash
   cd FramePack
   python endframe_ichi.py  # Default is Japanese UI
   python endframe_ichi_f1.py  # Default is Japanese UI
   # For English UI:
   python endframe_ichi.py --lang en
   python endframe_ichi_f1.py --lang en
   # For Traditional Chinese UI:
   python endframe_ichi.py --lang zh-tw
   python endframe_ichi_f1.py --lang zh-tw
   ```

## Docker Setup Instructions

### Prerequisites
- Docker installed on your system
- Docker Compose installed
- NVIDIA Container Toolkit installed for GPU usage

### Docker Setup Process

1. **Using akitaonrails' Docker Implementation**:
   ```bash
   git clone https://github.com/akitaonrails/FramePack-Docker-CUDA.git
   cd FramePack-Docker-CUDA
   mkdir outputs
   mkdir hf_download
   
   # Build Docker image
   docker build -t framepack-torch26-cu124:latest .
   
   # Run container with GPU support
   docker run -it --rm --gpus all -p 7860:7860 \
   -v ./outputs:/app/outputs \
   -v ./hf_download:/app/hf_download \
   framepack-torch26-cu124:latest
   ```

2. **Alternative Docker Compose Setup**:
   - Create a `docker-compose.yml` file with the following content:
   ```yaml
   version: '3'
   services:
     framepack:
       build: .
       ports:
         - "7860:7860"
       volumes:
         - ./outputs:/app/outputs
         - ./hf_download:/app/hf_download
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: all
                 capabilities: [gpu]
       # Language selection (default is English)
       command: ["--lang", "en"]  # Options: "ja" (Japanese), "zh-tw" (Traditional Chinese), "en" (English)
   ```
   
   - Create a `Dockerfile` in the same directory:
   ```dockerfile
   FROM python:3.10-slim
   
   ENV DEBIAN_FRONTEND=noninteractive
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       git \
       wget \
       ffmpeg \
       && rm -rf /var/lib/apt/lists/*
   
   # Set up working directory
   WORKDIR /app
   
   # Clone repositories
   RUN git clone https://github.com/lllyasviel/FramePack.git . && \
       git clone https://github.com/git-ai-code/FramePack-eichi.git /tmp/FramePack-eichi
   
   # Copy FramePack-eichi files (to root directory, same as Linux setup)
   RUN cp /tmp/FramePack-eichi/webui/endframe_ichi.py . && \
       cp /tmp/FramePack-eichi/webui/endframe_ichi_f1.py . && \
       cp -r /tmp/FramePack-eichi/webui/eichi_utils . && \
       cp -r /tmp/FramePack-eichi/webui/lora_utils . && \
       cp -r /tmp/FramePack-eichi/webui/diffusers_helper . && \
       cp -r /tmp/FramePack-eichi/webui/locales . && \
       rm -rf /tmp/FramePack-eichi
   
   # Install PyTorch and dependencies
   RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   RUN pip install -r requirements.txt
   RUN pip install sageattention==1.0.6
   
   # Create output directories
   RUN mkdir -p outputs hf_download
   
   # Set HuggingFace cache directory
   ENV HF_HOME=/app/hf_download
   
   # Expose port for WebUI
   EXPOSE 7860
   
   # Launch FramePack-eichi (from root directory, same as Linux setup)
   ENTRYPOINT ["python", "endframe_ichi.py", "--listen"]
   ```
   
   - Build and run with Docker Compose:
   ```bash
   docker-compose build
   docker-compose up
   ```

3. **Accessing the WebUI**:
   - Once the container is running, the WebUI will be available at http://localhost:7860

4. **GPU Passthrough Configuration**:
   - Ensure NVIDIA Container Toolkit is properly installed
   - The `--gpus all` parameter (or its equivalent in docker-compose.yml) is required for GPU passthrough
   - Check if GPUs are accessible inside the container with:
     ```bash
     docker exec -it [container_id] nvidia-smi
     ```

## macOS (Apple Silicon) Setup Instructions

FramePack-eichi can be used on Apple Silicon Macs through brandon929/FramePack fork, which uses Metal Performance Shaders instead of CUDA.

### Prerequisites
- macOS with Apple Silicon (M1, M2, or M3 chip)
- Homebrew (macOS package manager)
- Python 3.10
- **Memory Requirements**: Minimum 16GB RAM, recommended 32GB+
  - 8GB models are likely to experience severe performance degradation and processing errors
  - 16GB models will be limited to short videos (3-5 seconds) and low resolution settings
  - 32GB+ models allow for comfortable processing (M2/M3 Ultra recommended)

### Installation Steps

1. **Installing Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   - Follow any additional instructions to add Homebrew to your PATH.

2. **Installing Python 3.10**:
   ```bash
   brew install python@3.10
   ```

3. **Cloning the macOS-Compatible Fork**:
   ```bash
   git clone https://github.com/brandon929/FramePack.git
   cd FramePack
   ```

4. **Installing Metal-Enabled PyTorch** (CPU version, Metal support added via PyTorch MPS):
   ```bash
   pip3.10 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
   ```

5. **Installing Dependencies**:
   ```bash
   pip3.10 install -r requirements.txt
   ```

6. **Launching the Web Interface**:
   ```bash
   python3.10 demo_gradio.py --fp32
   ```
   
   The `--fp32` flag is important for Apple Silicon compatibility. M1/M2/M3 processors may not fully support float16 and bfloat16 used in the original models.

7. **After launching**, open a web browser and access the URL displayed in the terminal (usually http://127.0.0.1:7860).

### Special Considerations for Apple Silicon

- **Metal Performance**: 
  - Use the `--fp32` flag for compatibility with Apple Silicon
- **Resolution Settings**: 
  - 16GB RAM: Maximum 416×416 resolution recommended
  - 32GB RAM: Maximum 512×512 resolution recommended
  - 64GB RAM: Maximum 640×640 resolution can be attempted
- **Performance Comparison**:
  - Generation speed is significantly slower compared to NVIDIA GPUs
  - 5-second video generation time comparison:
    - RTX 4090: ~6 minutes
    - M2 Max: ~25-30 minutes
    - M3 Max: ~20-25 minutes
    - M2 Ultra: ~15-20 minutes
    - M3 Ultra: ~12-15 minutes
- **Memory Management**: 
  - Apple Silicon unified memory architecture means GPU/CPU share the same memory pool
  - Monitor "Memory Pressure" in Activity Monitor and reduce settings if compression is high
  - Increased swap usage will drastically reduce performance and impact SSD lifespan
  - Strongly recommended to close other resource-intensive apps during generation
  - Restart the application after extended use to resolve memory leaks

## WSL Setup Instructions

Setting up FramePack-eichi in WSL provides a Linux environment on Windows with GPU acceleration through NVIDIA's WSL drivers.

### Prerequisites
- Windows 10 (version 2004 or later) or Windows 11
- NVIDIA GPU (RTX 30XX, 40XX, or 50XX series recommended, minimum 8GB VRAM)
- Administrator access
- Updated NVIDIA drivers supporting WSL2

### Installation Steps

1. **Installing WSL2**:
   
   Open PowerShell as administrator and run:
   ```powershell
   wsl --install
   ```
   
   This command installs WSL2 with Ubuntu as the default Linux distribution. Restart your computer when prompted.

2. **Verifying WSL2 is Properly Installed**:
   ```powershell
   wsl --status
   ```
   
   Ensure "WSL 2" is shown as the default version.

3. **Updating the WSL Kernel** (if needed):
   ```powershell
   wsl --update
   ```

4. **Installing NVIDIA Drivers for WSL**:
   
   Download and install the latest NVIDIA drivers that support WSL from NVIDIA's website. Do not install NVIDIA drivers inside the WSL environment - WSL uses the Windows drivers.

5. **Launch Ubuntu and Verify GPU Access**:
   
   Launch Ubuntu from the Start menu or run `wsl` in PowerShell, and check NVIDIA GPU detection:
   ```bash
   nvidia-smi
   ```
   
   You should see your GPU information displayed.

6. **Set Up Environment in WSL**:
   ```bash
   # Update package lists
   sudo apt update && sudo apt upgrade -y
   
   # Install Python and development tools
   sudo apt install -y python3.10 python3.10-venv python3-pip git
   
   # Clone FramePack-eichi repository
   git clone https://github.com/git-ai-code/FramePack-eichi.git
   cd FramePack-eichi
   
   # Create and activate virtual environment
   python3.10 -m venv venv
   source venv/bin/activate
   
   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   
   # Install dependencies
   pip install -r requirements.txt
   ```

7. **Launch FramePack-eichi**:
   ```bash
   python endframe_ichi.py
   ```

   You can also specify a language:
   ```bash
   python endframe_ichi.py --lang en  # For English
   ```

8. **Access the Web Interface** by opening a browser in Windows and navigating to the URL displayed in the terminal (typically http://127.0.0.1:7860).

## Anaconda Environment Setup Instructions

### Creating a New Conda Environment

```bash
# Create a new conda environment with Python 3.10
conda create -n framepack-eichi python=3.10
conda activate framepack-eichi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Manual Installation from Source

```bash
# Clone the original FramePack repository
git clone https://github.com/lllyasviel/FramePack.git
cd FramePack

# Clone the FramePack-eichi repository to a temporary location
git clone https://github.com/git-ai-code/FramePack-eichi.git temp_eichi

# Copy extended webui files (to root directory, same as Linux setup)
cp temp_eichi/webui/endframe_ichi.py .
cp temp_eichi/webui/endframe_ichi_f1.py .
cp -r temp_eichi/webui/eichi_utils .
cp -r temp_eichi/webui/lora_utils .
cp -r temp_eichi/webui/diffusers_helper .
cp -r temp_eichi/webui/locales .

# Copy language-specific batch files to the root directory (choose appropriate file)
cp temp_eichi/run_endframe_ichi.bat .  # Japanese (default)
# cp temp_eichi/run_endframe_ichi_en.bat .  # English
# cp temp_eichi/run_endframe_ichi_zh-tw.bat .  # Traditional Chinese

# Install dependencies
pip install -r requirements.txt

# Clean up temporary directory
rm -rf temp_eichi
```

### Special Considerations for Conda

- When installing via conda, you may encounter dependency conflicts with PyTorch packages
- For best results, install PyTorch, torchvision, and torchaudio via pip using the official index URL rather than conda channels
- Optional acceleration packages like xformers, flash-attn, and sageattention should be installed separately after the main environment is created

## Google Colab Setup Instructions

### May 2025 Latest Colab Setup (Most Stable)

The following script provides the most stable setup for Colab's latest environment (as of May 2025). It has been specifically tested in A100 GPU environments.

```python
# Install git if not already installed
!apt-get update && apt-get install -y git

# Clone FramePack repository
!git clone https://github.com/lllyasviel/FramePack.git
%cd FramePack

# Install PyTorch (CUDA-enabled version)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Upgrade Requests and NumPy for Colab environment
!pip install requests==2.32.3 numpy==2.0.0

# Install FramePack dependencies
!pip install -r requirements.txt

# Install SageAttention for speed optimization (optional)
!pip install sageattention==1.0.6

# Start FramePack demo (uncomment to run)
# !python demo_gradio.py --share

# Install FramePack-eichi
!git clone https://github.com/git-ai-code/FramePack-eichi.git tmp
!rsync -av --exclude='diffusers_helper' tmp/webui/ ./
!cp tmp/webui/diffusers_helper/bucket_tools.py diffusers_helper/
!cp tmp/webui/diffusers_helper/memory.py diffusers_helper/
!rm -rf tmp

# Run FramePack-eichi
!python endframe_ichi.py --share
```

> **Important**: The above method specifically copies the `diffusers_helper/bucket_tools.py` file individually. This is necessary to avoid the common "ImportError: cannot import name 'SAFE_RESOLUTIONS' from 'diffusers_helper.bucket_tools'" error.

### Alternative Colab Setup Method

Below is an alternative setup method. Prefer the above method for newer environments.

```python
# Clone FramePack-eichi repository
!git clone https://github.com/git-ai-code/FramePack-eichi.git tmp

# Clone basic FramePack
!git clone https://github.com/lllyasviel/FramePack.git
%cd /content/FramePack

# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
!pip install -r requirements.txt

# Set up eichi extensions (to root directory, same as Linux setup)
!cp /content/tmp/webui/endframe_ichi.py .
!cp /content/tmp/webui/endframe_ichi_f1.py .
!cp -r /content/tmp/webui/eichi_utils .
!cp -r /content/tmp/webui/lora_utils .
!cp -r /content/tmp/webui/diffusers_helper .
!cp -r /content/tmp/webui/locales .
!cp /content/tmp/run_endframe_ichi.bat .

# Set PYTHONPATH environment variable
%env PYTHONPATH=/content/FramePack:$PYTHONPATH

# Launch WebUI with public URL
%cd /content/FramePack
!python endframe_ichi.py --share
```

### Google Drive Integration and Output Configuration

To save generated videos to Google Drive:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set up output directory
import os
OUTPUT_DIR = "/content/drive/MyDrive/FramePack-eichi-outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Launch framepack with output directory specified
!python endframe_ichi.py --share --output_dir={OUTPUT_DIR}
```

### Common Troubleshooting for Colab

1. **'SAFE_RESOLUTIONS' Import Error**:
   ```
   ImportError: cannot import name 'SAFE_RESOLUTIONS' from 'diffusers_helper.bucket_tools'
   ```
   - **Solution**: Use the May 2025 latest setup script above, which includes individual copying of diffusers_helper files

2. **Memory Shortage Errors**:
   ```
   RuntimeError: CUDA out of memory
   ```
   - **Solutions**: 
     - Reduce resolution (e.g., 416×416)
     - Reduce number of keyframes
     - Reduce batch size
     - Adjust GPU inference preserved memory setting

3. **Session Disconnection**:
   - **Solutions**:
     - Avoid long processing times
     - Save progress to Google Drive
     - Keep browser tab active

### VRAM/RAM Considerations for Different Colab Tiers

| Colab Tier | GPU Type | VRAM | Performance | Notes |
|------------|----------|------|-------------|-------|
| Free       | T4       | 16GB | Limited     | Sufficient for basic use with short videos (1-5 seconds) |
| Pro        | A100     | 40GB | Good        | Can handle longer videos and multiple keyframes |
| Pro+       | A100     | 80GB | Excellent   | Best performance, capable of complex generations |

### Optimal Settings for Colab

1. **Hardware Accelerator Settings**:
   - Menu "Runtime" → "Change runtime type" → Set "Hardware accelerator" to "GPU"
   - Pro/Pro+ users should select "High RAM" or "High-memory" option if available

2. **Recommended Batch Size and Resolution Settings**:
   - **T4 GPU (Free)**: Batch size 4, resolution 416x416
   - **A100 GPU (Pro)**: Batch size 8, resolution up to 640x640
   - **A100 GPU (Pro+/High-memory)**: Batch size 16, resolution up to 768x768

## Cloud Environment (AWS/GCP/Azure) Setup Instructions

### AWS EC2 Setup

#### Recommended Instance Types:
- **g4dn.xlarge**: 1 NVIDIA T4 GPU (16GB), 4 vCPU, 16GB RAM
- **g4dn.2xlarge**: 1 NVIDIA T4 GPU (16GB), 8 vCPU, 32GB RAM
- **g5.xlarge**: 1 NVIDIA A10G GPU (24GB), 4 vCPU, 16GB RAM
- **p3.2xlarge**: 1 NVIDIA V100 GPU (16GB), 8 vCPU, 61GB RAM

#### Setup Steps:

1. **Launch EC2 Instance** - Use Deep Learning AMI (Ubuntu) with your selected instance type
2. **Connect to Instance via SSH**:
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```
3. **Update System Packages**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
4. **Clone Repositories**:
   ```bash
   git clone https://github.com/lllyasviel/FramePack.git
   cd FramePack
   git clone https://github.com/git-ai-code/FramePack-eichi.git temp_eichi
   # Copy files to root directory, same as Linux setup
   cp temp_eichi/webui/endframe_ichi.py .
   cp temp_eichi/webui/endframe_ichi_f1.py .
   cp -r temp_eichi/webui/eichi_utils .
   cp -r temp_eichi/webui/lora_utils .
   cp -r temp_eichi/webui/diffusers_helper .
   cp -r temp_eichi/webui/locales .
   cp temp_eichi/run_endframe_ichi_en.bat .  # English version
   rm -rf temp_eichi
   ```
5. **Install Dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```
6. **Configure Security Group** - Allow incoming traffic on port 7860
7. **Run with Public Visibility**:
   ```bash
   python endframe_ichi.py --server --listen --port 7860
   ```
8. **Access the UI** - http://your-instance-ip:7860

### Google Cloud Platform (GCP) Setup

#### Recommended Instance Types:
- **n1-standard-8** + 1 NVIDIA T4 GPU
- **n1-standard-8** + 1 NVIDIA V100 GPU
- **n1-standard-8** + 1 NVIDIA A100 GPU

#### Setup Steps:

1. **Create VM Instance** with Deep Learning VM Image
2. **Enable GPU** and select appropriate GPU type
3. **Connect to Instance via SSH**
4. **Follow the same steps as for AWS EC2** to set up FramePack-eichi
5. **Configure Firewall Rules** - Allow incoming traffic on port 7860

### Microsoft Azure Setup

#### Recommended VM Sizes:
- **Standard_NC6s_v3**: 1 NVIDIA V100 GPU (16GB)
- **Standard_NC4as_T4_v3**: 1 NVIDIA T4 GPU (16GB)
- **Standard_NC24ads_A100_v4**: 1 NVIDIA A100 GPU (80GB)

#### Setup Steps:
1. **Create VM** with Data Science Virtual Machine (Ubuntu)
2. **Connect to VM via SSH**
3. **Follow the same steps as for AWS EC2** to set up FramePack-eichi
4. **Configure Network Security Group** - Allow incoming traffic on port 7860

## Common Errors and Troubleshooting Procedures

### Installation Errors

#### Python Dependency Conflicts
- **Symptoms**: Error messages about incompatible package versions
- **Solutions**: 
  - Explicitly use Python 3.10 (not 3.11, 3.12, or higher)
  - Install PyTorch with the correct CUDA version
  - Create a new virtual environment if dependency errors occur

#### CUDA Installation and Compatibility Issues
- **Symptoms**: "CUDA is not available" errors, warnings about running on CPU
- **Solutions**:
  - Ensure you're using a supported GPU (RTX 30XX, 40XX, or 50XX series recommended)
  - Install the correct CUDA toolkit (12.6 recommended)
  - Troubleshoot in Python:
    ```python
    import torch
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    ```

#### Package Installation Failures
- **Symptoms**: PIP installation errors, wheel build failures
- **Solutions**:
  - Use the one-click installer for Windows (instead of manual installation)
  - For Linux: Install necessary build dependencies
  - If SageAttention installation fails, you can continue without it
  - Use package_installer.zip from Issue #138 to install advanced optimization packages

### Runtime Errors

#### CUDA Out-of-Memory Errors
- **Symptoms**: "CUDA out of memory" error messages, crashes during high-memory phases of generation
- **Solutions**:
  - Increase the `gpu_memory_preservation` value (try values between 6-16GB)
  - Close other GPU-intensive applications
  - Restart and try again
  - Reduce image resolution (512x512 recommended for low VRAM)
  - Set a larger Windows page file (3x physical RAM)
  - Ensure sufficient system RAM (32GB+ recommended)

#### Model Loading Failures
- **Symptoms**: Error messages when loading model shards, process crashes during model initialization
- **Solutions**:
  - Run `update.bat` before starting the application
  - Verify that all models are properly downloaded in the `webui/hf_download` folder
  - Allow automatic download to complete if models are missing (about 30GB)
  - If manually placing models, copy files to the correct `framepack\webui\hf_download` folder

#### WebUI Launch Issues
- **Symptoms**: Gradio interface doesn't appear after launch, browser shows "can't connect" error
- **Solutions**:
  - Try a different port with the `--port XXXX` command line option
  - Ensure no other applications are using port 7860 (Gradio's default)
  - Use the `--inbrowser` option to automatically open the browser
  - Check console output for specific error messages

### Platform-Specific Issues

#### Windows-Specific Issues
- **Symptoms**: Path-related errors, DLL load failures, batch files don't execute properly
- **Solutions**:
  - Install to a short path (e.g., C:\FramePack) to avoid path length limitations
  - Run batch files as administrator if permission issues occur
  - If DLL load errors appear:
    - Install Visual C++ Redistributable packages
    - Check that antivirus software isn't blocking execution

#### Linux-Specific Issues
- **Symptoms**: Missing library errors, package build failures, GUI display issues
- **Solutions**:
  - On Debian/Ubuntu, install required system libraries:
    ```
    sudo apt-get install libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libopenblas-dev
    ```
  - For GPU detection issues, make sure NVIDIA drivers are correctly installed:
    ```
    nvidia-smi
    ```

#### macOS-Specific Issues
- **Symptoms**: Metal/MPS-related errors, low performance on Apple Silicon
- **Solutions**:
  - Run with the `--fp32` flag (M1/M2 may not fully support fp16/bf16)
  - For video format issues, adjust MP4 compression settings to around 16 (default)
  - Acknowledge significantly reduced performance compared to NVIDIA hardware

#### WSL Setup Issues
- **Symptoms**: GPU not detected in WSL, extremely low performance in WSL
- **Solutions**:
  - Ensure you're using WSL2 (not WSL1): `wsl --set-version <Distro> 2`
  - Install the dedicated NVIDIA drivers for WSL
  - Create a `.wslconfig` file in your Windows user directory:
    ```
    [wsl2]
    memory=16GB  # Adjust based on your system
    processors=8  # Adjust based on your system
    gpumemory=8GB  # Adjust based on your GPU
    ```

### Performance Issues

#### Slow Generation Times and Optimization Techniques
- **Symptoms**: Excessively long generation times, lower-than-expected performance compared to benchmarks
- **Solutions**:
  - Install optimization libraries:
    - Download package_installer.zip from Issue #138 and run package_installer.bat
    - This will install xformers, flash-attn, and sage-attn where possible
  - Enable teacache for faster (but potentially lower quality) generation
  - Close other GPU-intensive applications
  - Reduce resolution for faster generation (at the cost of quality)

#### Memory Leaks and Management
- **Symptoms**: Increasing memory usage over time, degraded performance across multiple generations
- **Solutions**:
  - Restart the application between long generation sessions
  - Monitor GPU memory usage:
    ```
    nvidia-smi -l 1
    ```
  - Restart Python process if CPU/memory leaks occur
  - Use explicit model unloading when switching settings
  - Don't load multiple LoRAs simultaneously if not needed

## Information Sources

1. Official Repositories:
   - FramePack-eichi: https://github.com/git-ai-code/FramePack-eichi
   - Original FramePack: https://github.com/lllyasviel/FramePack

2. Community Resources:
   - FramePack Docker Implementation: https://github.com/akitaonrails/FramePack-Docker-CUDA
   - Apple Silicon Compatible Fork: https://github.com/brandon929/FramePack

3. Official Documentation:
   - README and wiki of the FramePack-eichi GitHub repository
   - Developer comments in GitHub Issues

4. Troubleshooting Resources:
   - FramePack Issue #138 (Acceleration Libraries): https://github.com/lllyasviel/FramePack/issues/138
   - WSL CUDA Configuration Documentation: https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl

This guide provides comprehensive setup instructions for FramePack-eichi and the best practices for operation in various environments. Choose the setup path optimal for your environment and refer to the troubleshooting procedures as needed.
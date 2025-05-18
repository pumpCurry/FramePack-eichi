# FramePack-eichi 設置指南：各環境構建步驟的綜合手冊 | [日本語](README_setup.md) | [English](README_setup_en.md) | [Русский](README_setup_ru.md)

> **免責聲明**：本文檔彙集了互聯網上的信息，不保證在所有環境中都能正常運行。由於環境和版本的差異，此處描述的步驟可能無法正常運行。請根據您的具體環境進行必要的調整。同時，建議您始終參考官方存儲庫中的最新信息。

FramePack-eichi 是一個 AI 視頻生成系統，可使用文本提示從單一圖像創建短視頻。它是 Stanford 大學 Lvmin Zhang 和 Maneesh Agrawala 開發的原始 FramePack 的分支，增加了許多功能和擴展。本指南提供各環境下的準確設置步驟、系統要求和故障排除提示。

## 系統要求

### RAM 要求
- **最低**: 16GB（可運行但性能受限）
- **推薦**: 32GB（足夠標準操作）
- **最佳**: 64GB（適合長視頻、LoRA 使用和高解析度處理）
- 如果 RAM 不足，系統將使用 SSD 交換空間，這可能會縮短 SSD 的使用壽命

### VRAM 要求
- **最低**: 8GB VRAM（FramePack-eichi 的推薦最低值）
- **低 VRAM 模式**: 自動啟用並高效管理內存
  - 可通過 `gpu_memory_preservation` 設置調整（默認: 10GB）
  - 較低值 = 處理時可用 VRAM 更多 = 更快但風險更高
  - 較高值 = 處理時可用 VRAM 更少 = 更慢但更穩定
- **高 VRAM 模式**: 當檢測到超過 100GB 的可用 VRAM 時自動啟用
  - 模型常駐 GPU 內存（速度提升約 20%）
  - 無需定期加載/卸載模型

### CPU 要求
- 未明確指定最低 CPU 型號
- **推薦**: 8 核或更多的現代多核 CPU
- CPU 性能影響加載時間和預處理/後處理
- 大部分實際生成處理在 GPU 上運行

### 存儲要求
- **應用程序代碼**: 通常 1-2GB
- **模型**: 約 30GB（首次啟動時自動下載）
- **輸出和臨時文件**: 取決於視頻長度、解析度和壓縮設置
- **總推薦容量**: 150GB 或更多
- 推薦使用 SSD 進行頻繁的讀寫操作

### 支持的 GPU 型號
- **官方支持**: NVIDIA RTX 30XX、40XX、50XX 系列（支持 fp16 和 bf16 數據格式）
- **最低推薦**: RTX 3060（或同等 8GB+ VRAM）
- **確認能運行**: RTX 3060、3070Ti、4060Ti、4090
- **非官方/未測試**: GTX 10XX/20XX 系列
- **AMD GPU**: 未明確提及支持
- **Intel GPU**: 未明確提及支持

## Windows 設置說明

### 前提條件
- Windows 10/11
- 支持 CUDA 12.6 的 NVIDIA GPU 驅動
- Python 3.10.x
- 7-Zip（用於解壓安裝包）

### 逐步說明
1. **安裝基本 FramePack**:
   - 前往[官方 FramePack 倉庫](https://github.com/lllyasviel/FramePack)
   - 點擊「Click Here to Download One-Click Package (CUDA 12.6 + PyTorch 2.6)」
   - 下載並解壓 7z 包到選擇的位置
   - 運行 `update.bat`（獲取最新錯誤修復非常重要）
   - 運行 `run.bat` 首次啟動 FramePack
   - 首次運行時將自動下載所需模型（約 30GB）

2. **安裝 FramePack-eichi**:
   - 克隆或下載 [FramePack-eichi 倉庫](https://github.com/git-ai-code/FramePack-eichi)
   - 將適合語言的批處理文件（日語用 `run_endframe_ichi.bat`、英語用 `run_endframe_ichi_en.bat`、繁體中文用 `run_endframe_ichi_zh-tw.bat`、俄語用 `run_endframe_ichi_ru.bat`）複製到 FramePack 根目錄
   - F1版本請使用 `run_endframe_ichi_f1.bat`（或語言版 `run_endframe_ichi_en_f1.bat`、`run_endframe_ichi_zh-tw_f1.bat`、`run_endframe_ichi_f1_ru.bat`）
   - 單幀推論請使用 `run_oneframe_ichi.bat`（或語言版 `run_oneframe_ichi_en.bat`、`run_oneframe_ichi_zh-tw.bat`、`run_oneframe_ichi_ru.bat`）
   - 將以下文件/文件夾從 FramePack-eichi 複製到 FramePack 的 `webui` 文件夾：
     - `endframe_ichi.py`
     - `eichi_utils` 文件夾（包含 `lora_preset_manager.py`、`model_downloader.py`、`vae_settings.py` 等 - v1.9.3 新增）
     - `lora_utils` 文件夾
     - `diffusers_helper` 文件夾
     - `locales` 文件夾（包含 `ru.json` 俄語翻譯文件 - v1.9.3 新增）

3. **安裝加速庫（可選但推薦）**:
   - 從 [FramePack Issue #138](https://github.com/lllyasviel/FramePack/issues/138) 下載加速包安裝程序
   - 將 `package_installer.zip` 文件解壓到 FramePack 根目錄
   - 運行 `package_installer.bat` 並按照屏幕上的說明操作（通常只需按 Enter 鍵）
   - 重啟 FramePack 並確認控制台中顯示以下信息：
     ```
     Xformers is installed!
     Flash Attn is not installed! （這是正常的）
     Sage Attn is installed!
     ```

4. **啟動 FramePack-eichi**:
   - 從 FramePack 根目錄運行 `run_endframe_ichi.bat`（或相應的語言版本）
   - WebUI 將在默認瀏覽器中打開

5. **驗證**:
   - 上傳圖像到 WebUI
   - 輸入描述所需動作的提示詞
   - 點擊「開始生成」確認視頻生成正常工作

## Linux 設置說明

### 支持的 Linux 發行版
- Ubuntu 22.04 LTS 及更新版本（主要支持）
- 其他支持 Python 3.10 的發行版也應該可以工作

### 所需包和依賴項
- 支持 CUDA 12.6 的 NVIDIA GPU 驅動
- Python 3.10.x
- CUDA Toolkit 12.6
- 支持 CUDA 的 PyTorch 2.6

### 安裝步驟

1. **設置 Python 環境**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **安裝支持 CUDA 的 PyTorch**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

3. **克隆和設置 FramePack**:
   ```bash
   git clone https://github.com/lllyasviel/FramePack.git
   cd FramePack
   pip install -r requirements.txt
   ```

4. **克隆和設置 FramePack-eichi**:
   ```bash
   git clone https://github.com/git-ai-code/FramePack-eichi.git
   # 複製必要文件
   cp FramePack-eichi/webui/endframe_ichi.py FramePack/
   cp FramePack-eichi/webui/endframe_ichi_f1.py FramePack/
   cp -r FramePack-eichi/webui/eichi_utils FramePack/
   cp -r FramePack-eichi/webui/lora_utils FramePack/
   cp -r FramePack-eichi/webui/diffusers_helper FramePack/
   cp -r FramePack-eichi/webui/locales FramePack/
   ```

5. **安裝加速庫（可選）**:
   ```bash
   # sage-attention（推薦）
   pip install sageattention==1.0.6
   
   # xformers（如果支持）
   pip install xformers
   ```

6. **啟動 FramePack-eichi**:
   ```bash
   cd FramePack
   python endframe_ichi.py  # 默認為日語 UI
   python endframe_ichi_f1.py  # 默認為日語 UI
   # 英語 UI：
   python endframe_ichi.py --lang en
   python endframe_ichi_f1.py --lang en
   # 繁體中文 UI：
   python endframe_ichi.py --lang zh-tw
   python endframe_ichi_f1.py --lang zh-tw
   ```

## Docker 設置說明

### 前提條件
- 系統已安裝 Docker
- 系統已安裝 Docker Compose
- 已安裝 NVIDIA Container Toolkit 用於 GPU 使用

### Docker 設置流程

1. **使用 akitaonrails 的 Docker 實現**:
   ```bash
   git clone https://github.com/akitaonrails/FramePack-Docker-CUDA.git
   cd FramePack-Docker-CUDA
   mkdir outputs
   mkdir hf_download
   
   # 構建 Docker 鏡像
   docker build -t framepack-torch26-cu124:latest .
   
   # 運行帶 GPU 支持的容器
   docker run -it --rm --gpus all -p 7860:7860 \
   -v ./outputs:/app/outputs \
   -v ./hf_download:/app/hf_download \
   framepack-torch26-cu124:latest
   ```

2. **替代 Docker Compose 設置**:
   - 創建包含以下內容的 `docker-compose.yml` 文件：
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
       # 語言選擇（默認為英語）
       command: ["--lang", "zh-tw"]  # 選項: "ja"（日語）, "zh-tw"（繁體中文）, "en"（英語）
   ```
   
   - 在同一目錄中創建 `Dockerfile`：
   ```dockerfile
   FROM python:3.10-slim
   
   ENV DEBIAN_FRONTEND=noninteractive
   
   # 安裝系統依賴
   RUN apt-get update && apt-get install -y \
       git \
       wget \
       ffmpeg \
       && rm -rf /var/lib/apt/lists/*
   
   # 設置工作目錄
   WORKDIR /app
   
   # 克隆倉庫
   RUN git clone https://github.com/lllyasviel/FramePack.git . && \
       git clone https://github.com/git-ai-code/FramePack-eichi.git /tmp/FramePack-eichi
   
   # 複製 FramePack-eichi 文件（复制到根目錄，與 Linux 設置相同）
   RUN cp /tmp/FramePack-eichi/webui/endframe_ichi.py . && \
       cp /tmp/FramePack-eichi/webui/endframe_ichi_f1.py . && \
       cp -r /tmp/FramePack-eichi/webui/eichi_utils . && \
       cp -r /tmp/FramePack-eichi/webui/lora_utils . && \
       cp -r /tmp/FramePack-eichi/webui/diffusers_helper . && \
       cp -r /tmp/FramePack-eichi/webui/locales . && \
       rm -rf /tmp/FramePack-eichi
   
   # 安裝 PyTorch 和依賴項
   RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   RUN pip install -r requirements.txt
   RUN pip install sageattention==1.0.6
   
   # 創建輸出目錄
   RUN mkdir -p outputs hf_download
   
   # 設置 HuggingFace 緩存目錄
   ENV HF_HOME=/app/hf_download
   
   # 暴露 WebUI 端口
   EXPOSE 7860
   
   # 啟動 FramePack-eichi（從根目錄，與 Linux 設置相同）
   ENTRYPOINT ["python", "endframe_ichi.py", "--listen"]
   ```
   
   - 使用 Docker Compose 構建和運行：
   ```bash
   docker-compose build
   docker-compose up
   ```

3. **訪問 WebUI**:
   - 容器運行後，WebUI 將在 http://localhost:7860 可用

4. **GPU 透傳配置**:
   - 確保 NVIDIA Container Toolkit 已正確安裝
   - GPU 透傳需要 `--gpus all` 參數（或 docker-compose.yml 中的等效項）
   - 使用以下命令檢查容器內是否可訪問 GPU：
     ```bash
     docker exec -it [container_id] nvidia-smi
     ```

## macOS（Apple Silicon）設置說明

FramePack-eichi 可通過 brandon929/FramePack 分支在 Apple Silicon Mac 上使用，該分支使用 Metal Performance Shaders 代替 CUDA。

### 前提條件
- 配備 Apple Silicon（M1、M2 或 M3 芯片）的 macOS
- Homebrew（macOS 包管理器）
- Python 3.10
- **內存要求**: 最低 16GB RAM，推薦 32GB+
  - 8GB 型號可能會遇到嚴重性能下降和處理錯誤
  - 16GB 型號將僅限於短視頻（3-5 秒）和低解析度設置
  - 32GB+ 型號允許舒適處理（推薦 M2/M3 Ultra）

### 安裝步驟

1. **安裝 Homebrew**（如果尚未安裝）:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   - 按照任何附加說明將 Homebrew 添加到您的 PATH。

2. **安裝 Python 3.10**:
   ```bash
   brew install python@3.10
   ```

3. **克隆 macOS 兼容分支**:
   ```bash
   git clone https://github.com/brandon929/FramePack.git
   cd FramePack
   ```

4. **安裝支持 Metal 的 PyTorch**（CPU 版本，通過 PyTorch MPS 添加 Metal 支持）:
   ```bash
   pip3.10 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
   ```

5. **安裝依賴項**:
   ```bash
   pip3.10 install -r requirements.txt
   ```

6. **啟動 Web 界面**:
   ```bash
   python3.10 demo_gradio.py --fp32
   ```
   
   `--fp32` 標誌對於 Apple Silicon 兼容性很重要。M1/M2/M3 處理器可能不完全支持原始模型中使用的 float16 和 bfloat16。

7. **啟動後**，打開 Web 瀏覽器並訪問終端中顯示的 URL（通常是 http://127.0.0.1:7860）。

### Apple Silicon 特殊考慮事項

- **Metal 性能**：
  - 在 Apple Silicon 上運行時使用 `--fp32` 標誌以確保兼容性
- **解析度設置**：
  - 16GB RAM：建議最大 416×416 解析度
  - 32GB RAM：建議最大 512×512 解析度
  - 64GB RAM：可嘗試最大 640×640 解析度
- **性能比較**：
  - 生成速度與 NVIDIA GPU 相比明顯較慢
  - 5 秒視頻生成時間比較：
    - RTX 4090：約 6 分鐘
    - M2 Max：約 25-30 分鐘
    - M3 Max：約 20-25 分鐘
    - M2 Ultra：約 15-20 分鐘
    - M3 Ultra：約 12-15 分鐘
- **內存管理**：
  - Apple Silicon 統一內存架構意味著 GPU/CPU 共享同一內存池
  - 在活動監視器中監控「內存壓力」，如果壓縮率高則降低設置
  - 增加的交換使用將大幅降低性能並影響 SSD 壽命
  - 強烈建議在生成過程中關閉其他資源密集型應用
  - 長時間使用後重啟應用程序以解決內存洩漏

## WSL 設置說明

在 WSL 中設置 FramePack-eichi 可通過 NVIDIA 的 WSL 驅動提供具有 GPU 加速的 Windows 上的 Linux 環境。

### 前提條件
- Windows 10（2004 版或更高版本）或 Windows 11
- NVIDIA GPU（推薦 RTX 30XX、40XX 或 50XX 系列，最低 8GB VRAM）
- 管理員訪問權限
- 支持 WSL2 的更新 NVIDIA 驅動

### 安裝步驟

1. **安裝 WSL2**:
   
   以管理員身份打開 PowerShell 並運行：
   ```powershell
   wsl --install
   ```
   
   此命令以 Ubuntu 為默認 Linux 發行版安裝 WSL2。按提示重啟計算機。

2. **驗證 WSL2 是否正確安裝**:
   ```powershell
   wsl --status
   ```
   
   確保「WSL 2」顯示為默認版本。

3. **更新 WSL 內核**（如需要）:
   ```powershell
   wsl --update
   ```

4. **安裝適用於 WSL 的 NVIDIA 驅動**:
   
   從 NVIDIA 網站下載並安裝支持 WSL 的最新 NVIDIA 驅動。不要在 WSL 環境內安裝 NVIDIA 驅動 - WSL 使用 Windows 驅動。

5. **啟動 Ubuntu 並驗證 GPU 訪問**:
   
   從開始菜單啟動 Ubuntu 或在 PowerShell 中運行 `wsl`，並檢查 NVIDIA GPU 檢測：
   ```bash
   nvidia-smi
   ```
   
   應顯示您的 GPU 信息。

6. **在 WSL 中設置環境**:
   ```bash
   # 更新包列表
   sudo apt update && sudo apt upgrade -y
   
   # 安裝 Python 和開發工具
   sudo apt install -y python3.10 python3.10-venv python3-pip git
   
   # 克隆 FramePack-eichi 倉庫
   git clone https://github.com/git-ai-code/FramePack-eichi.git
   cd FramePack-eichi
   
   # 創建並激活虛擬環境
   python3.10 -m venv venv
   source venv/bin/activate
   
   # 安裝支持 CUDA 的 PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   
   # 安裝依賴項
   pip install -r requirements.txt
   ```

7. **啟動 FramePack-eichi**:
   ```bash
   python endframe_ichi.py
   ```

   您還可以指定語言：
   ```bash
   python endframe_ichi.py --lang zh-tw  # 使用繁體中文
   ```

8. **訪問 Web 界面**，在 Windows 中打開瀏覽器並導航到終端中顯示的 URL（通常是 http://127.0.0.1:7860）。

## Anaconda 環境設置說明

### 創建新的 Conda 環境

```bash
# 使用 Python 3.10 創建新的 conda 環境
conda create -n framepack-eichi python=3.10
conda activate framepack-eichi

# 安裝支持 CUDA 的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 從源代碼手動安裝

```bash
# 克隆原始 FramePack 倉庫
git clone https://github.com/lllyasviel/FramePack.git
cd FramePack

# 將 FramePack-eichi 倉庫克隆到臨時位置
git clone https://github.com/git-ai-code/FramePack-eichi.git temp_eichi

# 複製擴展的 webui 文件（到根目錄，與 Linux 設置相同）
cp temp_eichi/webui/endframe_ichi.py .
cp temp_eichi/webui/endframe_ichi_f1.py .
cp -r temp_eichi/webui/eichi_utils .
cp -r temp_eichi/webui/lora_utils .
cp -r temp_eichi/webui/diffusers_helper .
cp -r temp_eichi/webui/locales .

# 將特定語言的批處理文件複製到根目錄（選擇適當的文件）
cp temp_eichi/run_endframe_ichi_zh-tw.bat .  # 繁體中文
# cp temp_eichi/run_endframe_ichi.bat .  # 日語（默認）
# cp temp_eichi/run_endframe_ichi_en.bat .  # 英語

# 安裝依賴項
pip install -r requirements.txt

# 清理臨時目錄
rm -rf temp_eichi
```

### Conda 特殊考慮事項

- 通過 conda 安裝時，可能會遇到與 PyTorch 包的依賴衝突
- 為獲得最佳結果，請通過 pip 使用官方索引 URL 而非 conda 頻道安裝 PyTorch、torchvision 和 torchaudio
- 可選的加速包如 xformers、flash-attn 和 sageattention 應在創建主環境後單獨安裝

## Google Colab 設置說明

### 2025 年 5 月最新 Colab 設置（最穩定）

以下腳本提供了 Colab 最新環境（截至 2025 年 5 月）的最穩定設置。它已在 A100 GPU 環境中進行了專門測試。

```python
# 如果尚未安裝 git，則安裝
!apt-get update && apt-get install -y git

# 克隆 FramePack 倉庫
!git clone https://github.com/lllyasviel/FramePack.git
%cd FramePack

# 安裝 PyTorch（支持 CUDA 的版本）
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 為 Colab 環境升級 Requests 和 NumPy
!pip install requests==2.32.3 numpy==2.0.0

# 安裝 FramePack 依賴項
!pip install -r requirements.txt

# 安裝 SageAttention 以進行速度優化（可選）
!pip install sageattention==1.0.6

# 啟動 FramePack 演示（取消註釋以運行）
# !python demo_gradio.py --share

# 安裝 FramePack-eichi
!git clone https://github.com/git-ai-code/FramePack-eichi.git tmp
!rsync -av --exclude='diffusers_helper' tmp/webui/ ./
!cp tmp/webui/diffusers_helper/bucket_tools.py diffusers_helper/
!cp tmp/webui/diffusers_helper/memory.py diffusers_helper/
!rm -rf tmp

# 運行 FramePack-eichi
!python endframe_ichi.py --share --lang zh-tw
```

> **重要**：上述方法專門單獨複製 `diffusers_helper/bucket_tools.py` 文件。這是避免常見的「ImportError: cannot import name 'SAFE_RESOLUTIONS' from 'diffusers_helper.bucket_tools'」錯誤所必需的。

### 替代 Colab 設置方法

以下是替代設置方法。對於較新的環境，優先使用上述方法。

```python
# 克隆 FramePack-eichi 倉庫
!git clone https://github.com/git-ai-code/FramePack-eichi.git tmp

# 克隆基本 FramePack
!git clone https://github.com/lllyasviel/FramePack.git
%cd /content/FramePack

# 安裝依賴項
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
!pip install -r requirements.txt

# 設置 eichi 擴展（到根目錄，與 Linux 設置相同）
!cp /content/tmp/webui/endframe_ichi.py .
!cp /content/tmp/webui/endframe_ichi_f1.py .
!cp -r /content/tmp/webui/eichi_utils .
!cp -r /content/tmp/webui/lora_utils .
!cp -r /content/tmp/webui/diffusers_helper .
!cp -r /content/tmp/webui/locales .
!cp /content/tmp/run_endframe_ichi.bat .

# 設置 PYTHONPATH 環境變量
%env PYTHONPATH=/content/FramePack:$PYTHONPATH

# 使用公共 URL 啟動 WebUI
%cd /content/FramePack
!python endframe_ichi.py --share --lang zh-tw
```

### Google Drive 集成和輸出配置

要將生成的視頻保存到 Google Drive：

```python
# 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 設置輸出目錄
import os
OUTPUT_DIR = "/content/drive/MyDrive/FramePack-eichi-outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 指定輸出目錄啟動 framepack
!python endframe_ichi.py --share --output_dir={OUTPUT_DIR} --lang zh-tw
```

### Colab 常見故障排除

1. **'SAFE_RESOLUTIONS' 導入錯誤**:
   ```
   ImportError: cannot import name 'SAFE_RESOLUTIONS' from 'diffusers_helper.bucket_tools'
   ```
   - **解決方案**: 使用上面的 2025 年 5 月最新設置腳本，其中包括 diffusers_helper 文件的單獨複製

2. **內存不足錯誤**:
   ```
   RuntimeError: CUDA out of memory
   ```
   - **解決方案**：
     - 降低解析度（例如，416×416）
     - 減少關鍵幀數量
     - 減小批處理大小
     - 調整 GPU 推理保留內存設置

3. **會話斷開**:
   - **解決方案**：
     - 避免長時間處理
     - 將進度保存到 Google Drive
     - 保持瀏覽器標籤活動

### 不同 Colab 層級的 VRAM/RAM 考慮事項

| Colab 層級 | GPU 類型 | VRAM | 性能 | 備註 |
|------------|----------|------|-------------|-------|
| 免費       | T4       | 16GB | 有限     | 足夠基本使用，短視頻（1-5 秒） |
| Pro        | A100     | 40GB | 良好        | 可處理較長視頻和多個關鍵幀 |
| Pro+       | A100     | 80GB | 優秀   | 最佳性能，能夠進行複雜生成 |

### Colab 最優設置

1. **硬件加速器設置**:
   - 菜單「運行時」→「更改運行時類型」→ 將「硬件加速器」設置為「GPU」
   - Pro/Pro+ 用戶應選擇「高 RAM」或「高內存」選項（如果可用）

2. **推薦批處理大小和解析度設置**:
   - **T4 GPU（免費）**: 批處理大小 4，解析度 416x416
   - **A100 GPU（Pro）**: 批處理大小 8，解析度最高 640x640
   - **A100 GPU（Pro+/高內存）**: 批處理大小 16，解析度最高 768x768

## 雲環境（AWS/GCP/Azure）設置說明

### AWS EC2 設置

#### 推薦實例類型：
- **g4dn.xlarge**: 1 NVIDIA T4 GPU (16GB), 4 vCPU, 16GB RAM
- **g4dn.2xlarge**: 1 NVIDIA T4 GPU (16GB), 8 vCPU, 32GB RAM
- **g5.xlarge**: 1 NVIDIA A10G GPU (24GB), 4 vCPU, 16GB RAM
- **p3.2xlarge**: 1 NVIDIA V100 GPU (16GB), 8 vCPU, 61GB RAM

#### 設置步驟：

1. **啟動 EC2 實例** - 使用選定實例類型的 Deep Learning AMI（Ubuntu）
2. **通過 SSH 連接到實例**:
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```
3. **更新系統包**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
4. **克隆倉庫**:
   ```bash
   git clone https://github.com/lllyasviel/FramePack.git
   cd FramePack
   git clone https://github.com/git-ai-code/FramePack-eichi.git temp_eichi
   # 複製文件到根目錄，與 Linux 設置相同
   cp temp_eichi/webui/endframe_ichi.py .
   cp temp_eichi/webui/endframe_ichi_f1.py .
   cp -r temp_eichi/webui/eichi_utils .
   cp -r temp_eichi/webui/lora_utils .
   cp -r temp_eichi/webui/diffusers_helper .
   cp -r temp_eichi/webui/locales .
   cp temp_eichi/run_endframe_ichi_zh-tw.bat .  # 繁體中文版
   rm -rf temp_eichi
   ```
5. **安裝依賴項**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```
6. **配置安全組** - 允許端口 7860 上的入站流量
7. **以公共可見性運行**:
   ```bash
   python endframe_ichi.py --server --listen --port 7860 --lang zh-tw
   ```
8. **訪問 UI** - http://your-instance-ip:7860

### Google Cloud Platform (GCP) 設置

#### 推薦實例類型：
- **n1-standard-8** + 1 NVIDIA T4 GPU
- **n1-standard-8** + 1 NVIDIA V100 GPU
- **n1-standard-8** + 1 NVIDIA A100 GPU

#### 設置步驟：

1. **使用 Deep Learning VM 映像創建 VM 實例**
2. **啟用 GPU** 並選擇適當的 GPU 類型
3. **通過 SSH 連接到實例**
4. **按照與 AWS EC2 相同的步驟**設置 FramePack-eichi
5. **配置防火牆規則** - 允許端口 7860 上的入站流量

### Microsoft Azure 設置

#### 推薦 VM 大小：
- **Standard_NC6s_v3**: 1 NVIDIA V100 GPU (16GB)
- **Standard_NC4as_T4_v3**: 1 NVIDIA T4 GPU (16GB)
- **Standard_NC24ads_A100_v4**: 1 NVIDIA A100 GPU (80GB)

#### 設置步驟：
1. **使用 Data Science Virtual Machine (Ubuntu) 創建 VM**
2. **通過 SSH 連接到 VM**
3. **按照與 AWS EC2 相同的步驟**設置 FramePack-eichi
4. **配置網絡安全組** - 允許端口 7860 上的入站流量

## 常見錯誤和故障排除程序

### 安裝錯誤

#### Python 依賴衝突
- **症狀**：關於不兼容包版本的錯誤消息
- **解決方案**：
  - 明確使用 Python 3.10（不要使用 3.11、3.12 或更高版本）
  - 使用正確的 CUDA 版本安裝 PyTorch
  - 如果出現依賴錯誤，創建新的虛擬環境

#### CUDA 安裝和兼容性問題
- **症狀**：「CUDA is not available」錯誤，關於在 CPU 上運行的警告
- **解決方案**：
  - 確保使用支持的 GPU（推薦 RTX 30XX、40XX 或 50XX 系列）
  - 安裝正確的 CUDA 工具包（推薦 12.6）
  - 在 Python 中進行故障排除：
    ```python
    import torch
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    ```

#### 包安裝失敗
- **症狀**：PIP 安裝錯誤，wheel 構建失敗
- **解決方案**：
  - 對於 Windows，使用一鍵安裝程序（而不是手動安裝）
  - 對於 Linux：安裝必要的構建依賴項
  - 如果 SageAttention 安裝失敗，可以繼續而不使用它
  - 使用 Issue #138 中的 package_installer.zip 安裝高級優化包

### 運行時錯誤

#### CUDA 內存不足錯誤
- **症狀**：「CUDA out of memory」錯誤消息，在生成的高內存階段崩潰
- **解決方案**：
  - 增加 `gpu_memory_preservation` 值（嘗試 6-16GB 之間的值）
  - 關閉其他 GPU 密集型應用程序
  - 重啟並重試
  - 降低圖像解析度（低 VRAM 推薦 512x512）
  - 設置更大的 Windows 頁面文件（物理 RAM 的 3 倍）
  - 確保足夠的系統 RAM（推薦 32GB+）

#### 模型加載失敗
- **症狀**：加載模型分片時的錯誤消息，模型初始化期間的進程崩潰
- **解決方案**：
  - 在啟動應用程序前運行 `update.bat`
  - 驗證所有模型是否在 `webui/hf_download` 文件夾中正確下載
  - 如果模型缺失，允許自動下載完成（約 30GB）
  - 如果手動放置模型，將文件複製到正確的 `framepack\webui\hf_download` 文件夾

#### WebUI 啟動問題
- **症狀**：啟動後 Gradio 界面不出現，瀏覽器顯示「無法連接」錯誤
- **解決方案**：
  - 使用 `--port XXXX` 命令行選項嘗試不同端口
  - 確保沒有其他應用程序使用端口 7860（Gradio 的默認端口）
  - 使用 `--inbrowser` 選項自動打開瀏覽器
  - 檢查控制台輸出是否有特定錯誤消息

### 平台特定問題

#### Windows 特定問題
- **症狀**：路徑相關錯誤，DLL 加載失敗，批處理文件無法正確執行
- **解決方案**：
  - 安裝到短路徑（例如，C:\FramePack）以避免路徑長度限制
  - 如果出現權限問題，以管理員身份運行批處理文件
  - 如果出現 DLL 加載錯誤：
    - 安裝 Visual C++ Redistributable 包
    - 檢查防病毒軟件是否阻止執行

#### Linux 特定問題
- **症狀**：找不到庫錯誤，包構建失敗，GUI 顯示問題
- **解決方案**：
  - 在 Debian/Ubuntu 上，安裝所需的系統庫：
    ```
    sudo apt-get install libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libopenblas-dev
    ```
  - 對於 GPU 檢測問題，確保正確安裝 NVIDIA 驅動：
    ```
    nvidia-smi
    ```

#### macOS 特定問題
- **症狀**：Metal/MPS 相關錯誤，Apple Silicon 上的低性能
- **解決方案**：
  - 使用 `--fp32` 標誌運行（M1/M2 可能不完全支持 fp16/bf16）
  - 對於視頻格式問題，將 MP4 壓縮設置調整為約 16（默認）
  - 承認與 NVIDIA 硬件相比性能顯著降低

#### WSL 設置問題
- **症狀**：WSL 中未檢測到 GPU，WSL 中極低的性能
- **解決方案**：
  - 確保使用 WSL2（而非 WSL1）：`wsl --set-version <Distro> 2`
  - 安裝專用於 WSL 的 NVIDIA 驅動
  - 在 Windows 用戶目錄中創建 `.wslconfig` 文件：
    ```
    [wsl2]
    memory=16GB  # 根據系統調整
    processors=8  # 根據系統調整
    gpumemory=8GB  # 根據 GPU 調整
    ```

### 性能問題

#### 生成時間慢和優化技術
- **症狀**：生成時間過長，與基準相比性能低於預期
- **解決方案**：
  - 安裝優化庫：
    - 從 Issue #138 下載 package_installer.zip 並運行 package_installer.bat
    - 這將在可能的情況下安裝 xformers、flash-attn 和 sage-attn
  - 啟用 teacache 以獲得更快（但可能質量較低）的生成
  - 關閉其他 GPU 密集型應用程序
  - 降低解析度以加快生成（以質量為代價）

#### 內存洩漏和管理
- **症狀**：隨時間增加的內存使用量，多次生成後性能下降
- **解決方案**：
  - 在長時間生成會話之間重啟應用程序
  - 監控 GPU 內存使用：
    ```
    nvidia-smi -l 1
    ```
  - 如果出現 CPU/內存洩漏，重啟 Python 進程
  - 切換設置時使用顯式模型卸載
  - 如果不需要，不要同時加載多個 LoRA

## 信息來源

1. 官方倉庫：
   - FramePack-eichi: https://github.com/git-ai-code/FramePack-eichi
   - 原始 FramePack: https://github.com/lllyasviel/FramePack

2. 社區資源：
   - FramePack Docker 實現：https://github.com/akitaonrails/FramePack-Docker-CUDA
   - Apple Silicon 兼容分支：https://github.com/brandon929/FramePack

3. 官方文檔：
   - FramePack-eichi GitHub 倉庫的 README 和 wiki
   - GitHub Issues 中的開發者評論

4. 故障排除資源：
   - FramePack Issue #138（加速庫）：https://github.com/lllyasviel/FramePack/issues/138
   - WSL CUDA 配置文檔：https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl

本指南提供了 FramePack-eichi 的全面設置說明和各環境下的最佳操作實踐。選擇最適合您環境的設置路徑，並根據需要參考故障排除程序。
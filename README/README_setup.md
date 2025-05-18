# FramePack-eichi セットアップガイド: すべての環境向け総合インストールマニュアル | [English](README_setup_en.md) | [繁體中文](README_setup_zh.md) | [Русский](README_setup_ru.md)

> **免責事項**: このドキュメントはインターネットから収集した情報をまとめたものであり、すべての環境での機能を保証するものではありません。環境やバージョンの違いにより、記載された手順が正常に機能しない場合があります。必要に応じて、ご使用の環境に合わせて調整してください。また、公式リポジトリの最新情報を常に参照することをお勧めします。

FramePack-eichiは、テキストプロンプトを使用して1枚の画像から短い動画を作成するAI動画生成システムです。これはスタンフォード大学のLvmin ZhangとManeesh Agrawalaによって開発されたオリジナルのFramePackをフォークし、多数の追加機能と強化を施したものです。このガイドでは、各環境に対する正確なセットアップ手順、システム要件、およびトラブルシューティングのヒントを提供します。

## システム要件

### RAM要件
- **最小**: 16GB（動作するが、パフォーマンスに制限あり）
- **推奨**: 32GB（標準的な操作に十分）
- **最適**: 64GB（長い動画、LoRAの使用、高解像度処理に理想的）
- 十分なRAMがない場合、システムはSSDスワップスペースを使用しますが、SSDの寿命を縮める可能性があります

### VRAM要件
- **最小**: 8GB VRAM（FramePack-eichiの推奨最小値）
- **低VRAMモード**: 自動的に有効化され、効率的にメモリを管理
  - `gpu_memory_preservation`設定で調整可能（デフォルト: 10GB）
  - 値を下げる = 処理に使用するVRAMが増える = 高速だがリスクも高い
  - 値を上げる = 処理に使用するVRAMが減る = 低速だが安定性が高い
- **高VRAMモード**: 100GB以上の空きVRAMが検出されると自動的に有効化
  - モデルがGPUメモリに常駐（約20%高速）
  - 定期的なモデルのロード/アンロードが不要

### CPU要件
- 明示的な最小CPUモデルは指定されていません
- **推奨**: 8コア以上の最新のマルチコアCPU
- CPUパフォーマンスはロード時間や前処理/後処理に影響します
- 実際の生成処理のほとんどはGPUで実行されます

### ストレージ要件
- **アプリケーションコード**: 通常1-2GB
- **モデル**: 約30GB（初回起動時に自動的にダウンロード）
- **出力と一時ファイル**: 動画の長さ、解像度、圧縮設定によって異なる
- **推奨総容量**: 150GB以上
- 頻繁な読み書き操作のためSSDを推奨

### サポートされるGPUモデル
- **公式サポート**: NVIDIA RTX 30XX、40XX、50XXシリーズ（fp16およびbf16データフォーマットをサポート）
- **推奨最小**: RTX 3060（または同等の8GB以上のVRAM）
- **動作確認済み**: RTX 3060、3070Ti、4060Ti、4090
- **非公式/未テスト**: GTX 10XX/20XXシリーズ
- **AMD GPU**: 明示的なサポートの言及なし
- **Intel GPU**: 明示的なサポートの言及なし

## Windowsセットアップ手順

### 前提条件
- Windows 10/11
- CUDA 12.6をサポートするドライバーを備えたNVIDIA GPU
- Python 3.10.x
- 7-Zip（インストールパッケージの展開用）

### 手順
1. **基本FramePackのインストール**:
   - [公式FramePackリポジトリ](https://github.com/lllyasviel/FramePack)にアクセス
   - 「Download One-Click Package (CUDA 12.6 + PyTorch 2.6)」をクリック
   - 7zパッケージをダウンロードし、任意の場所に展開
   - `update.bat`を実行（最新のバグ修正を取得するために重要）
   - `run.bat`を実行して初めてFramePackを起動
   - 必要なモデル（約30GB）は初回実行時に自動的にダウンロードされます

2. **FramePack-eichiのインストール**:
   - [FramePack-eichiリポジトリ](https://github.com/git-ai-code/FramePack-eichi)をクローンまたはダウンロード
   - 適切な言語のバッチファイル（日本語は`run_endframe_ichi.bat`、英語は`run_endframe_ichi_en.bat`、繁体字中国語は`run_endframe_ichi_zh-tw.bat`、ロシア語は`run_endframe_ichi_ru.bat`）をFramePackのルートディレクトリにコピー
   - F1版を使用する場合は`run_endframe_ichi_f1.bat`（または言語別の`run_endframe_ichi_en_f1.bat`、`run_endframe_ichi_zh-tw_f1.bat`、`run_endframe_ichi_f1_ru.bat`）
   - 単一フレーム推論の場合は`run_oneframe_ichi.bat`（または言語別の`run_oneframe_ichi_en.bat`、`run_oneframe_ichi_zh-tw.bat`、`run_oneframe_ichi_ru.bat`）
   - FramePack-eichiから以下のファイル/フォルダをFramePackの`webui`フォルダにコピー:
     - `endframe_ichi.py`
     - `eichi_utils`フォルダ（`lora_preset_manager.py`、`model_downloader.py`、`vae_settings.py`などを含む - v1.9.3で追加）
     - `lora_utils`フォルダ
     - `diffusers_helper`フォルダ
     - `locales`フォルダ（`ru.json`ロシア語翻訳ファイルを含む - v1.9.3で追加）

3. **高速化ライブラリのインストール（オプションだが推奨）**:
   - [FramePack Issue #138](https://github.com/lllyasviel/FramePack/issues/138)から高速化パッケージインストーラーをダウンロード
   - `package_installer.zip`ファイルをFramePackのルートディレクトリに展開
   - `package_installer.bat`を実行し、画面の指示に従う（通常はEnterキーを押すだけ）
   - FramePackを再起動し、コンソールに以下のメッセージが表示されることを確認:
     ```
     Xformers is installed!
     Flash Attn is not installed! (This is normal)
     Sage Attn is installed!
     ```

4. **FramePack-eichiの起動**:
   - FramePackのルートディレクトリから`run_endframe_ichi.bat`（または適切な言語バージョン）を実行
   - WebUIがデフォルトブラウザで開きます

5. **動作確認**:
   - WebUIに画像をアップロード
   - 希望する動きを説明するプロンプトを入力
   - 「生成開始」をクリックして動画生成が機能していることを確認

## Linuxセットアップ手順

### サポートされるLinuxディストリビューション
- Ubuntu 22.04 LTS以降（主要サポート）
- Python 3.10をサポートする他のディストリビューションも動作するはず

### 必要なパッケージと依存関係
- CUDA 12.6をサポートするNVIDIA GPUドライバー
- Python 3.10.x
- CUDA Toolkit 12.6
- CUDAサポート付きPyTorch 2.6

### インストール手順

1. **Python環境のセットアップ**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **CUDAサポート付きPyTorchのインストール**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

3. **FramePackのクローンとセットアップ**:
   ```bash
   git clone https://github.com/lllyasviel/FramePack.git
   cd FramePack
   pip install -r requirements.txt
   ```

4. **FramePack-eichiのクローンとセットアップ**:
   ```bash
   git clone https://github.com/git-ai-code/FramePack-eichi.git
   # 必要なファイルをコピー
   cp FramePack-eichi/webui/endframe_ichi.py FramePack/
   cp FramePack-eichi/webui/endframe_ichi_f1.py FramePack/
   cp -r FramePack-eichi/webui/eichi_utils FramePack/
   cp -r FramePack-eichi/webui/lora_utils FramePack/
   cp -r FramePack-eichi/webui/diffusers_helper FramePack/
   cp -r FramePack-eichi/webui/locales FramePack/
   ```

5. **高速化ライブラリのインストール（オプション）**:
   ```bash
   # sage-attention（推奨）
   pip install sageattention==1.0.6
   
   # xformers（サポートされている場合）
   pip install xformers
   ```

6. **FramePack-eichiの起動**:
   ```bash
   cd FramePack
   python endframe_ichi.py  # デフォルトは日本語UI
   python endframe_ichi_f1.py  # デフォルトは日本語UI
   # 英語UIの場合:
   python endframe_ichi.py --lang en
   python endframe_ichi_f1.py --lang en
   # 繁体字中国語UIの場合:
   python endframe_ichi.py --lang zh-tw
   python endframe_ichi_f1.py --lang zh-tw
   ```

## Dockerセットアップ手順

### 前提条件
- システムにDockerがインストールされている
- Docker Composeがインストールされている
- GPU使用のためのNVIDIA Container Toolkitがインストールされている

### Dockerセットアッププロセス

1. **akitaonrailsのDocker実装を使用**:
   ```bash
   git clone https://github.com/akitaonrails/FramePack-Docker-CUDA.git
   cd FramePack-Docker-CUDA
   mkdir outputs
   mkdir hf_download
   
   # Dockerイメージのビルド
   docker build -t framepack-torch26-cu124:latest .
   
   # GPUサポート付きでコンテナを実行
   docker run -it --rm --gpus all -p 7860:7860 \
   -v ./outputs:/app/outputs \
   -v ./hf_download:/app/hf_download \
   framepack-torch26-cu124:latest
   ```

2. **代替Docker Compose設定**:
   - 以下の内容で`docker-compose.yml`ファイルを作成:
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
       # 言語選択（デフォルトは英語）
       command: ["--lang", "en"]  # オプション: "ja"（日本語）、"zh-tw"（繁体字中国語）、"en"（英語）
   ```
   
   - 同じディレクトリに`Dockerfile`を作成:
   ```dockerfile
   FROM python:3.10-slim
   
   ENV DEBIAN_FRONTEND=noninteractive
   
   # システム依存関係のインストール
   RUN apt-get update && apt-get install -y \
       git \
       wget \
       ffmpeg \
       && rm -rf /var/lib/apt/lists/*
   
   # 作業ディレクトリの設定
   WORKDIR /app
   
   # リポジトリのクローン
   RUN git clone https://github.com/lllyasviel/FramePack.git . && \
       git clone https://github.com/git-ai-code/FramePack-eichi.git /tmp/FramePack-eichi
   
   # FramePack-eichiファイルのコピー（Linuxセットアップ手順と同様にルートディレクトリに配置）
   RUN cp /tmp/FramePack-eichi/webui/endframe_ichi.py . && \
       cp /tmp/FramePack-eichi/webui/endframe_ichi_f1.py . && \
       cp -r /tmp/FramePack-eichi/webui/eichi_utils . && \
       cp -r /tmp/FramePack-eichi/webui/lora_utils . && \
       cp -r /tmp/FramePack-eichi/webui/diffusers_helper . && \
       cp -r /tmp/FramePack-eichi/webui/locales . && \
       rm -rf /tmp/FramePack-eichi
   
   # PyTorchと依存関係のインストール
   RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   RUN pip install -r requirements.txt
   RUN pip install sageattention==1.0.6
   
   # 出力ディレクトリの作成
   RUN mkdir -p outputs hf_download
   
   # HuggingFaceキャッシュディレクトリの設定
   ENV HF_HOME=/app/hf_download
   
   # WebUI用のポートを公開
   EXPOSE 7860
   
   # FramePack-eichiの起動（Linuxセットアップ手順と同様にルートディレクトリから実行）
   ENTRYPOINT ["python", "endframe_ichi.py", "--listen"]
   ```
   
   - Docker Composeでビルドして実行:
   ```bash
   docker-compose build
   docker-compose up
   ```

3. **WebUIへのアクセス**:
   - コンテナが実行されると、WebUIはhttp://localhost:7860で利用可能になります

4. **GPUパススルー設定**:
   - NVIDIA Container Toolkitが適切にインストールされていることを確認
   - GPU渡しには`--gpus all`パラメータ（またはdocker-compose.ymlの同等の設定）が必要
   - コンテナ内でGPUにアクセスできるか以下のコマンドで確認:
     ```bash
     docker exec -it [container_id] nvidia-smi
     ```

## macOS（Apple Silicon）セットアップ手順

FramePack-eichiは、CUDAの代わりにMetal Performance Shadersを使用するbrandon929/FramePackフォークを通じてApple SiliconのMacで使用できます。

### 前提条件
- Apple Silicon（M1、M2、またはM3チップ）搭載のmacOS
- Homebrew（macOSパッケージマネージャー）
- Python 3.10
- **メモリ要件**: 最小16GB RAM、推奨32GB+
  - 8GBモデルは深刻なパフォーマンス低下と処理エラーを経験する可能性が高い
  - 16GBモデルは短い動画（3-5秒）と低解像度設定に制限される
  - 32GB+モデルで快適な処理が可能（M2/M3 Ultra推奨）

### インストール手順

1. **Homebrewのインストール**（まだインストールされていない場合）:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   - Homebrewをパスに追加するための追加指示に従ってください。

2. **Python 3.10のインストール**:
   ```bash
   brew install python@3.10
   ```

3. **macOS互換フォークのクローン**:
   ```bash
   git clone https://github.com/brandon929/FramePack.git
   cd FramePack
   ```

4. **Metal対応PyTorchのインストール**（CPUバージョン、PyTorch MPSを介したMetalサポートが追加される）:
   ```bash
   pip3.10 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
   ```

5. **依存関係のインストール**:
   ```bash
   pip3.10 install -r requirements.txt
   ```

6. **Webインターフェースの起動**:
   ```bash
   python3.10 demo_gradio.py --fp32
   ```
   
   `--fp32`フラグはApple Siliconの互換性のために重要です。M1/M2/M3プロセッサは、元のモデルで使用されているfloat16およびbfloat16を完全にサポートしていない場合があります。

7. **起動後**、Webブラウザを開き、ターミナルに表示されるURL（通常はhttp://127.0.0.1:7860）にアクセスします。

### Apple Siliconの特別な考慮事項

- **Metalパフォーマンス**: 
  - Apple Siliconとの互換性のために`--fp32`フラグを使用
- **解像度設定**: 
  - 16GB RAM: 最大416×416解像度を推奨
  - 32GB RAM: 最大512×512解像度を推奨
  - 64GB RAM: 最大640×640解像度を試すことが可能
- **パフォーマンス比較**:
  - 生成速度はNVIDIA GPUと比較して大幅に遅い
  - 5秒の動画生成時間比較:
    - RTX 4090: 約6分
    - M2 Max: 約25-30分
    - M3 Max: 約20-25分
    - M2 Ultra: 約15-20分
    - M3 Ultra: 約12-15分
- **メモリ管理**: 
  - Apple Siliconの統合メモリアーキテクチャはGPU/CPUが同じメモリプールを共有することを意味する
  - アクティビティモニタで「メモリプレッシャー」を監視し、圧縮が高い場合は設定を下げる
  - スワップ使用量の増加はパフォーマンスを大幅に低下させ、SSDの寿命に影響する
  - 生成中は他のリソース集約型アプリを閉じることを強く推奨
  - メモリリークを解決するために長時間使用後はアプリケーションを再起動する

## WSLセットアップ手順

WSLでFramePack-eichiをセットアップすると、NVIDIAのWSLドライバーを通じてGPUアクセラレーションを備えたWindowsでLinux環境を提供します。

### 前提条件
- Windows 10（バージョン2004以降）またはWindows 11
- NVIDIA GPU（RTX 30XX、40XX、または50XXシリーズ推奨、最小8GB VRAM）
- 管理者アクセス
- WSL2をサポートする更新されたNVIDIAドライバー

### インストール手順

1. **WSL2のインストール**:
   
   管理者としてPowerShellを開き、以下を実行:
   ```powershell
   wsl --install
   ```
   
   このコマンドはデフォルトのLinuxディストリビューションとしてUbuntuを使用してWSL2をインストールします。指示があればコンピュータを再起動してください。

2. **WSL2が適切にインストールされていることを確認**:
   ```powershell
   wsl --status
   ```
   
   デフォルトバージョンとして「WSL 2」が表示されていることを確認してください。

3. **WSLカーネルの更新**（必要な場合）:
   ```powershell
   wsl --update
   ```

4. **WSL用NVIDIAドライバーのインストール**:
   
   NVIDIAのWebサイトからWSLをサポートする最新のNVIDIAドライバーをダウンロードしてインストールします。WSL環境内にNVIDIAドライバーをインストールしないでください - WSLはWindowsのドライバーを使用します。

5. **Ubuntuを起動してGPUアクセスを確認**:
   
   スタートメニューからUbuntuを起動するか、PowerShellで`wsl`を実行し、NVIDIA GPU検出を確認:
   ```bash
   nvidia-smi
   ```
   
   GPUの情報が表示されるはずです。

6. **WSLで環境をセットアップ**:
   ```bash
   # パッケージリストの更新
   sudo apt update && sudo apt upgrade -y
   
   # Pythonと開発ツールのインストール
   sudo apt install -y python3.10 python3.10-venv python3-pip git
   
   # FramePack-eichiリポジトリのクローン
   git clone https://github.com/git-ai-code/FramePack-eichi.git
   cd FramePack-eichi
   
   # 仮想環境の作成と有効化
   python3.10 -m venv venv
   source venv/bin/activate
   
   # CUDAサポート付きPyTorchのインストール
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   
   # 依存関係のインストール
   pip install -r requirements.txt
   ```

7. **FramePack-eichiの起動**:
   ```bash
   python endframe_ichi.py
   ```

   言語を指定することもできます:
   ```bash
   python endframe_ichi.py --lang en  # 英語の場合
   ```

8. **Webインターフェースにアクセス**するには、Windowsでブラウザを開き、ターミナルに表示されるURL（通常はhttp://127.0.0.1:7860）に移動します。

## Anaconda環境セットアップ手順

### 新しいConda環境の作成

```bash
# Python 3.10を使用して新しいconda環境を作成
conda create -n framepack-eichi python=3.10
conda activate framepack-eichi

# CUDAサポート付きPyTorchのインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### ソースからの手動インストール

```bash
# オリジナルのFramePackリポジトリをクローン
git clone https://github.com/lllyasviel/FramePack.git
cd FramePack

# FramePack-eichiリポジトリを一時的な場所にクローン
git clone https://github.com/git-ai-code/FramePack-eichi.git temp_eichi

# 拡張webファイルをコピー（Linuxセットアップ手順と同様にルートディレクトリに配置）
cp temp_eichi/webui/endframe_ichi.py .
cp temp_eichi/webui/endframe_ichi_f1.py .
cp -r temp_eichi/webui/eichi_utils .
cp -r temp_eichi/webui/lora_utils .
cp -r temp_eichi/webui/diffusers_helper .
cp -r temp_eichi/webui/locales .

# 言語固有のバッチファイルをルートディレクトリにコピー（適切なファイルを選択）
cp temp_eichi/run_endframe_ichi.bat .  # 日本語（デフォルト）
# cp temp_eichi/run_endframe_ichi_en.bat .  # 英語
# cp temp_eichi/run_endframe_ichi_zh-tw.bat .  # 繁体字中国語

# 依存関係のインストール
pip install -r requirements.txt

# 一時ディレクトリの削除
rm -rf temp_eichi
```

### Condaの特別な考慮事項

- condaを介してインストールする場合、PyTorchパッケージとの依存関係の競合が発生する可能性があります
- 最良の結果を得るには、condaチャネルではなく、公式インデックスURLを使用してpip経由でPyTorch、torchvision、およびtorchaudioをインストールしてください
- xformers、flash-attn、sageattentionなどのオプションの高速化パッケージは、メイン環境が作成された後に個別にインストールする必要があります

## Google Colabセットアップ手順

### 2025年5月最新Colabセットアップ（最も安定）

以下のスクリプトは、Colabの最新環境（2025年5月現在）向けの最も安定したセットアップを提供します。これはA100 GPU環境で特にテストされています。

```python
# gitがまだインストールされていない場合はインストール
!apt-get update && apt-get install -y git

# FramePackリポジトリをクローン
!git clone https://github.com/lllyasviel/FramePack.git
%cd FramePack

# PyTorch（CUDA対応バージョン）のインストール
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Colab環境用にRequestsとNumPyをアップグレード
!pip install requests==2.32.3 numpy==2.0.0

# FramePack依存関係のインストール
!pip install -r requirements.txt

# 速度最適化用のSageAttentionをインストール（オプション）
!pip install sageattention==1.0.6

# FramePackデモを開始（実行するにはコメントを解除）
# !python demo_gradio.py --share

# FramePack-eichiをインストール
!git clone https://github.com/git-ai-code/FramePack-eichi.git tmp
!rsync -av --exclude='diffusers_helper' tmp/webui/ ./
!cp tmp/webui/diffusers_helper/bucket_tools.py diffusers_helper/
!cp tmp/webui/diffusers_helper/memory.py diffusers_helper/
!rm -rf tmp

# FramePack-eichiを実行
!python endframe_ichi.py --share
```

> **重要**: 上記の方法では`diffusers_helper/bucket_tools.py`ファイルを個別にコピーしています。これは一般的な「ImportError: cannot import name 'SAFE_RESOLUTIONS' from 'diffusers_helper.bucket_tools'」エラーを回避するために必要です。

### 代替Colabセットアップ方法

以下は代替のセットアップ方法です。より新しい環境では上記の方法を優先してください。

```python
# FramePack-eichiリポジトリをクローン
!git clone https://github.com/git-ai-code/FramePack-eichi.git tmp

# 基本FramePackをクローン
!git clone https://github.com/lllyasviel/FramePack.git
%cd /content/FramePack

# 依存関係のインストール
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
!pip install -r requirements.txt

# eichi拡張のセットアップ
!mkdir -p webui 
!cp -r /content/tmp/webui/* webui/
!cp /content/tmp/run_endframe_ichi.bat .

# PYTHONPATH環境変数の設定
%env PYTHONPATH=/content/FramePack:$PYTHONPATH

# 公開URLでWebUIを起動
%cd /content/FramePack
!python endframe_ichi.py --share
```

### Google Driveの統合と出力設定

生成された動画をGoogle Driveに保存するには:

```python
# Google Driveをマウント
from google.colab import drive
drive.mount('/content/drive')

# 出力ディレクトリを設定
import os
OUTPUT_DIR = "/content/drive/MyDrive/FramePack-eichi-outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 出力ディレクトリを指定してframeppackを起動
!python endframe_ichi.py --share --output_dir={OUTPUT_DIR}
```

### Colabの一般的なトラブルシューティング

1. **'SAFE_RESOLUTIONS'インポートエラー**:
   ```
   ImportError: cannot import name 'SAFE_RESOLUTIONS' from 'diffusers_helper.bucket_tools'
   ```
   - **解決策**: diffusers_helperファイルの個別コピーを含む2025年5月の最新セットアップスクリプトを使用

2. **メモリ不足エラー**:
   ```
   RuntimeError: CUDA out of memory
   ```
   - **解決策**: 
     - 解像度を下げる（例: 416×416）
     - キーフレーム数を減らす
     - バッチサイズを減らす
     - GPU推論保存メモリ設定を調整する

3. **セッション切断**:
   - **解決策**:
     - 長い処理時間を避ける
     - 進行状況をGoogle Driveに保存する
     - ブラウザタブをアクティブに保つ

### 異なるColabティアのVRAM/RAM考慮事項

| Colabティア | GPUタイプ | VRAM | パフォーマンス | 備考 |
|------------|----------|------|-------------|-------|
| 無料       | T4       | 16GB | 制限あり     | 短い動画（1-5秒）の基本的な使用に十分 |
| Pro        | A100     | 40GB | 良好        | より長い動画と複数のキーフレームを処理可能 |
| Pro+       | A100     | 80GB | 優れている   | 最高のパフォーマンス、複雑な生成が可能 |

### Colabの最適設定

1. **ハードウェアアクセラレータ設定**:
   - メニュー「ランタイム」→「ランタイムのタイプを変更」→「ハードウェアアクセラレータ」を「GPU」に設定
   - Pro/Pro+ユーザーは、可能であれば「高RAM」または「高メモリ」オプションを選択すべき

2. **推奨バッチサイズと解像度設定**:
   - **T4 GPU（無料）**: バッチサイズ4、解像度416x416
   - **A100 GPU（Pro）**: バッチサイズ8、解像度最大640x640
   - **A100 GPU（Pro+/高メモリ）**: バッチサイズ16、解像度最大768x768

## クラウド環境（AWS/GCP/Azure）セットアップ手順

### AWS EC2セットアップ

#### 推奨インスタンスタイプ:
- **g4dn.xlarge**: 1 NVIDIA T4 GPU（16GB）、4 vCPU、16GB RAM
- **g4dn.2xlarge**: 1 NVIDIA T4 GPU（16GB）、8 vCPU、32GB RAM
- **g5.xlarge**: 1 NVIDIA A10G GPU（24GB）、4 vCPU、16GB RAM
- **p3.2xlarge**: 1 NVIDIA V100 GPU（16GB）、8 vCPU、61GB RAM

#### セットアップ手順:

1. **EC2インスタンスの起動** - 選択したインスタンスタイプを使用してDeep Learning AMI（Ubuntu）を使用
2. **SSHでインスタンスに接続**:
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```
3. **システムパッケージの更新**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
4. **リポジトリのクローン**:
   ```bash
   git clone https://github.com/lllyasviel/FramePack.git
   cd FramePack
   git clone https://github.com/git-ai-code/FramePack-eichi.git temp_eichi
   # Linuxセットアップ手順と同様にルートディレクトリに配置
   cp temp_eichi/webui/endframe_ichi.py .
   cp temp_eichi/webui/endframe_ichi_f1.py .
   cp -r temp_eichi/webui/eichi_utils .
   cp -r temp_eichi/webui/lora_utils .
   cp -r temp_eichi/webui/diffusers_helper .
   cp -r temp_eichi/webui/locales .
   cp temp_eichi/run_endframe_ichi_en.bat .  # 英語バージョン
   rm -rf temp_eichi
   ```
5. **依存関係のインストール**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```
6. **セキュリティグループの設定** - ポート7860への着信トラフィックを許可
7. **公開表示で実行**:
   ```bash
   python endframe_ichi.py --server --listen --port 7860
   ```
8. **UIへのアクセス** - http://your-instance-ip:7860

### Google Cloud Platform（GCP）セットアップ

#### 推奨インスタンスタイプ:
- **n1-standard-8** + 1 NVIDIA T4 GPU
- **n1-standard-8** + 1 NVIDIA V100 GPU
- **n1-standard-8** + 1 NVIDIA A100 GPU

#### セットアップ手順:

1. **Deep Learning VMイメージでVMインスタンスを作成**
2. **GPUを有効化**し、適切なGPUタイプを選択
3. **SSHでインスタンスに接続**
4. **AWS EC2と同じ手順に従って**FramePack-eichiをセットアップ
5. **ファイアウォールルールの設定** - ポート7860への着信トラフィックを許可

### Microsoft Azureセットアップ

#### 推奨VMサイズ:
- **Standard_NC6s_v3**: 1 NVIDIA V100 GPU（16GB）
- **Standard_NC4as_T4_v3**: 1 NVIDIA T4 GPU（16GB）
- **Standard_NC24ads_A100_v4**: 1 NVIDIA A100 GPU（80GB）

#### セットアップ手順:
1. **Data Science Virtual Machine（Ubuntu）でVMを作成**
2. **SSHでVMに接続**
3. **AWS EC2と同じ手順に従って**FramePack-eichiをセットアップ
4. **ネットワークセキュリティグループの設定** - ポート7860への着信トラフィックを許可

## 一般的なエラーとトラブルシューティング手順

### インストールエラー

#### Python依存関係の競合
- **症状**: 互換性のないパッケージバージョンに関するエラーメッセージ
- **解決策**: 
  - 明示的にPython 3.10を使用（3.11、3.12、またはそれ以上ではない）
  - 正しいCUDAバージョンでPyTorchをインストール
  - 依存関係エラーが発生した場合は新しい仮想環境を作成

#### CUDAインストールと互換性の問題
- **症状**: 「CUDAは利用できません」エラー、CPUでの実行に関する警告
- **解決策**:
  - サポートされているGPU（RTX 30XX、40XX、または50XXシリーズ推奨）を使用していることを確認
  - 正しいCUDAツールキット（12.6推奨）をインストール
  - Pythonでトラブルシューティング:
    ```python
    import torch
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    ```

#### パッケージインストールの失敗
- **症状**: PIPインストールエラー、ホイールビルドの失敗
- **解決策**:
  - Windows向けのワンクリックインストーラーを使用（手動インストールの代わりに）
  - Linux向け: 必要なビルド依存関係をインストール
  - SageAttentionのインストールが失敗しても、それなしで続行できる
  - Issue #138からpackage_installer.zipを使用して高度な最適化パッケージをインストール

### ランタイムエラー

#### CUDAメモリ不足エラー
- **症状**: 「CUDA out of memory」エラーメッセージ、生成の高メモリフェーズでのクラッシュ
- **解決策**:
  - `gpu_memory_preservation`値を増やす（6〜16GBの間の値を試す）
  - 他のGPU集約型アプリケーションを閉じる
  - 再起動して再試行
  - 画像解像度を下げる（低VRAMには512x512を推奨）
  - より大きなWindowsページファイルを設定（物理RAMの3倍）
  - 十分なシステムRAM（32GB+推奨）を確保

#### モデルロードの失敗
- **症状**: モデルシャードのロード時のエラーメッセージ、モデル初期化時のプロセスクラッシュ
- **解決策**:
  - アプリケーションを開始する前に`update.bat`を実行
  - すべてのモデルが`webui/hf_download`フォルダに適切にダウンロードされていることを確認
  - モデルが不足している場合は自動ダウンロードが完了するのを許可（約30GB）
  - モデルを手動で配置する場合は、ファイルを正しい`framepack\webui\hf_download`フォルダにコピー

#### WebUI起動の問題
- **症状**: 起動後にGradioインターフェースが表示されない、ブラウザに「接続できません」エラーが表示される
- **解決策**:
  - `--port XXXX`コマンドラインオプションで別のポートを試す
  - ポート7860（Gradioのデフォルト）を使用している他のアプリケーションがないことを確認
  - `--inbrowser`オプションを使用して自動的にブラウザを開く
  - 特定のエラーメッセージについてコンソール出力を確認

### プラットフォーム固有の問題

#### Windows固有の問題
- **症状**: パス関連のエラー、DLLロードの失敗、バッチファイルが適切に実行されない
- **解決策**:
  - パス長の制限を避けるために、短いパス（例：C:\FramePack）にインストール
  - 権限の問題が発生した場合は、管理者としてバッチファイルを実行
  - DLLロードエラーが表示される場合:
    - Visual C++ Redistributableパッケージをインストール
    - アンチウイルスソフトウェアが実行をブロックしていないか確認

#### Linux固有の問題
- **症状**: ライブラリ不足エラー、パッケージビルドの失敗、GUI表示の問題
- **解決策**:
  - Debian/Ubuntuでは、必要なシステムライブラリをインストール:
    ```
    sudo apt-get install libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libopenblas-dev
    ```
  - GPU検出の問題については、NVIDIAドライバーが正しくインストールされていることを確認:
    ```
    nvidia-smi
    ```

#### macOS固有の問題
- **症状**: Metal/MPS関連のエラー、Apple Siliconでの低パフォーマンス
- **解決策**:
  - `--fp32`フラグで実行（M1/M2はfp16/bf16を完全にサポートしていない可能性がある）
  - 動画フォーマットの問題については、MP4圧縮設定を約16（デフォルト）に調整
  - NVIDIAハードウェアと比較して大幅に性能が低下することを認識

#### WSLセットアップの問題
- **症状**: WSLでGPUが検出されない、WSLで非常に低いパフォーマンス
- **解決策**:
  - WSL2を使用していることを確認（WSL1ではない）: `wsl --set-version <Distro> 2`
  - WSL専用のNVIDIAドライバーをインストール
  - Windowsユーザーディレクトリに`.wslconfig`ファイルを作成:
    ```
    [wsl2]
    memory=16GB  # システムに基づいて調整
    processors=8  # システムに基づいて調整
    gpumemory=8GB  # GPUに基づいて調整
    ```

### パフォーマンスの問題

#### 遅い生成時間と最適化テクニック
- **症状**: 過度に長い生成時間、ベンチマークと比較して予想よりも低いパフォーマンス
- **解決策**:
  - 最適化ライブラリをインストール:
    - Issue #138からpackage_installer.zipをダウンロードしてpackage_installer.batを実行
    - これにより可能な場合はxformers、flash-attn、sage-attnがインストールされる
  - より高速な（ただし潜在的に品質の低い）生成のためにteacacheを有効にする
  - 他のGPU集約型アプリケーションを閉じる
  - より高速な生成のために解像度を下げる（品質を犠牲にする）

#### メモリリークと管理
- **症状**: 時間の経過とともに増加するメモリ使用量、複数の生成にわたるパフォーマンスの低下
- **解決策**:
  - 長い生成セッションの間にアプリケーションを再起動
  - GPUメモリ使用量を監視:
    ```
    nvidia-smi -l 1
    ```
  - CPU/メモリリークが発生した場合はPythonプロセスを再起動
  - 設定を切り替える際に明示的なモデルアンロードを使用
  - 必要でない場合は複数のLoRAを同時にロードしない

## 情報源

1. 公式リポジトリ:
   - FramePack-eichi: https://github.com/git-ai-code/FramePack-eichi
   - オリジナルFramePack: https://github.com/lllyasviel/FramePack

2. コミュニティリソース:
   - FramePack Docker実装: https://github.com/akitaonrails/FramePack-Docker-CUDA
   - Apple Silicon互換フォーク: https://github.com/brandon929/FramePack

3. 公式ドキュメント:
   - FramePack-eichi GitHubリポジトリのREADMEとwiki
   - GitHub Issuesの開発者コメント

4. トラブルシューティングリソース:
   - FramePack Issue #138（高速化ライブラリ）: https://github.com/lllyasviel/FramePack/issues/138
   - WSL CUDA設定ドキュメント: https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl

このガイドはFramePack-eichiの包括的なセットアップ手順と、様々な環境での運用のためのベストプラクティスを提供します。環境に最適なセットアップパスを選択し、必要に応じてトラブルシューティング手順を参照してください。
import os
import sys
sys.path.append(os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './submodules/FramePack'))))

# グローバル変数 - 停止フラグと通知状態管理
user_abort = False
user_abort_notified = False

# バッチ処理とキュー機能用グローバル変数
batch_stopped = False  # バッチ処理中断フラグ

# テキストエンコード結果のキャッシュ用グローバル変数を初期化
cached_prompt = None
cached_n_prompt = None
cached_llama_vec = None
cached_llama_vec_n = None
cached_clip_l_pooler = None
cached_clip_l_pooler_n = None
cached_llama_attention_mask = None
cached_llama_attention_mask_n = None
queue_enabled = False  # キュー機能の有効/無効フラグ
queue_type = "prompt"  # キューのタイプ（"prompt" または "image"）
prompt_queue_file_path = None  # プロンプトキューファイルのパス
image_queue_files = []  # イメージキューのファイルリスト

from diffusers_helper.hf_login import login

import os
import random  # ランダムシード生成用
import time
import traceback  # デバッグログ出力用
import yaml
import argparse
import json
import glob
import subprocess  # フォルダを開くために必要
from PIL import Image

# PNGメタデータ処理モジュールのインポート
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from eichi_utils.png_metadata import (
    embed_metadata_to_png, extract_metadata_from_png, extract_metadata_from_numpy_array,
    PROMPT_KEY, SEED_KEY, SECTION_PROMPT_KEY, SECTION_NUMBER_KEY
)

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--lang", type=str, default='ja', help="Language: ja, zh-tw, en, ru")
args = parser.parse_args()

# Load translations from JSON files
from locales.i18n_extended import (set_lang, translate)
set_lang(args.lang)

# サーバーがすでに実行中かチェック
import socket
def is_port_in_use(port):
    """指定ポートが使用中かどうかを確認"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    except:
        # エラーが発生した場合はポートが使用されていないと判断
        return False

# キャッシュ制御設定
# ローカルファイルの優先的利用を無効化し安定性を向上
use_cache_files = True  # ファイルキャッシュは使用
first_run = True  # 通常は初回起動として扱う

# ポートが使用中の場合は警告を表示
if is_port_in_use(args.port):
    print(translate("警告: ポート {0} はすでに使用されています。他のインスタンスが実行中かもしれません。").format(args.port))
    print(translate("5秒後に処理を続行します..."))
    first_run = False  # 初回実行ではない
    time.sleep(5)  # 5秒待機して続行

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
    print(translate("HF_HOMEを設定: {0}").format(os.environ['HF_HOME']))
else:
    print(translate("既存のHF_HOMEを使用: {0}").format(os.environ['HF_HOME']))

# LoRAサポートの確認
has_lora_support = False
has_fp8_support = False
try:
    import lora_utils
    from lora_utils.fp8_optimization_utils import check_fp8_support, apply_fp8_monkey_patch
    
    has_lora_support = True
    # FP8サポート確認
    has_e4m3, has_e5m2, has_scaled_mm = check_fp8_support()
    has_fp8_support = has_e4m3 and has_e5m2
    
    if has_fp8_support:
        print(translate("LoRAサポートとFP8最適化が有効です"))
        print(translate("FP8最適化は実際のモデルロード時に適用されます"))
        # FP8最適化のモンキーパッチはLoRA適用時に使用される
        # apply_fp8_monkey_patch()は引数が必要なため、ここでは呼び出さない
    else:
        print(translate("LoRAサポートが有効です (FP8最適化はサポートされていません)"))
except ImportError:
    print(translate("LoRAサポートが無効です（lora_utilsモジュールがインストールされていません）"))

# 設定管理モジュールをインポート
from eichi_utils.settings_manager import (
    get_settings_file_path,
    get_output_folder_path,
    initialize_settings,
    load_settings,
    save_settings,
    open_output_folder
)

import gradio as gr
from eichi_utils.ui_styles import get_app_css
import torch
import einops
import safetensors.torch as sf
import numpy as np
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

# フォルダを開く関数
def open_folder(folder_path):
    """指定されたフォルダをOSに依存せず開く"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(translate("フォルダを作成しました: {0}").format(folder_path))

    try:
        if os.name == 'nt':  # Windows
            subprocess.Popen(['explorer', folder_path])
        elif os.name == 'posix':  # Linux/Mac
            try:
                subprocess.Popen(['xdg-open', folder_path])
            except:
                subprocess.Popen(['open', folder_path])
        print(translate("フォルダを開きました: {0}").format(folder_path))
        return True
    except Exception as e:
        print(translate("フォルダを開く際にエラーが発生しました: {0}").format(e))
        return False
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, gpu_complete_modules, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from eichi_utils.ui_styles import get_app_css
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

from eichi_utils.transformer_manager import TransformerManager
from eichi_utils.text_encoder_manager import TextEncoderManager

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 100

print(translate('Free VRAM {0} GB').format(free_mem_gb))
print(translate('High-VRAM Mode: {0}').format(high_vram))

# モデルを並列ダウンロードしておく
from eichi_utils.model_downloader import ModelDownloader
ModelDownloader().download_original()

# グローバルなモデル状態管理インスタンスを作成（モデルは実際に使用するまでロードしない）
transformer_manager = TransformerManager(device=gpu, high_vram_mode=high_vram, use_f1_model=False)
text_encoder_manager = TextEncoderManager(device=gpu, high_vram_mode=high_vram)

# LoRAの状態を確認
def reload_transformer_if_needed():
    """transformerモデルが必要に応じてリロードする"""
    try:
        # 既存のensure_transformer_stateメソッドを使用する
        if hasattr(transformer_manager, 'ensure_transformer_state'):
            return transformer_manager.ensure_transformer_state()
        # 互換性のために古い方法も維持
        elif hasattr(transformer_manager, '_needs_reload') and hasattr(transformer_manager, '_reload_transformer'):
            if transformer_manager._needs_reload():
                return transformer_manager._reload_transformer()
            return True
        return False
    except Exception as e:
        print(f"transformerリロードエラー: {e}")
        traceback.print_exc()
        return False

# 遅延ロード方式に変更 - 起動時にはtokenizerのみロードする
try:
    # tokenizerのロードは起動時から行う
    try:
        print(translate("tokenizer, tokenizer_2のロードを開始します..."))
        tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
        tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
        print(translate("tokenizer, tokenizer_2のロードが完了しました"))
    except Exception as e:
        print(translate("tokenizer, tokenizer_2のロードに失敗しました: {0}").format(e))
        traceback.print_exc()
        print(translate("5秒間待機後に再試行します..."))
        time.sleep(5)
        tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
        tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    
    # feature_extractorは軽量なのでここでロード
    try:
        print(translate("feature_extractorのロードを開始します..."))
        feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
        print(translate("feature_extractorのロードが完了しました"))
    except Exception as e:
        print(translate("feature_extractorのロードに失敗しました: {0}").format(e))
        print(translate("再試行します..."))
        time.sleep(2)
        feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    
    # 他の重いモデルは遅延ロード方式に変更
    # 変数の初期化だけ行い、実際のロードはworker関数内で行う
    vae = None
    text_encoder = None
    text_encoder_2 = None
    transformer = None
    image_encoder = None
    
    # transformerダウンロードの確保だけは起動時に
    try:
        # モデルのダウンロードを確保（実際のロードはまだ行わない）
        print(translate("モデルのダウンロードを確保します..."))
        transformer_manager.ensure_download_models()
        print(translate("モデルのダウンロードを確保しました"))
    except Exception as e:
        print(translate("モデルダウンロード確保エラー: {0}").format(e))
        traceback.print_exc()
    
except Exception as e:
    print(translate("初期化エラー: {0}").format(e))
    print(translate("プログラムを終了します..."))
    import sys
    sys.exit(1)

# モデル設定のデフォルト値を定義（実際のモデルロードはworker関数内で行う）
def setup_vae_if_loaded():
    global vae
    if vae is not None:
        vae.eval()
        if not high_vram:
            vae.enable_slicing()
            vae.enable_tiling()
        vae.to(dtype=torch.float16)
        vae.requires_grad_(False)
        if high_vram:
            vae.to(gpu)
            
def setup_image_encoder_if_loaded():
    global image_encoder
    if image_encoder is not None:
        image_encoder.eval()
        image_encoder.to(dtype=torch.float16)
        image_encoder.requires_grad_(False)
        if high_vram:
            image_encoder.to(gpu)

stream = AsyncStream()

# フォルダ構造を先に定義
webui_folder = os.path.dirname(os.path.abspath(__file__))
base_path = webui_folder  # endframe_ichiとの互換性のため

# 設定保存用フォルダの設定
settings_folder = os.path.join(webui_folder, 'settings')
os.makedirs(settings_folder, exist_ok=True)

# 設定から出力フォルダを取得
app_settings = load_settings()
output_folder_name = app_settings.get('output_folder', 'outputs')
print(translate("設定から出力フォルダを読み込み: {0}").format(output_folder_name))

# 出力フォルダのフルパスを生成
outputs_folder = get_output_folder_path(output_folder_name)
os.makedirs(outputs_folder, exist_ok=True)

# グローバル変数
g_frame_size_setting = "1フレーム"
batch_stopped = False  # バッチ処理中断フラグ
queue_enabled = False  # キュー機能の有効/無効フラグ
queue_type = "prompt"  # キューのタイプ（"prompt" または "image"）
prompt_queue_file_path = None  # プロンプトキューのファイルパス
image_queue_files = []  # イメージキューのファイルリスト
input_folder_name_value = "inputs"  # 入力フォルダの名前（デフォルト値）

# イメージキューのための画像ファイルリストを取得する関数（グローバル関数）
def get_image_queue_files():
    """入力フォルダから画像ファイルを取得してイメージキューに追加する関数"""
    global image_queue_files

    # 入力フォルダの設定
    input_folder = os.path.join(base_path, input_folder_name_value)

    # フォルダが存在しない場合は作成
    os.makedirs(input_folder, exist_ok=True)

    # すべての画像ファイルを取得
    image_exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    image_files = []

    # 同じファイルが複数回追加されないようにセットを使用
    file_set = set()

    for ext in image_exts:
        pattern = os.path.join(input_folder, '*' + ext)
        for file in glob.glob(pattern):
            if file not in file_set:
                file_set.add(file)
                image_files.append(file)

        pattern = os.path.join(input_folder, '*' + ext.upper())
        for file in glob.glob(pattern):
            if file not in file_set:
                file_set.add(file)
                image_files.append(file)

    # ファイルを修正日時の昇順でソート
    image_files.sort(key=lambda x: os.path.getmtime(x))

    print(translate("入力ディレクトリから画像ファイル{0}個を読み込みました").format(len(image_files)))

    # デバッグ - 読み込んだファイルのリストを確認
    if len(image_files) > 0:
        print(translate("[DEBUG] イメージキュー: 最初の画像ファイル = {0}").format(image_files[0]))

    image_queue_files = image_files
    return image_files

# ワーカー関数
@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, steps, cfg, gs, rs,
           gpu_memory_preservation, use_teacache, lora_files=None, lora_files2=None, lora_scales_text="0.8,0.8,0.8",
           output_dir=None, use_lora=False, fp8_optimization=False, resolution=640,
           latent_window_size=9, latent_index=0, use_clean_latents_2x=True, use_clean_latents_4x=True, use_clean_latents_post=True,
           lora_mode=None, lora_dropdown1=None, lora_dropdown2=None, lora_dropdown3=None, lora_files3=None,
           batch_index=None, use_queue=False, prompt_queue_file=None):
    
    # モデル変数をグローバルとして宣言（遅延ロード用）
    global vae, text_encoder, text_encoder_2, transformer, image_encoder
    global queue_enabled, queue_type, prompt_queue_file_path, image_queue_files
    # テキストエンコード結果をキャッシュするグローバル変数
    global cached_prompt, cached_n_prompt, cached_llama_vec, cached_llama_vec_n, cached_clip_l_pooler, cached_clip_l_pooler_n
    global cached_llama_attention_mask, cached_llama_attention_mask_n

    # キュー状態のログ出力
    use_queue_flag = bool(use_queue)
    queue_type_flag = queue_type
    if use_queue_flag:
        print(translate("[DEBUG] worker: キュー状態: {0}, タイプ: {1}").format(use_queue_flag, queue_type_flag))

        if queue_type_flag == "prompt" and prompt_queue_file_path is not None:
            print(translate("[DEBUG] worker: プロンプトキューファイルパス: {0}").format(prompt_queue_file_path))

        elif queue_type_flag == "image" and len(image_queue_files) > 0:
            print(translate("[DEBUG] worker: イメージキュー詳細: 画像数={0}, batch_index={1}").format(
                len(image_queue_files), batch_index))

    job_id = generate_timestamp()
    
    # 1フレームモード固有の設定
    total_latent_sections = 1  # 1セクションに固定
    frame_count = 1  # 1フレームモード
    
    # 詳細設定のログ出力
    print(translate("\n[詳細設定]"))
    print(translate("RoPE値 (latent_window_size): {0}").format(latent_window_size))
    print(translate("レイテントインデックス: {0}").format(latent_index))
    print(translate("clean_latents_2xを使用: {0}").format(use_clean_latents_2x))
    print(translate("clean_latents_4xを使用: {0}").format(use_clean_latents_4x))
    print(translate("clean_latents_postを使用: {0}").format(use_clean_latents_post))
    
    # 出力フォルダの設定
    if output_dir:
        outputs_folder = output_dir
    else:
        # 出力フォルダはwebui内のoutputsに固定
        outputs_folder = os.path.join(webui_folder, 'outputs')
    
    os.makedirs(outputs_folder, exist_ok=True)
    
    # プログレスバーの初期化
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
    
    # モデルや中間ファイルなどのキャッシュ利用フラグ
    use_cached_files = use_cache_files
    
    try:
        # LoRA 設定 - ディレクトリ選択モードをサポート
        current_lora_paths = []
        current_lora_scales = []
        
        if use_lora and has_lora_support:
            print(translate("\u25c6 LoRA情報: use_lora = {0}, has_lora_support = {1}").format(use_lora, has_lora_support))
            print(translate("\u25c6 LoRAモード: {0}").format(lora_mode))
            
            if lora_mode == translate("ディレクトリから選択"):
                # ディレクトリから選択モードの場合
                lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
                print(translate("\u25c6 LoRAディレクトリ: {0}").format(lora_dir))
                
                # ドロップダウンの選択項目を処理
                dropdown_paths = []
                
                # 各ドロップダウンからLoRAを追加
                for dropdown_idx, dropdown_value in enumerate([lora_dropdown1, lora_dropdown2, lora_dropdown3]):
                    dropdown_name = f"LoRA{dropdown_idx+1}"
                    if dropdown_value and dropdown_value != translate("なし"):
                        lora_path = os.path.join(lora_dir, dropdown_value)
                        print(translate("\u25c6 {name}のロード試行: パス={path}").format(name=dropdown_name, path=lora_path))
                        if os.path.exists(lora_path):
                            current_lora_paths.append(lora_path)
                            print(translate("\u25c6 {name}を選択: {path}").format(name=dropdown_name, path=lora_path))
                        else:
                            # パスを修正して再試行（単なるファイル名の場合）
                            if os.path.dirname(lora_path) == lora_dir and not os.path.isabs(dropdown_value):
                                # すでに正しく構築されているので再試行不要
                                pass
                            else:
                                # 直接ファイル名だけで試行
                                lora_path_retry = os.path.join(lora_dir, os.path.basename(str(dropdown_value)))
                                print(translate("\u25c6 {name}を再試行: {path}").format(name=dropdown_name, path=lora_path_retry))
                                if os.path.exists(lora_path_retry):
                                    current_lora_paths.append(lora_path_retry)
                                    print(translate("\u25c6 {name}を選択 (パス修正後): {path}").format(name=dropdown_name, path=lora_path_retry))
                                else:
                                    print(translate("\u25c6 選択された{name}が見つかりません: {file}").format(name=dropdown_name, file=dropdown_value))
            else:
                # ファイルアップロードモードの場合
                # 全LoRAファイルを収集
                all_lora_files = []
                
                # 各LoRAファイルを処理
                for file_idx, lora_file_obj in enumerate([lora_files, lora_files2, lora_files3]):
                    if lora_file_obj is None:
                        continue
                        
                    file_name = f"LoRAファイル{file_idx+1}"
                    print(translate("\u25c6 {name}の処理").format(name=file_name))
                    
                    if isinstance(lora_file_obj, list):
                        # 複数のファイルが含まれている場合
                        for file in lora_file_obj:
                            if hasattr(file, 'name') and file.name:
                                current_lora_paths.append(file.name)
                                print(translate("\u25c6 {name}: {file}").format(name=file_name, file=os.path.basename(file.name)))
                    else:
                        # 単一のファイル
                        if hasattr(lora_file_obj, 'name') and lora_file_obj.name:
                            current_lora_paths.append(lora_file_obj.name)
                            print(translate("\u25c6 {name}: {file}").format(name=file_name, file=os.path.basename(lora_file_obj.name)))
            
            # スケール値を処理
            if current_lora_paths:  # LoRAパスがある場合のみ解析
                try:
                    scales_text = lora_scales_text.strip()
                    scales = [float(scale.strip()) for scale in scales_text.split(',') if scale.strip()]
                    current_lora_scales = scales
                    
                    if len(current_lora_scales) < len(current_lora_paths):
                        print(translate("[INFO] LoRAスケールの数が不足しているため、デフォルト値で補完します"))
                        current_lora_scales.extend([0.8] * (len(current_lora_paths) - len(current_lora_scales)))
                    elif len(current_lora_scales) > len(current_lora_paths):
                        print(translate("[INFO] LoRAスケールの数が多すぎるため、不要なものを切り捨てます"))
                        current_lora_scales = current_lora_scales[:len(current_lora_paths)]
                        
                    # 最終的なLoRAとスケールの対応を表示
                    for i, (path, scale) in enumerate(zip(current_lora_paths, current_lora_scales)):
                        print(translate("\u25c6 LoRA {0}: {1} (スケール: {2})").format(i+1, os.path.basename(path), scale))
                except Exception as e:
                    print(translate("[ERROR] LoRAスケール解析エラー: {0}").format(e))
                    # デフォルト値で埋める
                    current_lora_scales = [0.8] * len(current_lora_paths)
                    for i, (path, scale) in enumerate(zip(current_lora_paths, current_lora_scales)):
                        print(translate("\u25c6 LoRA {0}: {1} (デフォルトスケール: {2})").format(i+1, os.path.basename(path), scale))
        
        # -------- LoRA 設定 START ---------
        # UI設定のuse_loraフラグ値を保存
        original_use_lora = use_lora
        print(f"[DEBUG] UI設定のuse_loraフラグの値: {original_use_lora}")

        # UIでLoRA使用が有効になっていた場合、ファイル選択に関わらず強制的に有効化
        if original_use_lora:
            use_lora = True
            print(translate("[INFO] UIでLoRA使用が有効化されているため、LoRA使用を有効にします"))

        print(f"[DEBUG] 最終的なuse_loraフラグ: {use_lora}")

        # LoRA設定を更新（リロードは行わない）
        print(translate("\n[INFO] LoRA設定を更新します："))
        print(translate("  - LoRAパス: {0}").format(current_lora_paths))
        print(translate("  - LoRAスケール: {0}").format(current_lora_scales))
        print(translate("  - 高VRAM: {0}").format(high_vram))
        print(translate("  - FP8最適化: {0}").format(fp8_optimization))

        # LoRA設定のみを更新
        transformer_manager.set_next_settings(
            lora_paths=current_lora_paths,
            lora_scales=current_lora_scales,
            high_vram_mode=high_vram,
            fp8_enabled=fp8_optimization,  # fp8_enabledパラメータを追加
            force_dict_split=True  # 常に辞書分割処理を行う
        )
        # -------- LoRA 設定 END ---------

        # -------- FP8 設定 START ---------
        # FP8設定（既にLoRA設定に含めたので不要）
        # この行は削除しても問題ありません
        # -------- FP8 設定 END ---------
        
        # セクション処理開始前にtransformerの状態を確認
        print(translate("\nLoRA適用前のtransformer状態チェック..."))
        try:
            # transformerの状態を確認し、必要に応じてリロード
            if not transformer_manager.ensure_transformer_state():
                raise Exception(translate("transformer状態の確認に失敗しました"))
                
            # 最新のtransformerインスタンスを取得
            transformer = transformer_manager.get_transformer()
            print(translate("transformer状態チェック完了"))
        except Exception as e:
            print(translate("transformerのリロードに失敗しました: {0}").format(e))
            traceback.print_exc()
            raise Exception(translate("transformerのリロードに失敗しました"))
        
        # 入力画像の処理
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))
        
        # 入力画像がNoneの場合はデフォルトの黒い画像を作成
        if input_image is None:
            print(translate("[INFO] 入力画像が指定されていないため、黒い画像を生成します"))
            # 指定された解像度の黒い画像を生成（デフォルトは640x640）
            height = width = resolution
            input_image = np.zeros((height, width, 3), dtype=np.uint8)
            input_image_np = input_image
        elif isinstance(input_image, str):
            # 文字列（ファイルパス）の場合は画像をロード
            print(translate("[INFO] 入力画像がファイルパスのため、画像をロードします: {0}").format(input_image))
            try:
                from PIL import Image
                img = Image.open(input_image)
                input_image = np.array(img)
                if len(input_image.shape) == 2:  # グレースケール画像の場合
                    input_image = np.stack((input_image,) * 3, axis=-1)
                elif input_image.shape[2] == 4:  # アルファチャンネル付きの場合
                    input_image = input_image[:, :, :3]
                H, W, C = input_image.shape
                height, width = find_nearest_bucket(H, W, resolution=resolution)
                input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
            except Exception as e:
                print(translate("[ERROR] 画像のロードに失敗しました: {0}").format(e))
                # エラーが発生した場合はデフォルトの黒い画像を使用
                height = width = resolution
                input_image = np.zeros((height, width, 3), dtype=np.uint8)
                input_image_np = input_image
        else:
            # 通常の画像オブジェクトの場合（通常の処理）
            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=resolution)
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        
        # 入力画像は必要な場合のみ保存（デバッグ用）
        # Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}_input.png'))
        
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        
        # VAE エンコーディング
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        
        try:
            # エンコード前のメモリ状態を記録
            free_mem_before_encode = get_cuda_free_memory_gb(gpu)
            print(translate("\nVAEエンコード前の空きVRAM: {0} GB").format(free_mem_before_encode))
            
            # VAEモデルのロード（未ロードの場合）
            if vae is None:
                print(translate("\nVAEモデルを初めてロードします..."))
                try:
                    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
                    setup_vae_if_loaded()  # VAEの設定を適用
                    print(translate("VAEモデルのロードが完了しました"))
                except Exception as e:
                    print(translate("VAEモデルのロードに失敗しました: {0}").format(e))
                    traceback.print_exc()
                    print(translate("5秒間待機後に再試行します..."))
                    time.sleep(5)
                    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
                    setup_vae_if_loaded()  # VAEの設定を適用
            
            # ハイVRAM以外では明示的にモデルをGPUにロード
            if not high_vram:
                print(translate("\nVAEモデルをGPUにロード..."))
                load_model_as_complete(vae, target_device=gpu)
            
            # VAEエンコード実行
            with torch.no_grad():  # 明示的にno_gradコンテキストを使用
                # 効率的な処理のために入力をGPUで処理
                input_image_gpu = input_image_pt.to(gpu)
                start_latent = vae_encode(input_image_gpu, vae)
                
                # 入力をCPUに戻す
                del input_image_gpu
                torch.cuda.empty_cache()
            
            # ローVRAMモードでは使用後すぐにCPUに戻す
            if not high_vram:
                vae.to('cpu')
                
                # メモリ状態をログ
                free_mem_after_encode = get_cuda_free_memory_gb(gpu)
                print(translate("\nVAEエンコード後の空きVRAM: {0} GB").format(free_mem_after_encode))
                print(translate("\nVAEエンコードで使用したVRAM: {0} GB").format(free_mem_before_encode - free_mem_after_encode))
                
                # メモリクリーンアップ
                torch.cuda.empty_cache()
        except Exception as e:
            print(translate("VAEエンコードエラー: {0}").format(e))
            
            # エラー発生時のメモリ解放
            if 'input_image_gpu' in locals():
                del input_image_gpu
            torch.cuda.empty_cache()
            
            raise e
        
        # CLIP Vision エンコーディング
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        
        try:
            # 画像エンコーダのロード（未ロードの場合）
            if image_encoder is None:
                print(translate("\n画像エンコーダを初めてロードします..."))
                try:
                    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
                    setup_image_encoder_if_loaded()  # 画像エンコーダの設定を適用
                    print(translate("画像エンコーダのロードが完了しました"))
                except Exception as e:
                    print(translate("画像エンコーダのロードに失敗しました: {0}").format(e))
                    print(translate("再試行します..."))
                    time.sleep(2)
                    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
                    setup_image_encoder_if_loaded()  # 画像エンコーダの設定を適用
            
            if not high_vram:
                print(translate("\n画像エンコーダをGPUにロード..."))
                load_model_as_complete(image_encoder, target_device=gpu)
            
            # CLIP Vision エンコード実行
            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
            
            # ローVRAMモードでは使用後すぐにCPUに戻す
            if not high_vram:
                image_encoder.to('cpu')
                
                # メモリ状態をログ
                free_mem_gb = get_cuda_free_memory_gb(gpu)
                print(translate("\nCLIP Vision エンコード後の空きVRAM {0} GB").format(free_mem_gb))
                
                # メモリクリーンアップ
                torch.cuda.empty_cache()
        except Exception as e:
            print(translate("CLIP Vision エンコードエラー: {0}").format(e))
            raise e
        
        # テキストエンコーディング
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        
        # イメージキューでカスタムプロンプトを使用しているかどうかを確認
        using_custom_prompt = False
        current_prompt = prompt  # デフォルトは共通プロンプト

        if queue_enabled and queue_type == "image" and batch_index is not None and batch_index > 0:
            if batch_index - 1 < len(image_queue_files):
                queue_img_path = image_queue_files[batch_index - 1]
                img_basename = os.path.splitext(queue_img_path)[0]
                txt_path = f"{img_basename}.txt"
                if os.path.exists(txt_path):
                    try:
                        # テキストファイルからカスタムプロンプトを読み込む
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            custom_prompt = f.read().strip()
                        
                        # カスタムプロンプトを設定
                        current_prompt = custom_prompt
                        
                        img_name = os.path.basename(queue_img_path)
                        using_custom_prompt = True
                        print(translate("[カスタムプロンプト情報] イメージキュー画像「{0}」の専用プロンプトを使用しています").format(img_name))
                    except Exception as e:
                        print(translate("[エラー] カスタムプロンプトファイルの読み込みに失敗しました: {0}").format(e))
                        using_custom_prompt = False  # エラーが発生した場合は共通プロンプトを使用

        # キャッシュの使用判断
        global cached_prompt, cached_n_prompt, cached_llama_vec, cached_llama_vec_n
        global cached_clip_l_pooler, cached_clip_l_pooler_n, cached_llama_attention_mask, cached_llama_attention_mask_n
        
        # プロンプトが変更されたかチェック
        use_cache = (cached_prompt == prompt and cached_n_prompt == n_prompt and 
                    cached_llama_vec is not None and cached_llama_vec_n is not None)
        
        if use_cache:
            # キャッシュを使用
            print(translate("\nキャッシュされたテキストエンコード結果を使用します"))
            llama_vec = cached_llama_vec
            clip_l_pooler = cached_clip_l_pooler
            llama_vec_n = cached_llama_vec_n
            clip_l_pooler_n = cached_clip_l_pooler_n
            llama_attention_mask = cached_llama_attention_mask
            llama_attention_mask_n = cached_llama_attention_mask_n
        else:
            # キャッシュなし - 新規エンコード
            try:
                # 常にtext_encoder_managerから最新のテキストエンコーダーを取得する
                print(translate("\nテキストエンコーダを初期化します..."))
                try:
                    # text_encoder_managerを使用して初期化
                    if not text_encoder_manager.ensure_text_encoder_state():
                        print(translate("テキストエンコーダの初期化に失敗しました。再試行します..."))
                        time.sleep(3)
                        if not text_encoder_manager.ensure_text_encoder_state():
                            raise Exception(translate("テキストエンコーダとtext_encoder_2の初期化に複数回失敗しました"))
                    text_encoder, text_encoder_2 = text_encoder_manager.get_text_encoders()
                    print(translate("テキストエンコーダの初期化が完了しました"))
                except Exception as e:
                    print(translate("テキストエンコーダのロードに失敗しました: {0}").format(e))
                    traceback.print_exc()
                    raise e
                
                if not high_vram:
                    print(translate("\nテキストエンコーダをGPUにロード..."))
                    fake_diffusers_current_device(text_encoder, gpu)
                    load_model_as_complete(text_encoder_2, target_device=gpu)
                
                # テキストエンコーディング実行
                # 実際に使用されるプロンプトを必ず表示
                full_prompt = prompt  # 実際に使用するプロンプト
                prompt_source = "共通プロンプト" # プロンプトの種類

                # プロンプトソースの判定
                if queue_enabled and queue_type == "prompt" and batch_index is not None:
                    # プロンプトキューの場合
                    prompt_source = "プロンプトキュー"
                    print(translate("\nプロンプトキューからのプロンプトをエンコードしています..."))
                    print(translate("[DEBUG] プロンプトキュー使用: queue_enabled={0}, queue_type={1}, batch_index={2}").format(queue_enabled, queue_type, batch_index))
                elif using_custom_prompt:
                    # イメージキューのカスタムプロンプトの場合
                    full_prompt = current_prompt  # カスタムプロンプトを使用
                    prompt_source = "カスタムプロンプト"
                    print(translate("\nカスタムプロンプトをエンコードしています..."))
                else:
                    # 通常の共通プロンプトの場合
                    print(translate("\n共通プロンプトをエンコードしています..."))
                
                # プロンプトの内容とソースを表示
                print(translate("プロンプトソース: {0}").format(prompt_source))
                print(translate("プロンプト全文: {0}").format(full_prompt))
                print(translate("\nプロンプトをエンコードしています..."))
                llama_vec, clip_l_pooler = encode_prompt_conds(full_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                
                if cfg == 1:
                    llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
                else:
                    print(translate("\nネガティブプロンプトをエンコードしています..."))
                    llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                
                # ローVRAMモードでは使用後すぐにCPUに戻す
                if not high_vram:
                    if text_encoder is not None and hasattr(text_encoder, 'to'):
                        text_encoder.to('cpu')
                    if text_encoder_2 is not None and hasattr(text_encoder_2, 'to'):
                        text_encoder_2.to('cpu')
                    
                    # メモリ状態をログ
                    free_mem_gb = get_cuda_free_memory_gb(gpu)
                    print(translate("\nテキストエンコード後の空きVRAM {0} GB").format(free_mem_gb))
                    
                    # メモリクリーンアップ
                    torch.cuda.empty_cache()
                
                # エンコード結果をキャッシュ
                print(translate("\nエンコード結果をキャッシュします"))
                cached_prompt = prompt
                cached_n_prompt = n_prompt
                
                # エンコード処理後にキャッシュを更新
                llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
                llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
                
                # キャッシュを更新
                cached_llama_vec = llama_vec
                cached_llama_vec_n = llama_vec_n
                cached_clip_l_pooler = clip_l_pooler
                cached_clip_l_pooler_n = clip_l_pooler_n
                cached_llama_attention_mask = llama_attention_mask
                cached_llama_attention_mask_n = llama_attention_mask_n
                
            except Exception as e:
                print(translate("テキストエンコードエラー: {0}").format(e))
                raise e
        
        # キャッシュを使用する場合は既にcrop_or_pad_yield_maskが適用済み
        if not use_cache:
            # キャッシュを使用しない場合のみ適用
            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        
        # データ型変換
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        
        # endframe_ichiと同様に、テキストエンコーダーのメモリを完全に解放
        if not high_vram:
            print(translate("\nテキストエンコーダを完全に解放します"))
            # テキストエンコーダーを完全に解放（endframe_ichiと同様に）
            text_encoder, text_encoder_2 = None, None
            text_encoder_manager.dispose_text_encoders()
            # 明示的なキャッシュクリア
            torch.cuda.empty_cache()
        
        # 1フレームモード用の設定
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = 1  # 1フレームモード
        
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        
        # 1フレームモード用に特別に設定
        latent_paddings = [0] * total_latent_sections
        print(translate("[INFO] 1フレームモード: latent_paddings = {0}に設定しました").format(latent_paddings))
        
        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0  # 常にTrue
            latent_padding_size = latent_padding * latent_window_size  # 常に0
            
            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return
            
            # 1フレームモード用のindices設定
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            
            # 詳細設定のlatent_indexに基づいたインデックス処理
            all_indices = torch.arange(0, latent_window_size).unsqueeze(0)
            if latent_index > 0 and latent_index < latent_window_size:
                print(translate("\n[INFO] カスタムレイテントインデックス {0} を使用します").format(latent_index))
                # ユーザー指定のインデックスを使用
                latent_indices = all_indices[:, latent_index:latent_index+1]
            else:
                print(translate("\n[INFO] デフォルトの最後のインデックスを使用します"))
                # デフォルトは最後のインデックス
                latent_indices = all_indices[:, -1:]
            
            # clean_latents設定
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            
            # 形状の確認と修正
            print(translate("[DEBUG] start_latent形状: {0}").format(start_latent.shape))
            print(translate("[DEBUG] history_latents形状: {0}").format(history_latents.shape))
            
            # start_latentの形状を確認
            if len(start_latent.shape) < 5:  # バッチとフレーム次元がない場合
                # [B, C, H, W] → [B, C, 1, H, W] の形に変換
                clean_latents_pre = start_latent.unsqueeze(2).to(history_latents.dtype).to(history_latents.device)
            else:
                clean_latents_pre = start_latent.to(history_latents.dtype).to(history_latents.device)
            
            print(translate("[DEBUG] clean_latents_pre形状(変換後): {0}").format(clean_latents_pre.shape))
            
            # history_latentsからデータを適切に分割
            try:
                # 分割前に形状確認
                frames_to_split = history_latents.shape[2]
                
                if frames_to_split >= 19:  # 1+2+16フレームを想定
                    # 正常分割
                    clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                else:
                    # フレーム数が不足している場合は適切なサイズで初期化
                    print(translate("[WARN] フレーム数が不足しているため、ゼロで初期化します"))
                    clean_latents_post = torch.zeros(1, 16, 1, height // 8, width // 8, dtype=history_latents.dtype, device=history_latents.device)
                    clean_latents_2x = torch.zeros(1, 16, 2, height // 8, width // 8, dtype=history_latents.dtype, device=history_latents.device)
                    clean_latents_4x = torch.zeros(1, 16, 16, height // 8, width // 8, dtype=history_latents.dtype, device=history_latents.device)
            except Exception as e:
                print(translate("[ERROR] history_latentsの分割中にエラー: {0}").format(e))
                # エラー発生時はゼロで初期化
                clean_latents_post = torch.zeros(1, 16, 1, height // 8, width // 8, dtype=torch.float32, device='cpu')
                clean_latents_2x = torch.zeros(1, 16, 2, height // 8, width // 8, dtype=torch.float32, device='cpu')
                clean_latents_4x = torch.zeros(1, 16, 16, height // 8, width // 8, dtype=torch.float32, device='cpu')
            
            # 詳細設定のオプションに基づいて処理
            if use_clean_latents_post:
                try:
                    # 次元確認のためのログ出力
                    print(translate("\n[DEBUG] clean_latents_pre形状: {0}").format(clean_latents_pre.shape))
                    print(translate("[DEBUG] clean_latents_post形状: {0}").format(clean_latents_post.shape))
                    
                    # 正しい形状に変換して結合
                    if len(clean_latents_pre.shape) != len(clean_latents_post.shape):
                        print(translate("[DEBUG] 形状が異なるため次元調整を行います"))
                        # 形状を合わせる
                        if len(clean_latents_pre.shape) < len(clean_latents_post.shape):
                            clean_latents_pre = clean_latents_pre.unsqueeze(2)
                        else:
                            clean_latents_post = clean_latents_post.unsqueeze(1)
                    
                    # 次元調整後の形状確認
                    print(translate("[DEBUG] 調整後 clean_latents_pre形状: {0}").format(clean_latents_pre.shape))
                    print(translate("[DEBUG] 調整後 clean_latents_post形状: {0}").format(clean_latents_post.shape))
                    
                    # 結合
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                    print(translate("[DEBUG] 結合後 clean_latents形状: {0}").format(clean_latents.shape))
                except Exception as e:
                    print(translate("\n[ERROR] clean_latentsの結合中にエラーが発生しました: {0}").format(e))
                    print(translate("[FALLBACK] 前処理のみを使用します"))
                    clean_latents = clean_latents_pre
                    if len(clean_latents.shape) == 4:  # [B, C, H, W]
                        clean_latents = clean_latents.unsqueeze(2)  # [B, C, 1, H, W]
            else:
                print(translate("\n[OPTIMIZE] clean_latents_postは無効化されています。生成が高速化されますが、ノイズが増える可能性があります"))
                # clean_latents_postを使用しない場合、前処理+空白レイテント（ゼロテンソル）を結合
                # これはオリジナルの実装をできるだけ維持しつつ、エラーを回避するためのアプローチ
                clean_latents_pre_shaped = clean_latents_pre
                if len(clean_latents_pre.shape) == 4:  # [B, C, H, W]
                    clean_latents_pre_shaped = clean_latents_pre.unsqueeze(2)  # [B, C, 1, H, W]
                
                # 空のレイテントを作成（形状を合わせる）
                shape = list(clean_latents_pre_shaped.shape)
                # [B, C, 1, H, W]の形状に対して、[B, C, 1, H, W]の空テンソルを作成
                empty_latent = torch.zeros_like(clean_latents_pre_shaped)
                
                # 結合して形状を維持
                clean_latents = torch.cat([clean_latents_pre_shaped, empty_latent], dim=2)
                print(translate("[DEBUG] 代替手法による結合後 clean_latents形状: {0}").format(clean_latents.shape))
            
            # transformerの初期化とロード（未ロードの場合）
            if transformer is None:
                print(translate("\ntransformerモデルを初めてロードします..."))
                try:
                    # transformerの状態を確認
                    if not transformer_manager.ensure_transformer_state():
                        raise Exception(translate("transformer状態の確認に失敗しました"))
                        
                    # transformerインスタンスを取得
                    transformer = transformer_manager.get_transformer()
                    print(translate("transformerの初期化が完了しました"))
                except Exception as e:
                    print(translate("transformerのロードに失敗しました: {0}").format(e))
                    traceback.print_exc()
                    
                    # 再試行
                    print(translate("transformerのロードを再試行します..."))
                    time.sleep(5)
                    
                    if not transformer_manager.ensure_transformer_state():
                        raise Exception(translate("transformer状態の再確認に失敗しました"))
                    
                    transformer = transformer_manager.get_transformer()
            
            # endframe_ichiと同様にtransformerをGPUに移動
            # vae, text_encoder, text_encoder_2, image_encoderをCPUに移動し、メモリを解放
            if not high_vram:
                # GPUメモリの解放 - transformerは処理中に必要なので含めない
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae
                )

                # FP8最適化の有無に関わらず、gpu_complete_modulesに登録してから移動
                if transformer not in gpu_complete_modules:
                    # endframe_ichiと同様に、unload_complete_modulesで確実に解放されるようにする
                    gpu_complete_modules.append(transformer)
                    print(translate("transformerをgpu_complete_modulesに登録しました"))

                # メモリ確保した上でGPUへ移動
                # GPUメモリ保存値を明示的に浮動小数点に変換
                preserved_memory = float(gpu_memory_preservation) if gpu_memory_preservation is not None else 6.0
                print(translate('Setting transformer memory preservation to: {0} GB').format(preserved_memory))
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=preserved_memory)
            else:
                # ハイVRAMモードでも正しくロードしてgpu_complete_modulesに追加
                load_model_as_complete(transformer, target_device=gpu, unload=True)

            # teacacheの設定
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)
            
            def callback(d):
                try:
                    preview = d['denoised']
                    preview = vae_decode_fake(preview)
                    
                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                    
                    if stream.input_queue.top() == 'end':
                        # グローバル変数を直接設定
                        global batch_stopped, user_abort, user_abort_notified
                        batch_stopped = True
                        user_abort = True
                        
                        # 通知は一度だけ行うようにする - user_abort_notifiedが設定されていない場合のみ表示
                        # 通常は既にend_process()内で設定済みなのでここでは表示されない
                        if not user_abort_notified:
                            print(translate("\n[INFO] 開始前または現在の処理完了後に停止します..."))
                            user_abort_notified = True
                        
                        # 中断検出をoutput_queueに通知
                        stream.output_queue.push(('end', None))
                        
                        # 戻り値に特殊値を設定して上位処理で検知できるようにする
                        return {'user_interrupt': True}
                    
                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = f'Sampling {current_step}/{steps}'
                    desc = translate('1フレームモード: サンプリング中...')
                    stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                except KeyboardInterrupt:
                    print(f"[DEBUG] コールバック中のKeyboardInterrupt: 安全に停止処理")
                    # 例外を再スローしない - 戻り値で制御
                    return {'user_interrupt': True}
                except Exception as e:
                    import traceback
                    print(f"[DEBUG] コールバック内でエラー: {type(e).__name__} - {e}")
                    print(f"[DEBUG] コールバックエラースタック: {traceback.format_exc()}")
            
            # 詳細設定に基づいてパラメータを準備
            # 形状チェックのデバッグ
            print(translate("\n[DEBUG] clean_latents_2x形状: {0}").format(clean_latents_2x.shape))
            print(translate("[DEBUG] clean_latents_4x形状: {0}").format(clean_latents_4x.shape))

            # 異常な次元数を持つテンソルを処理
            try:
                if len(clean_latents_2x.shape) > 5:
                    print(translate("[DEBUG] clean_latents_2xの形状を修正します"))
                    # エラーメッセージは[1, 16, 1, 1, 96, 64]のような6次元テンソルを示しています
                    # 必要なのは5次元テンソル[B, C, T, H, W]です
                    if clean_latents_2x.shape[2] == 1 and clean_latents_2x.shape[3] == 1:
                        # 余分な次元を削除
                        clean_latents_2x = clean_latents_2x.squeeze(2)  # [1, 16, 1, 96, 64]
                        print(translate("[DEBUG] 調整後 clean_latents_2x形状: {0}").format(clean_latents_2x.shape))
            except Exception as e:
                print(translate("[ERROR] clean_latents_2xの形状調整中にエラー: {0}").format(e))
            
            try:
                if len(clean_latents_4x.shape) > 5:
                    print(translate("[DEBUG] clean_latents_4xの形状を修正します"))
                    if clean_latents_4x.shape[2] == 1 and clean_latents_4x.shape[3] == 1:
                        # 余分な次元を削除
                        clean_latents_4x = clean_latents_4x.squeeze(2)  # [1, 16, 1, 96, 64]
                        print(translate("[DEBUG] 調整後 clean_latents_4x形状: {0}").format(clean_latents_4x.shape))
            except Exception as e:
                print(translate("[ERROR] clean_latents_4xの形状調整中にエラー: {0}").format(e))
            
            # clean_latents_2xとclean_latents_4xの設定に応じて変数を調整
            clean_latents_2x_param = clean_latents_2x if use_clean_latents_2x else None
            clean_latents_4x_param = clean_latents_4x if use_clean_latents_4x else None
            
            # 最適化オプションのログ
            if not use_clean_latents_2x:
                print(translate("\n[INFO] clean_latents_2xは無効化されています。出力画像に変化が発生します"))
            if not use_clean_latents_4x:
                print(translate("\n[INFO] clean_latents_4xは無効化されています。出力画像に変化が発生します"))
                
            # RoPE値を設定 - transformer内部のmax_positionを設定してみる
            print(translate("\n[INFO] 設定されたRoPE値(latent_window_size): {0}").format(latent_window_size))
            
            try:
                # transformerモデルの内部パラメータを調整
                # HunyuanVideoTransformerモデル内部のmax_positionに相当する値を変更する
                if hasattr(transformer, 'max_pos_embed_window_size'):
                    original_value = transformer.max_pos_embed_window_size
                    print(translate("[INFO] 元のmax_pos_embed_window_size: {0}").format(original_value))
                    transformer.max_pos_embed_window_size = latent_window_size
                    print(translate("[INFO] max_pos_embed_window_sizeを{0}に設定しました").format(latent_window_size))
                
                # RoFormerなどのRoPE実装を探して調整
                if hasattr(transformer, 'attn_processors'):
                    print(translate("[INFO] attn_processorsが見つかりました、RoPE関連設定を探します"))
                    # 詳細は出力しない
                
                # HunyuanVideo特有の実装を探す
                if hasattr(transformer, 'create_image_rotary_emb'):
                    print(translate("[INFO] create_image_rotary_embを調整中..."))
            except Exception as e:
                print(translate("[WARN] RoPE値の設定中にエラーが発生しました: {0}").format(e))
                print(translate("[INFO] デフォルト値を使用します"))
            
            # この値は保存されますが、実際のモデル内部には適用されません（テンソルサイズエラー回避のため）
            print(translate("[INFO] sample_hunyuan関数を呼び出します"))
            
            # サンプリング前の最終メモリクリア（endframe_ichiと同様）
            print(translate("\nサンプリング前の最終メモリ最適化を実行"))
            # 不要な変数を明示的に解放
            image_encoder = None
            vae = None
            # 明示的なキャッシュクリア
            torch.cuda.empty_cache()
            
            # sample_hunyuan関数呼び出し部分
            try:
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu,
                    dtype=torch.bfloat16,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x_param,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x_param,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )
                
                # コールバックからの戻り値をチェック（コールバック関数が特殊な値を返した場合）
                if isinstance(generated_latents, dict) and generated_latents.get('user_interrupt'):
                    # ユーザーが中断したことを検出したが、メッセージは出さない（既に表示済み）
                    # 現在のバッチは完了させる（KeyboardInterruptは使わない）
                    print(translate("[DEBUG] バッチ内処理を完了します"))
                else:
                    print(translate("[INFO] 生成は正常に完了しました"))
                
                # サンプリング直後のメモリクリーンアップ（重要）
                # transformerの中間状態を明示的にクリア（KVキャッシュに相当）
                if hasattr(transformer, 'enable_teacache'):
                    transformer.enable_teacache = False
                    print(translate("[INFO] transformerのキャッシュをクリア"))
                
                # 不要なモデル変数を積極的に解放
                torch.cuda.empty_cache()
                
            except KeyboardInterrupt:
                print(translate("[INFO] キーボード割り込みを検出しました - 安全に停止します"))
                # リソースのクリーンアップ
                del llama_vec, llama_vec_n, llama_attention_mask, llama_attention_mask_n
                del clip_l_pooler, clip_l_pooler_n
                try:
                    # モデルをCPUに移動（可能な場合のみ）
                    if 'transformer' in locals() and transformer is not None:
                        if hasattr(transformer, 'cpu'):
                            transformer.cpu()
                    # GPUキャッシュをクリア
                    torch.cuda.empty_cache()
                except Exception as cleanup_e:
                    print(translate("[WARN] 停止時のクリーンアップでエラー: {0}").format(cleanup_e))
                # バッチ停止フラグを設定
                batch_stopped = True
                return None
                
            except RuntimeError as e:
                error_msg = str(e)
                if "size of tensor" in error_msg:
                    print(translate("\n[ERROR] テンソルサイズの不一致エラーが発生しました: {0}").format(error_msg))
                    print(translate("[ERROR] 現在の実装ではこのエラーは発生しないはずです。開発者に報告してください。"))
                    
                    # デバッグ情報を提供
                    print(translate("[DEBUG] clean_latents形状: {0}").format(clean_latents.shape if 'clean_latents' in locals() else "未定義"))
                    print(translate("[DEBUG] latent_indices形状: {0}").format(latent_indices.shape if 'latent_indices' in locals() else "未定義"))
                    raise e
                else:
                    # その他のランタイムエラーはそのまま投げる
                    raise e
            
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            
            # 生成完了後のメモリ最適化 - 軽量な処理に変更
            if not high_vram:
                # transformerのメモリを軽量に解放（辞書リセットなし）
                print(translate("\n生成完了 - transformerをアンロード中..."))

                # 元の方法に戻す - 軽量なオフロードで速度とメモリのバランスを取る
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)

                # アンロード後のメモリ状態をログ
                free_mem_gb_after_unload = get_cuda_free_memory_gb(gpu)
                print(translate("transformerアンロード後の空きVRAM {0} GB").format(free_mem_gb_after_unload))
                
                # VAEをデコード用にロード（VAEが解放されていた場合は再ロード）
                if vae is None:
                    print(translate("VAEモデルを再ロードします..."))
                    try:
                        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
                        setup_vae_if_loaded()  # VAEの設定を適用
                        print(translate("VAEモデルの再ロードが完了しました"))
                    except Exception as e:
                        print(translate("VAEモデルの再ロードに失敗しました: {0}").format(e))
                        traceback.print_exc()
                        print(translate("5秒間待機後に再試行します..."))
                        time.sleep(5)
                        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
                        setup_vae_if_loaded()  # VAEの設定を適用
                
                print(translate("VAEをGPUにロード中..."))
                load_model_as_complete(vae, target_device=gpu)
                
                # ロード後のメモリ状態をログ
                free_mem_gb_after_vae = get_cuda_free_memory_gb(gpu)
                print(translate("VAEロード後の空きVRAM {0} GB").format(free_mem_gb_after_vae))
                
                # 追加のメモリクリーンアップ
                torch.cuda.empty_cache()
            
            # 実際に使用するラテントを抽出
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            
            # 使用していないテンソルを早めに解放
            del history_latents
            torch.cuda.empty_cache()
            
            # 1フレームモードではVAEデコードを行い、画像を直接保存
            try:
                print(translate("[INFO] 1フレームモード: 生成されたラテントから直接画像を保存します"))
                print(translate("[INFO] 現在のラテント形状: {0}").format(real_history_latents.shape))
                
                # VAEデコード処理前にメモリを確認
                free_mem_before_decode = get_cuda_free_memory_gb(gpu)
                print(translate("[INFO] VAEデコード前の空きVRAM: {0} GB").format(free_mem_before_decode))
                
                # VAEが解放されていた場合は再ロード
                if vae is None:
                    print(translate("VAEモデルを最終デコード用に再ロードします..."))
                    try:
                        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
                        setup_vae_if_loaded()  # VAEの設定を適用
                        # VAEをGPUに移動
                        load_model_as_complete(vae, target_device=gpu)
                        print(translate("VAEモデルの再ロードが完了しました"))
                    except Exception as e:
                        print(translate("VAEモデルの再ロードに失敗しました: {0}").format(e))
                        traceback.print_exc()
                        raise e
                
                # VAEデコード処理
                with torch.no_grad():  # 明示的にno_gradコンテキストを使用
                    # ラテントをCPUではなくGPUに置いて効率的に処理
                    real_history_latents_gpu = real_history_latents.to(gpu)
                    decoded_image = vae_decode(real_history_latents_gpu, vae).cpu()
                    
                    # 不要なGPU上のラテントをすぐに解放
                    del real_history_latents_gpu
                    torch.cuda.empty_cache()
                
                # デコード後にVAEをCPUに移動してメモリを解放
                if not high_vram and vae is not None:
                    vae.to('cpu')
                    torch.cuda.empty_cache()
                
                # デコード後のメモリを確認
                free_mem_after_decode = get_cuda_free_memory_gb(gpu)
                print(translate("[INFO] VAEデコード後の空きVRAM: {0} GB").format(free_mem_after_decode))
                
                # 単一フレームを抽出
                frame = decoded_image[0, :, 0, :, :]
                frame = torch.clamp(frame, -1., 1.) * 127.5 + 127.5
                frame = frame.detach().cpu().to(torch.uint8)
                frame = einops.rearrange(frame, 'c h w -> h w c').numpy()
                
                # デコード結果を解放
                del decoded_image
                del real_history_latents
                torch.cuda.empty_cache()
                
                # メタデータを設定
                metadata = {
                    PROMPT_KEY: prompt,
                    SEED_KEY: str(seed)
                }
                
                # 画像として保存（メタデータ埋め込み）
                from PIL import Image
                output_filename = os.path.join(outputs_folder, f'{job_id}_oneframe.png')
                pil_img = Image.fromarray(frame)
                pil_img.save(output_filename)  # 一度保存
                
                # メタデータを埋め込み
                try:
                    # 関数は2つの引数しか取らないので修正
                    embed_metadata_to_png(output_filename, metadata)
                    print(translate("[INFO] 画像メタデータを埋め込みました"))
                except Exception as e:
                    print(translate("[WARNING] メタデータ埋め込みエラー: {0}").format(e))
                
                print(translate("[INFO] 1フレーム画像を保存しました: {0}").format(output_filename))
                
                # MP4保存はスキップして、画像ファイルパスを返す
                stream.output_queue.push(('file', output_filename))
                
            except Exception as e:
                print(translate("[ERROR] 1フレームの画像保存中にエラーが発生しました: {0}").format(e))
                traceback.print_exc()
                
                # エラー発生時のメモリ解放を試みる
                if 'real_history_latents_gpu' in locals():
                    del real_history_latents_gpu
                if 'real_history_latents' in locals():
                    del real_history_latents
                if 'decoded_image' in locals():
                    del decoded_image
                torch.cuda.empty_cache()
            
            break  # 1フレーム生成は1回のみ
            
    except Exception as e:
        print(translate("\n処理中にエラーが発生しました: {0}").format(e))
        traceback.print_exc()
        
        # エラー時の詳細なメモリクリーンアップ
        try:
            if not high_vram:
                print(translate("\nエラー発生時のメモリクリーンアップを実行..."))
                
                # 効率的なクリーンアップのために、重いモデルから順にアンロード
                models_to_unload = [
                    ('transformer', transformer), 
                    ('vae', vae), 
                    ('image_encoder', image_encoder), 
                    ('text_encoder', text_encoder), 
                    ('text_encoder_2', text_encoder_2)
                ]
                
                # 各モデルを個別にアンロードして解放
                for model_name, model in models_to_unload:
                    if model is not None:
                        try:
                            print(translate("{0}をアンロード中...").format(model_name))
                            # モデルをCPUに移動
                            if hasattr(model, 'to'):
                                model.to('cpu')
                            # 参照を明示的に削除
                            if model_name == 'transformer':
                                transformer = None
                            elif model_name == 'vae':
                                vae = None
                            elif model_name == 'image_encoder':
                                image_encoder = None
                            elif model_name == 'text_encoder':
                                text_encoder = None
                            elif model_name == 'text_encoder_2':
                                text_encoder_2 = None
                            # 各モデル解放後にすぐメモリ解放
                            torch.cuda.empty_cache()
                        except Exception as unload_error:
                            print(translate("{0}のアンロード中にエラー: {1}").format(model_name, unload_error))
                
                # 一括アンロード - endframe_ichiと同じアプローチでモデルを明示的に解放
                if transformer is not None:
                    # まずtransformer_managerの状態をリセット - これが重要
                    transformer_manager.current_state['is_loaded'] = False
                    # FP8最適化モードの有無に関わらず常にCPUに移動
                    transformer.to('cpu')
                    print(translate("transformerをCPUに移動しました"))

                # endframe_ichi.pyと同様に明示的にすべてのモデルを一括アンロード
                # モデルを直接リストで渡す（引数展開ではなく）
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )
                print(translate("すべてのモデルをアンロードしました"))
                
                # 明示的なガベージコレクション（複数回）
                import gc
                print(translate("ガベージコレクション実行中..."))
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                
                # メモリ状態を報告
                free_mem_gb = get_cuda_free_memory_gb(gpu)
                print(translate("クリーンアップ後の空きVRAM {0} GB").format(free_mem_gb))
                
                # 追加の変数クリーンアップ
                for var_name in ['start_latent', 'decoded_image', 'history_latents', 'real_history_latents', 
                              'real_history_latents_gpu', 'generated_latents', 'input_image_pt', 'input_image_gpu']:
                    if var_name in locals():
                        try:
                            exec(f"del {var_name}")
                            print(translate("変数 {0} を解放しました").format(var_name))
                        except:
                            pass
        except Exception as cleanup_error:
            print(translate("メモリクリーンアップ中にエラー: {0}").format(cleanup_error))
    
    # 処理完了を通知（個別バッチの完了）
    print(translate("\n処理が完了しました"))
    
    # worker関数内では効果音を鳴らさない（バッチ処理全体の完了時のみ鳴らす）
    
    stream.output_queue.push(('end', None))
    return

def handle_open_folder_btn(folder_name):
    """フォルダ名を保存し、そのフォルダを開く - endframe_ichiと同じ実装"""
    if not folder_name or not folder_name.strip():
        folder_name = "outputs"

    # フォルダパスを取得
    folder_path = get_output_folder_path(folder_name)

    # 設定を更新して保存
    settings = load_settings()
    old_folder_name = settings.get('output_folder')

    if old_folder_name != folder_name:
        settings['output_folder'] = folder_name
        save_result = save_settings(settings)
        if save_result:
            # グローバル変数も更新
            global output_folder_name, outputs_folder
            output_folder_name = folder_name
            outputs_folder = folder_path
        print(translate("出力フォルダ設定を保存しました: {folder_name}").format(folder_name=folder_name))

    # フォルダを開く
    open_output_folder(folder_path)

    # 出力ディレクトリ入力欄とパス表示を更新
    return gr.update(value=folder_name), gr.update(value=folder_path)
    
def update_from_image_metadata(image, should_copy):
    """画像からメタデータを抽出してプロンプトとシードを更新する関数
    
    Args:
        image: 画像データ（numpy配列）
        should_copy: メタデータを複写するかどうかの指定
        
    Returns:
        tuple: (プロンプト更新データ, シード値更新データ)
    """
    if not should_copy or image is None:
        return gr.update(), gr.update()
    
    try:
        # NumPy配列からメタデータを抽出
        metadata = extract_metadata_from_numpy_array(image)
        
        if not metadata:
            print(translate("画像にメタデータが含まれていません"))
            return gr.update(), gr.update()
        
        # プロンプトとSEEDをUIに反映
        prompt_update = gr.update()
        seed_update = gr.update()
        
        if PROMPT_KEY in metadata and metadata[PROMPT_KEY]:
            prompt_update = gr.update(value=metadata[PROMPT_KEY])
            print(translate("プロンプトを画像から取得: {0}").format(metadata[PROMPT_KEY]))
        
        if SEED_KEY in metadata and metadata[SEED_KEY]:
            # SEED値を整数に変換
            try:
                seed_value = int(metadata[SEED_KEY])
                seed_update = gr.update(value=seed_value)
                print(translate("シード値を画像から取得: {0}").format(seed_value))
            except ValueError:
                print(translate("シード値の変換に失敗しました: {0}").format(metadata[SEED_KEY]))
        
        return prompt_update, seed_update
    
    except Exception as e:
        print(translate("メタデータ抽出エラー: {0}").format(e))
        return gr.update(), gr.update()

def check_metadata_on_checkbox_change(should_copy, image):
    """チェックボックスの状態が変更された時に画像からメタデータを抽出する関数"""
    return update_from_image_metadata(image, should_copy)

def process(input_image, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache,
            lora_files, lora_files2, lora_scales_text, use_lora, fp8_optimization, resolution, output_directory=None,
            batch_count=1, use_random_seed=False, latent_window_size=9, latent_index=0,
            use_clean_latents_2x=True, use_clean_latents_4x=True, use_clean_latents_post=True,
            lora_mode=None, lora_dropdown1=None, lora_dropdown2=None, lora_dropdown3=None, lora_files3=None,
            use_rope_batch=False, use_queue=False, prompt_queue_file=None):
    global stream
    global batch_stopped, user_abort, user_abort_notified
    global queue_enabled, queue_type, prompt_queue_file_path, image_queue_files

    # 引数の型確認
    print(translate("[DEBUG] process: キュー設定 - use_queue = {0} (型: {1})").format(use_queue, type(use_queue).__name__))
    if prompt_queue_file is not None:
        print(translate("[DEBUG] process: prompt_queue_file = {0} (型: {1})").format(
            prompt_queue_file.name if hasattr(prompt_queue_file, 'name') else "ファイル名なし",
            type(prompt_queue_file).__name__))

    # 新たな処理開始時にグローバルフラグをリセット
    user_abort = False
    user_abort_notified = False
    
    # プロセス開始時にバッチ中断フラグをリセット
    batch_stopped = False

    # バッチ処理回数を確認し、詳細を出力
    # 型チェックしてから変換（数値でない場合はデフォルト値の1を使用）
    try:
        batch_count_val = int(batch_count)
        batch_count = max(1, min(batch_count_val, 100))  # 1〜100の間に制限
    except (ValueError, TypeError):
        print(translate("[WARN] バッチ処理回数が無効です。デフォルト値の1を使用します: {0}").format(batch_count))
        batch_count = 1  # デフォルト値
        
    # キュー関連の設定を保存
    queue_enabled = bool(use_queue)  # UIからの値をブール型に変換
    
    # プロンプトキューファイルが指定されている場合はパスを保存
    if queue_enabled and prompt_queue_file is not None:
        if hasattr(prompt_queue_file, 'name') and os.path.exists(prompt_queue_file.name):
            prompt_queue_file_path = prompt_queue_file.name
            queue_type = "prompt"  # キュータイプをプロンプトに設定
            print(translate("プロンプトキューファイルパスを設定: {0}").format(prompt_queue_file_path))
            
            # プロンプトファイルの内容を確認し、行数を出力
            try:
                with open(prompt_queue_file_path, 'r', encoding='utf-8') as f:
                    prompt_lines = [line.strip() for line in f.readlines() if line.strip()]
                    prompt_count = len(prompt_lines)
                    if prompt_count > 0:
                        if batch_count > prompt_count:
                            print(translate("\u25c6 バッチ処理回数: {0}回（プロンプトキュー行を優先: {1}行、残りは共通プロンプトで実施）").format(batch_count, prompt_count))
                        else:
                            print(translate("\u25c6 バッチ処理回数: {0}回（プロンプトキュー行を優先: {1}行）").format(batch_count, prompt_count))
                    else:
                        print(translate("\u25c6 バッチ処理回数: {0}回").format(batch_count))
            except Exception as e:
                print(translate("プロンプトファイル読み込みエラー: {0}").format(str(e)))
                print(translate("\u25c6 バッチ処理回数: {0}回").format(batch_count))
        else:
            print(translate("警告: プロンプトキューファイルが存在しません: {0}").format(
                prompt_queue_file.name if hasattr(prompt_queue_file, 'name') else "不明なファイル"))
            print(translate("\u25c6 バッチ処理回数: {0}回").format(batch_count))
    else:
        print(translate("\u25c6 バッチ処理回数: {0}回").format(batch_count))

    # キュー機能の設定を更新
    queue_ui_value = bool(use_queue)
    queue_enabled = queue_ui_value

    # キュー設定のログ出力
    print(translate("[DEBUG] キュー関連: use_queue_ui={0}, グローバル変数queue_enabled={1}").format(
        queue_ui_value, queue_enabled))

    # プロンプトキューの処理
    if queue_enabled and prompt_queue_file is not None:
        queue_type = "prompt"  # キュータイプをプロンプトに設定

        # アップロードされたファイルパスを取得
        if hasattr(prompt_queue_file, 'name') and os.path.exists(prompt_queue_file.name):
            prompt_queue_file_path = prompt_queue_file.name
            print(translate("プロンプトキューファイルパスを設定: {0}").format(prompt_queue_file_path))

            # ファイルの内容を確認（デバッグ用）
            try:
                with open(prompt_queue_file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    queue_prompts_count = len(lines)
                    print(translate("有効なプロンプト行数: {0}").format(queue_prompts_count))

                    if queue_prompts_count > 0:
                        # サンプルとして最初の数行を表示
                        sample_lines = lines[:min(3, queue_prompts_count)]
                        print(translate("プロンプトサンプル: {0}").format(sample_lines))

                        # バッチ数をプロンプト数に合わせる
                        if queue_prompts_count > batch_count:
                            print(translate("プロンプト数に合わせてバッチ数を自動調整: {0} → {1}").format(
                                batch_count, queue_prompts_count))
                            batch_count = queue_prompts_count
            except Exception as e:
                print(translate("プロンプトキューファイル読み込みエラー: {0}").format(str(e)))
        else:
            print(translate("プロンプトキューファイルが存在しないか無効です"))

    # イメージキューの処理
    elif queue_enabled and queue_ui_value and use_queue:
        # イメージキューが選択された場合
        queue_type = "image"  # キュータイプをイメージに設定

        # 入力フォルダから画像ファイルリストを取得
        get_image_queue_files()

        # イメージキューの数を確認
        image_queue_count = len(image_queue_files)
        print(translate("イメージキュー: {0}個の画像ファイルを読み込みました").format(image_queue_count))

        if image_queue_count > 0:
            # 入力画像を使う1回 + 画像ファイル分のバッチ数
            total_needed_batches = 1 + image_queue_count

            # 設定されたバッチ数より必要数が多い場合は調整
            if total_needed_batches > batch_count:
                print(translate("画像キュー数+1に合わせてバッチ数を自動調整: {0} → {1}").format(
                    batch_count, total_needed_batches))
                batch_count = total_needed_batches
    
    # 出力フォルダの設定
    global outputs_folder
    global output_folder_name
    if output_directory and output_directory.strip():
        # 出力フォルダパスを取得
        outputs_folder = get_output_folder_path(output_directory)
        print(translate("出力フォルダを設定: {0}").format(outputs_folder))

        # フォルダ名が現在の設定と異なる場合は設定ファイルを更新
        if output_directory != output_folder_name:
            settings = load_settings()
            settings['output_folder'] = output_directory
            if save_settings(settings):
                output_folder_name = output_directory
                print(translate("出力フォルダ設定を保存しました: {0}").format(output_directory))
    else:
        # デフォルト設定を使用
        outputs_folder = get_output_folder_path(output_folder_name)
        print(translate("デフォルト出力フォルダを使用: {0}").format(outputs_folder))

    # フォルダが存在しない場合は作成
    os.makedirs(outputs_folder, exist_ok=True)
    
    # 出力ディレクトリを設定
    output_dir = outputs_folder
    
    # バッチ処理のパラメータチェック
    batch_count = max(1, min(int(batch_count), 100))  # 1〜100の間に制限
    print(translate("\u25c6 バッチ処理回数: {0}回").format(batch_count))
    
    # 入力画像チェック - 厳格なチェックを避け、エラーを出力するだけに変更
    if input_image is None:
        print(translate("[WARN] 入力画像が指定されていません。デフォルトの画像を生成します。"))
        # 空の入力画像を生成
        # ここではNoneのままとし、実際のworker関数内でNoneの場合に対応する
    
    # LoRAの状態をログ出力
    if use_lora and has_lora_support:
        print(translate("[INFO] LoRAの使用設定: use_lora = {0}, has_lora_support = {1}").format(use_lora, has_lora_support))
        print(translate("[INFO] lora_mode = {0}").format(lora_mode))
        if lora_mode == translate("ファイルアップロード"):
            print(translate("[INFO] lora_files = {0}, 型: {1}").format(lora_files, type(lora_files)))
            print(translate("[INFO] lora_files2 = {0}, 型: {1}").format(lora_files2, type(lora_files2)))
            print(translate("[INFO] lora_files3 = {0}, 型: {1}").format(lora_files3, type(lora_files3)))
        else:
            print(translate("[INFO] lora_dropdown1 = {0}").format(lora_dropdown1))
            print(translate("[INFO] lora_dropdown2 = {0}").format(lora_dropdown2))
            print(translate("[INFO] lora_dropdown3 = {0}").format(lora_dropdown3))
        print(translate("[INFO] lora_scales_text = {0}").format(lora_scales_text))
    
    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)
    
    # バッチ処理用の変数 - 各フラグをリセット
    batch_stopped = False
    user_abort = False
    user_abort_notified = False
    original_seed = seed if seed else (random.randint(0, 2**32 - 1) if use_random_seed else 31337)
    
    # バッチ処理ループ
    # バッチ処理のサマリーを出力
    if queue_enabled:
        if queue_type == "prompt" and prompt_queue_file_path is not None:
            # プロンプトキュー情報をログに出力
            try:
                with open(prompt_queue_file_path, 'r', encoding='utf-8') as f:
                    queue_lines = [line.strip() for line in f.readlines() if line.strip()]
                    queue_lines_count = len(queue_lines)
                    print(translate("\n◆ バッチ処理情報: 合計{0}回").format(batch_count))
                    print(translate("◆ プロンプトキュー: 有効, プロンプト行数={0}行").format(queue_lines_count))

                    # 各プロンプトの概要を出力
                    print(translate("\n◆ プロンプトキュー内容:"))
                    for i in range(min(batch_count, queue_lines_count)):
                        prompt_preview = queue_lines[i][:50] + "..." if len(queue_lines[i]) > 50 else queue_lines[i]
                        print(translate("   └ バッチ{0}: {1}").format(i+1, prompt_preview))
            except:
                pass
        elif queue_type == "image" and len(image_queue_files) > 0:
            # イメージキュー情報をログに出力
            print(translate("\n◆ バッチ処理情報: 合計{0}回").format(batch_count))
            print(translate("◆ イメージキュー: 有効, 画像ファイル数={0}個").format(len(image_queue_files)))

            # 各画像ファイルの概要を出力
            print(translate("\n◆ イメージキュー内容:"))
            print(translate("   └ バッチ1: 入力画像 (最初のバッチは常に入力画像を使用)"))
            for i, img_path in enumerate(image_queue_files[:min(batch_count-1, len(image_queue_files))]):
                img_name = os.path.basename(img_path)
                print(translate("   └ バッチ{0}: {1}").format(i+2, img_name))
    else:
        print(translate("\n◆ バッチ処理情報: 合計{0}回").format(batch_count))
        print(translate("◆ キュー機能: 無効"))

    for batch_index in range(batch_count):
        # 停止フラグが設定されている場合は全バッチ処理を中止
        if batch_stopped:
            print(translate("\nバッチ処理がユーザーによって中止されました"))
            yield (
                None,
                gr.update(visible=False),
                translate("バッチ処理が中止されました。"),
                '',
                gr.update(interactive=True),
                gr.update(interactive=False, value=translate("End Generation"))
            )
            break

        # バッチ情報をログ出力
        if batch_count > 1:
            batch_info = translate("バッチ処理: {0}/{1}").format(batch_index + 1, batch_count)
            print(f"\n{batch_info}")
            # UIにもバッチ情報を表示
            yield None, gr.update(visible=False), batch_info, "", gr.update(interactive=False), gr.update(interactive=True)

        # 今回処理用のプロンプトとイメージを取得（キュー機能対応）
        current_prompt = prompt
        current_image = input_image

        # キュー機能の処理
        if queue_enabled:
            if queue_type == "prompt" and prompt_queue_file_path is not None:
                # プロンプトキューの処理
                if os.path.exists(prompt_queue_file_path):
                    try:
                        with open(prompt_queue_file_path, 'r', encoding='utf-8') as f:
                            lines = [line.strip() for line in f.readlines() if line.strip()]
                            if batch_index < len(lines):
                                # プロンプトキューからプロンプトを取得
                                current_prompt = lines[batch_index]
                                print(f"◆ プロンプトキュー実行中: バッチ {batch_index+1}/{batch_count}")
                                print(f"  └ プロンプト: 「{current_prompt[:50]}...」")
                            else:
                                print(f"◆ プロンプトキュー実行中: バッチ {batch_index+1}/{batch_count} はプロンプト行数を超えているため元のプロンプトを使用")
                    except Exception as e:
                        print(f"◆ プロンプトキューファイル読み込みエラー: {str(e)}")

            elif queue_type == "image" and len(image_queue_files) > 0:
                # イメージキューの処理
                # 最初のバッチは入力画像を使用
                if batch_index == 0:
                    print(f"◆ イメージキュー実行中: バッチ {batch_index+1}/{batch_count} は入力画像を使用")
                elif batch_index > 0:
                    # 2回目以降はイメージキューの画像を順番に使用
                    image_index = batch_index - 1  # 0回目（入力画像）の分を引く

                    if image_index < len(image_queue_files):
                        current_image = image_queue_files[image_index]
                        image_filename = os.path.basename(current_image)
                        print(f"◆ イメージキュー実行中: バッチ {batch_index+1}/{batch_count} の画像「{image_filename}」")
                        print(f"  └ 画像ファイルパス: {current_image}")
                        
                        # 同名のテキストファイルがあるか確認し、あれば内容をプロンプトとして使用
                        img_basename = os.path.splitext(current_image)[0]
                        txt_path = f"{img_basename}.txt"
                        if os.path.exists(txt_path):
                            try:
                                with open(txt_path, 'r', encoding='utf-8') as f:
                                    custom_prompt = f.read().strip()
                                if custom_prompt:
                                    print(translate("イメージキュー: 画像「{0}」用のテキストファイルを読み込みました").format(image_filename))
                                    print(translate("カスタムプロンプト: {0}").format(custom_prompt[:50] + "..." if len(custom_prompt) > 50 else custom_prompt))
                                    # カスタムプロンプトを設定（current_promptを上書き）
                                    current_prompt = custom_prompt
                            except Exception as e:
                                print(translate("イメージキュー: テキストファイル読み込みエラー: {0}").format(e))
                    else:
                        # 画像数が足りない場合は入力画像に戻る
                        print(f"◆ イメージキュー実行中: バッチ {batch_index+1}/{batch_count} は画像数を超えているため入力画像を使用")

        # RoPE値バッチ処理の場合はRoPE値をインクリメント、それ以外は通常のシードインクリメント
        current_seed = original_seed
        current_latent_window_size = latent_window_size
        
        if use_rope_batch:
            # RoPE値をインクリメント（最大64まで）
            new_rope_value = latent_window_size + batch_index
            
            # RoPE値が64を超えたら処理を終了
            if new_rope_value > 64:
                print(translate("\u25c6 RoPE値が上限（64）に達したため、処理を終了します"))
                break
                
            current_latent_window_size = new_rope_value
            print(translate("\u25c6 現在のRoPE値: {0}").format(current_latent_window_size))
        else:
            # 通常のバッチ処理：シード値をインクリメント
            current_seed = original_seed + batch_index
            if batch_count > 1:
                print(translate("\u25c6 初期シード値: {0}").format(current_seed))
        
        if batch_stopped:
            break
            
        try:
            # 新しいストリームを作成
            stream = AsyncStream()
            # 新しいストリームを作成（デバッグログ削除）
            
            # バッチインデックスをジョブIDに含める
            batch_suffix = f"{batch_index}" if batch_index > 0 else ""
            
            # 中断フラグの再確認
            if batch_stopped:
                print(f"[DEBUG] バッチ開始前に中断フラグが検出されました: batch_index={batch_index}")
                break
                
            # ワーカー実行 - 詳細設定パラメータを含む（キュー機能対応）
            async_run(worker, current_image, current_prompt, n_prompt, current_seed, steps, cfg, gs, rs,
                     gpu_memory_preservation, use_teacache, lora_files, lora_files2, lora_scales_text,
                     output_dir, use_lora, fp8_optimization, resolution,
                     current_latent_window_size, latent_index, use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post,
                     lora_mode, lora_dropdown1, lora_dropdown2, lora_dropdown3, lora_files3,
                     batch_index, use_queue, prompt_queue_file)
        except Exception as e:
            import traceback
            print(f"[DEBUG] バッチ{batch_index+1}の実行中にエラー発生: {type(e).__name__} - {e}")
            print(f"[DEBUG] エラースタック: {traceback.format_exc()}")
        
        output_filename = None
        
        # ジョブ完了まで監視
        try:
            # ストリーム待機開始（デバッグログは削除）
            while True:
                try:
                    flag, data = stream.output_queue.next()
                    
                    if flag == 'file':
                        output_filename = data
                        yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)
                    
                    if flag == 'progress':
                        preview, desc, html = data
                        yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
                    
                    if flag == 'end':
                        # endフラグを受信（デバッグログ削除）
                        # バッチ処理中は最後の画像のみを表示
                        if batch_index == batch_count - 1 or batch_stopped:  # 最後のバッチまたは中断された場合
                            completion_message = ""
                            if batch_stopped:
                                completion_message = translate("バッチ処理が中断されました（{0}/{1}）").format(batch_index + 1, batch_count)
                            else:
                                completion_message = translate("バッチ処理が完了しました（{0}/{1}）").format(batch_count, batch_count)
                            
                            # 完了メッセージでUIを更新
                            yield output_filename, gr.update(visible=False), completion_message, '', gr.update(interactive=True, value=translate("Start Generation")), gr.update(interactive=False, value=translate("End Generation"))
                        break
                        
                    # ユーザーが中断した場合
                    if stream.input_queue.top() == 'end' or batch_stopped:
                        batch_stopped = True
                        # 処理ループ内での中断検出（デバッグログ削除）
                        print(translate("バッチ処理が中断されました（{0}/{1}）").format(batch_index + 1, batch_count))
                        # endframe_ichiと同様のシンプルな実装に戻す
                        yield output_filename, gr.update(visible=False), translate("バッチ処理が中断されました"), '', gr.update(interactive=True), gr.update(interactive=False, value=translate("End Generation"))
                        return
                        
                except Exception as e:
                    import traceback
                    print(f"[DEBUG] ストリーム処理中にエラー: {type(e).__name__} - {e}")
                    print(f"[DEBUG] ストリームエラースタック: {traceback.format_exc()}")
                    # エラー後はループを抜ける
                    break
                    
        except KeyboardInterrupt:
            print(f"[DEBUG] ストリーム待機中にKeyboardInterrupt: 中断して資源を解放します")
            # 明示的なリソースクリーンアップ
            try:
                # グローバルモデル変数のクリーンアップ
                global transformer, text_encoder, text_encoder_2, vae, image_encoder
                # 各モデルが存在する場合にCPUに移動
                if transformer is not None and hasattr(transformer, 'cpu'):
                    try:
                        transformer.cpu()
                    except: pass
                if text_encoder is not None and hasattr(text_encoder, 'cpu'):
                    try:
                        text_encoder.cpu()
                    except: pass
                if text_encoder_2 is not None and hasattr(text_encoder_2, 'cpu'):
                    try:
                        text_encoder_2.cpu()
                    except: pass
                if vae is not None and hasattr(vae, 'cpu'):
                    try:
                        vae.cpu()
                    except: pass
                if image_encoder is not None and hasattr(image_encoder, 'cpu'):
                    try:
                        image_encoder.cpu()
                    except: pass
                
                # GPUキャッシュの完全クリア
                torch.cuda.empty_cache()
                print(f"[DEBUG] リソースのクリーンアップが完了しました")
            except Exception as cleanup_e:
                print(f"[DEBUG] リソースクリーンアップ中にエラー: {cleanup_e}")
            
            # UIをリセット
            yield None, gr.update(visible=False), translate("キーボード割り込みにより処理が中断されました"), '', gr.update(interactive=True, value=translate("Start Generation")), gr.update(interactive=False, value=translate("End Generation"))
            return
        except Exception as e:
            import traceback
            print(f"[DEBUG] バッチ処理外部ループでエラー: {type(e).__name__} - {e}")
            print(f"[DEBUG] 外部エラースタック: {traceback.format_exc()}")
            # UIをリセット
            yield None, gr.update(visible=False), translate("エラーにより処理が中断されました"), '', gr.update(interactive=True, value=translate("Start Generation")), gr.update(interactive=False, value=translate("End Generation"))
            return
    
    # すべてのバッチ処理が正常に完了した場合と中断された場合で表示メッセージを分ける
    if batch_stopped:
        if user_abort:
            print(translate("\n[INFO] ユーザーの指示により処理を停止しました"))
        else:
            print(translate("\n[INFO] バッチ処理が中断されました"))
    else:
        print(translate("\n[INFO] 全てのバッチ処理が完了しました"))
    
    # バッチ処理終了後は必ずフラグをリセット
    batch_stopped = False
    user_abort = False
    user_abort_notified = False
    
    # 処理完了時の効果音
    if HAS_WINSOUND:
        try:
            # Windows環境では完了音を鳴らす
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
            print(translate("[INFO] Windows完了通知音を再生しました"))
        except Exception as e:
            print(translate("[WARN] 完了通知音の再生に失敗しました: {0}").format(e))
    
    # 処理状態に応じてメッセージを表示
    if batch_stopped or user_abort:
        print("\n" + "-" * 50)
        print(translate("【ユーザー中断】処理は正常に中断されました - ") + time.strftime("%Y-%m-%d %H:%M:%S"))
        print("-" * 50 + "\n")
    else:
        print("\n" + "*" * 50)
        print(translate("【全バッチ処理完了】プロセスが完了しました - ") + time.strftime("%Y-%m-%d %H:%M:%S"))
        print("*" * 50 + "\n")
            
    return

def end_process():
    """生成終了ボタンが押された時の処理"""
    global stream
    global batch_stopped, user_abort, user_abort_notified

    # 重複停止通知を防止するためのチェック
    if not user_abort:
        # 現在のバッチと次のバッチ処理を全て停止するフラグを設定
        batch_stopped = True
        user_abort = True
        
        # 通知は一度だけ表示（ここで表示してフラグを設定）
        print(translate("\n停止ボタンが押されました。開始前または現在の処理完了後に停止します..."))
        user_abort_notified = True  # 通知フラグを設定
        
        # 現在実行中のバッチを停止
        stream.input_queue.push('end')

    # ボタンの名前を一時的に変更することでユーザーに停止処理が進行中であることを表示
    return gr.update(value=translate("停止処理中..."))

css = get_app_css()  # eichi_utilsのスタイルを使用
block = gr.Blocks(css=css).queue()
with block:
    # eichiと同じ半透明度スタイルを使用
    gr.HTML('<h1>FramePack<span class="title-suffix">-oichi</span></h1>')
    
    # 初期化時にtransformerの状態確認は行わない（必要時に遅延ロード）
    # ここではロードをスキップして、ワーカー関数内で必要になったときにだけロードする
    
    # Use Random Seedの初期値（先に定義して後で使う）
    use_random_seed_default = True
    seed_default = random.randint(0, 2**32 - 1) if use_random_seed_default else 31337
    
    # eichiのプリセット管理関連のインポート
    from eichi_utils.preset_manager import get_default_startup_prompt, load_presets, save_preset, delete_preset
    
    with gr.Row():
        with gr.Column(scale=1):
            # 左カラム - eichiと同じ順番
            # モードについての説明を画像枠の上に表示
            gr.Markdown(translate("**「1フレーム推論」モードでは、1枚の新しい未来の画像を生成します。**"))
            
            input_image = gr.Image(sources='upload', type="numpy", label=translate("画像"), height=320)
            
            # 解像度設定（画像の直下に）
            resolution = gr.Dropdown(
                label=translate("解像度"),
                choices=[512, 640, 768, 960, 1080],
                value=640,
                info=translate("出力画像の基準解像度。640推奨。960/1080は高負荷・高メモリ消費")
            )
            
            # バッチ処理設定 - endframe_ichiと同じUIにする
            with gr.Column(scale=1):
                batch_count = gr.Slider(
                    label=translate("バッチ処理回数"),
                    minimum=1,
                    maximum=100,
                    value=1,
                    step=1,
                    info=translate("同じ設定で連続生成する回数。SEEDは各回で+1されます")
                )
                
                # RoPE値バッチ処理用のチェックボックス
                use_rope_batch = gr.Checkbox(
                    label=translate("RoPE値バッチ処理を使用"),
                    value=False,
                    info=translate("チェックすると、SEEDではなくRoPE値を各バッチで+1していきます（64に達すると停止）")
                )

                # キュー機能設定 - endframe_ichiと同様の実装
                with gr.Column(scale=1):
                    # キュー機能のグループ
                    with gr.Group():
                        gr.Markdown(f"### " + translate("キュー機能"))

                        # キュー機能の使用有無
                        use_queue = gr.Checkbox(
                            label=translate("キュー機能を使用"),
                            value=False,
                            info=translate("入力ディレクトリの画像または指定したプロンプトリストを使用して連続して画像を生成します")
                        )

                        # キュータイプの選択
                        queue_type_selector = gr.Radio(
                            choices=[translate("プロンプトキュー"), translate("イメージキュー")],
                            value=translate("プロンプトキュー"),
                            label=translate("キュータイプ"),
                            visible=False,
                            interactive=True
                        )

                        # プロンプトキュー設定コンポーネント（初期状態では非表示）
                        with gr.Group(visible=False) as prompt_queue_group:
                            # プロンプトキューファイル入力
                            prompt_queue_file = gr.File(
                                label=translate("プロンプトキューファイル (.txt) - 1行に1つのプロンプトが記載されたテキストファイル"),
                                file_types=[".txt"]
                            )
                            gr.Markdown(translate("※ ファイル内の各行が別々のプロンプトとして処理されます。\n※ チェックボックスがオフの場合は無効。\n※ バッチ処理回数より行数が多い場合は行数分処理されます。\n※ バッチ処理回数が1でもキュー回数が優先されます。"))

                        # イメージキュー設定コンポーネント（初期状態では非表示）
                        with gr.Group(visible=False) as image_queue_group:
                            gr.Markdown(translate("※ 1回目はImage画像を使用し、2回目以降は入力フォルダの画像ファイルを名前順に使用します。\n※ バッチ回数が全画像数を超える場合、残りはImage画像で処理されます。\n※ バッチ処理回数が1でもキュー回数が優先されます。"))

                            # 入力フォルダ設定
                            with gr.Row():
                                input_folder_name = gr.Textbox(
                                    label=translate("入力フォルダ名"),
                                    value=input_folder_name_value,
                                    info=translate("入力画像ファイルを格納するフォルダ名")
                                )
                                open_input_folder_btn = gr.Button(value="📂 " + translate("保存及び入力フォルダを開く"), size="md")

                                # 入力フォルダ名の変更を検知してグローバル変数を更新する関数（設定保存はしない）
                                def update_input_folder(folder_name):
                                    global input_folder_name_value
                                    # 無効な文字を削除（パス区切り文字やファイル名に使えない文字）
                                    folder_name = ''.join(c for c in folder_name if c.isalnum() or c in ('_', '-'))
                                    input_folder_name_value = folder_name
                                    print(translate("入力フォルダ名をメモリに保存: {0}（保存及び入力フォルダを開くボタンを押すと保存されます）").format(folder_name))
                                    return gr.update(value=folder_name)

                                # 入力フォルダを開く関数
                                def open_input_folder():
                                    """入力フォルダを開く処理（保存も実行）"""
                                    global input_folder_name_value

                                    # 念のため設定を保存
                                    settings = load_settings()
                                    settings['input_folder'] = input_folder_name_value
                                    save_settings(settings)
                                    print(translate("入力フォルダ設定を保存しました: {0}").format(input_folder_name_value))

                                    # 入力フォルダのパスを取得
                                    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_folder_name_value)

                                    # フォルダが存在しなければ作成
                                    if not os.path.exists(input_dir):
                                        os.makedirs(input_dir, exist_ok=True)
                                        print(translate("入力ディレクトリを作成しました: {0}").format(input_dir))

                                    # 画像ファイルリストを更新
                                    get_image_queue_files()

                                    # フォルダを開く
                                    open_folder(input_dir)
                                    return None

                        # キュー機能のトグルハンドラー
                        def toggle_queue_settings(use_queue_val):
                            # グローバル変数を使用
                            global queue_enabled, queue_type

                            # デバッグ情報
                            print(f"トグル関数: チェックボックスの型={type(use_queue_val).__name__}, 値={use_queue_val}")

                            # チェックボックスの値をブール値に確実に変換
                            is_enabled = False

                            # Gradioオブジェクトの場合
                            if hasattr(use_queue_val, 'value'):
                                is_enabled = bool(use_queue_val.value)
                            else:
                                # 直接値の場合
                                is_enabled = bool(use_queue_val)

                            # グローバル変数を更新
                            queue_enabled = is_enabled

                            if is_enabled:
                                # キューが有効の場合
                                print(translate("[DEBUG] キュー機能を有効化しました"))

                                # キュータイプセレクタとキュータイプに応じたグループを表示
                                if queue_type == "prompt":
                                    return [
                                        gr.update(visible=True),  # queue_type_selector
                                        gr.update(visible=True),  # prompt_queue_group
                                        gr.update(visible=False)  # image_queue_group
                                    ]
                                else:  # image
                                    # イメージキュー選択時は画像ファイルリストを更新
                                    get_image_queue_files()
                                    return [
                                        gr.update(visible=True),  # queue_type_selector
                                        gr.update(visible=False),  # prompt_queue_group
                                        gr.update(visible=True)   # image_queue_group
                                    ]
                            else:
                                # キューが無効の場合、すべて非表示
                                print(translate("[DEBUG] キュー機能を無効化しました"))
                                queue_enabled = False
                                return [
                                    gr.update(visible=False),  # queue_type_selector
                                    gr.update(visible=False),  # prompt_queue_group
                                    gr.update(visible=False)   # image_queue_group
                                ]

                        # キュータイプ切替ハンドラー
                        def toggle_queue_type(queue_type_val):
                            global queue_type

                            # キュータイプをグローバル変数に保存
                            if queue_type_val == translate("プロンプトキュー"):
                                queue_type = "prompt"
                                return [gr.update(visible=True), gr.update(visible=False)]
                            else:
                                queue_type = "image"
                                # イメージキューを選択した場合、画像ファイルリストを更新
                                get_image_queue_files()
                                return [gr.update(visible=False), gr.update(visible=True)]

                        # イベントハンドラーの登録
                        use_queue.change(
                            fn=toggle_queue_settings,
                            inputs=[use_queue],
                            outputs=[queue_type_selector, prompt_queue_group, image_queue_group]
                        )

                        # キュータイプの選択イベントに関数を紐づけ
                        queue_type_selector.change(
                            fn=toggle_queue_type,
                            inputs=[queue_type_selector],
                            outputs=[prompt_queue_group, image_queue_group]
                        )

                        # 入力フォルダ名の変更イベントに関数を紐づけ
                        input_folder_name.change(
                            fn=update_input_folder,
                            inputs=[input_folder_name],
                            outputs=[input_folder_name]
                        )

                        # 入力フォルダを開くボタンにイベントを紐づけ
                        open_input_folder_btn.click(
                            fn=open_input_folder,
                            inputs=[],
                            outputs=[gr.Textbox(visible=False)]  # 一時的なフィードバック表示用（非表示）
                        )

            # 直接出力フォルダを開くボタンは削除（後で「保存および出力フォルダを開く」ボタンに置き換え）

            # 生成開始/中止ボタン - endframe_ichiと完全に同じ実装
            with gr.Row():
                start_button = gr.Button(value=translate("Start Generation"))
                end_button = gr.Button(value=translate("End Generation"), interactive=False)

            # FP8最適化設定
            with gr.Row():
                fp8_optimization = gr.Checkbox(
                    label=translate("FP8 最適化"),
                    value=True,
                    info=translate("メモリ使用量を削減し速度を改善（PyTorch 2.1以上が必要）")
                )

            # 埋め込みプロンプト機能 - 参照用に定義（表示はLoRA設定の下で行う）
            # グローバル変数として定義し、後で他の場所から参照できるようにする
            global copy_metadata
            copy_metadata = gr.Checkbox(
                label=translate("埋め込みプロンプトおよびシードを複写する"),
                value=False,
                info=translate("チェックをオンにすると、画像のメタデータからプロンプトとシードを自動的に取得します"),
                visible=False  # 元の位置では非表示
            )
                
            # メタデータ抽出結果表示用（非表示）
            extracted_info = gr.Markdown(visible=False)
            extracted_prompt = gr.Textbox(visible=False)
            extracted_seed = gr.Textbox(visible=False)
            
            # 詳細設定アコーディオン - 埋め込みプロンプト機能の直後に配置
            with gr.Accordion(translate("詳細設定"), open=False, elem_classes="section-accordion"):
                # レイテント処理設定セクション
                gr.Markdown(f"### " + translate("レイテント処理設定"))

                with gr.Row():
                    with gr.Column(scale=1):
                        # ツイートに基づくRoPE値の設定
                        latent_window_size = gr.Slider(
                            label=translate("RoPE値 (latent_window_size)"),
                            minimum=1,
                            maximum=64,
                            value=9,  # デフォルト値
                            step=1,
                            interactive=True,  # 明示的に対話可能に設定
                            info=translate("動きの変化量に影響します。大きい値ほど大きな変化が発生します。モデルの内部調整用パラメータです。")
                        )
                    
                    with gr.Column(scale=1):
                        # レイテントインデックス
                        latent_index = gr.Slider(
                            label=translate("レイテントインデックス"),
                            minimum=0,
                            maximum=64,
                            value=0,  # デフォルト値
                            step=1,
                            interactive=True,  # 明示的に対話可能に設定
                            info=translate("0は基本、大きい値で衣装変更などの効果が得られる場合があります。値が大きいとノイズが増えます。")
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # clean_latents_2xの有効/無効
                        use_clean_latents_2x = gr.Checkbox(
                            label=translate("clean_latents_2xを使用"),
                            value=True,
                            interactive=True,  # 明示的に対話可能に設定
                            info=translate("オフにすると変化が発生します。画質や速度に影響があります")
                        )
                    
                    with gr.Column(scale=1):
                        # clean_latents_4xの有効/無効
                        use_clean_latents_4x = gr.Checkbox(
                            label=translate("clean_latents_4xを使用"),
                            value=True,
                            interactive=True,  # 明示的に対話可能に設定
                            info=translate("オフにすると変化が発生します。画質や速度に影響があります")
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # clean_latents_postの有効/無効
                        use_clean_latents_post = gr.Checkbox(
                            label=translate("clean_latents_postを使用"),
                            value=True,
                            interactive=True,  # 明示的に対話可能に設定
                            info=translate("オフにするとかなり速くなりますが、ノイズが増える可能性があります")
                        )
            
            # 前回選択したLoRAモードを保存するためのグローバル変数
            global previous_lora_mode
            if 'previous_lora_mode' not in globals():
                previous_lora_mode = translate("ディレクトリから選択") 
                
            # LoRA設定 - endframe_ichiと同様の実装に拡張
            if has_lora_support:
                with gr.Group() as lora_settings_group:
                    gr.Markdown(f"### " + translate("LoRA設定"))
                    
                    # LoRA使用有無のチェックボックス
                    use_lora = gr.Checkbox(label=translate("LoRAを使用する"), value=False, info=translate("チェックをオンにするとLoRAを使用します（要16GB VRAM以上）"))

                    # LoRAモード選択（初期状態では非表示）
                    lora_mode = gr.Radio(
                        choices=[translate("ディレクトリから選択"), translate("ファイルアップロード")],
                        value=translate("ディレクトリから選択"),
                        label=translate("LoRA読み込み方式"),
                        visible=False  # 初期状態では非表示（toggle_lora_settingsで制御）
                    )

                    # ファイルアップロードグループ（初期状態では非表示）
                    with gr.Group(visible=False) as lora_upload_group:
                        # メインのLoRAファイル
                        lora_files = gr.File(
                            label=translate("LoRAファイル1 (.safetensors, .pt, .bin)"),
                            file_types=[".safetensors", ".pt", ".bin"]
                        )
                        # 追加のLoRAファイル
                        lora_files2 = gr.File(
                            label=translate("LoRAファイル2 (.safetensors, .pt, .bin)"),
                            file_types=[".safetensors", ".pt", ".bin"]
                        )
                        # 3つ目のLoRAファイル
                        lora_files3 = gr.File(
                            label=translate("LoRAファイル3 (.safetensors, .pt, .bin)"),
                            file_types=[".safetensors", ".pt", ".bin"]
                        )

                    # ディレクトリ選択グループ（初期状態では非表示）
                    with gr.Group(visible=False) as lora_dropdown_group:
                        # LoRAドロップダウン
                        none_choice = translate("なし")
                        lora_dropdown1 = gr.Dropdown(label=translate("LoRA1"), choices=[none_choice], value=none_choice)
                        lora_dropdown2 = gr.Dropdown(label=translate("LoRA2"), choices=[none_choice], value=none_choice)
                        lora_dropdown3 = gr.Dropdown(label=translate("LoRA3"), choices=[none_choice], value=none_choice)
                        
                        # ドロップダウン更新ボタン（下に配置）
                        lora_scan_button = gr.Button(value=translate("LoRAフォルダを再スキャン"), variant="secondary")

                    # スケール値の入力フィールド
                    lora_scales_text = gr.Textbox(
                        label=translate("LoRA適用強度 (カンマ区切り)"),
                        value="0.8,0.8,0.8",
                        info=translate("各LoRAのスケール値をカンマ区切りで入力 (例: 0.8,0.5,0.3)"),
                        visible=False
                    )

                    # LoRAディレクトリからファイル一覧を取得する関数
                    def scan_lora_directory():
                        """./loraディレクトリからLoRAモデルファイルを検索する関数"""
                        lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
                        choices = []
                        
                        # ディレクトリが存在しない場合は作成
                        if not os.path.exists(lora_dir):
                            os.makedirs(lora_dir, exist_ok=True)
                            print(translate("[INFO] LoRAディレクトリが存在しなかったため作成しました: {0}").format(lora_dir))
                        
                        # ディレクトリ内のファイルをリストアップ
                        for filename in os.listdir(lora_dir):
                            if filename.endswith(('.safetensors', '.pt', '.bin')):
                                choices.append(filename)
                        
                        # 空の選択肢がある場合は"なし"を追加
                        choices = sorted(choices)
                        
                        # なしの選択肢を最初に追加
                        none_choice = translate("なし")
                        choices.insert(0, none_choice)
                        
                        # 全ての選択肢が確実に文字列型であることを確認
                        for i, choice in enumerate(choices):
                            if not isinstance(choice, str):
                                # 明示的に文字列に変換
                                choices[i] = str(choice)
                        
                        print(translate("[INFO] LoRAディレクトリから{0}個のモデルを検出しました").format(len(choices) - 1))
                        return choices

                    # チェックボックスの状態によってLoRA設定の表示/非表示を切り替える関数
                    def toggle_lora_settings(use_lora):
                        # グローバル変数を使うように修正
                        global previous_lora_mode

                        # まだグローバル変数が定義されていなければ初期化
                        if 'previous_lora_mode' not in globals():
                            global previous_lora_mode
                            previous_lora_mode = translate("ディレクトリから選択")

                        # 現在のモード値を取得（UI要素が存在する場合）
                        current_mode = getattr(lora_mode, 'value', translate("ディレクトリから選択"))

                        # LoRAが無効化される場合、現在のモードを記憶
                        if not use_lora and current_mode:
                            previous_lora_mode = current_mode
                            print(translate("[DEBUG] 前回のLoRAモードを保存: {0}").format(previous_lora_mode))

                        if use_lora:
                            # LoRA使用時は前回のモードを復元
                            is_upload_mode = previous_lora_mode == translate("ファイルアップロード")

                            # 選択肢の更新
                            choices = scan_lora_directory() if not is_upload_mode else None

                            # モードに基づいた表示設定
                            return [
                                gr.update(visible=True, value=previous_lora_mode),  # lora_mode - 前回の値を復元
                                gr.update(visible=is_upload_mode),  # lora_upload_group
                                gr.update(visible=not is_upload_mode),  # lora_dropdown_group
                                gr.update(visible=True),  # lora_scales_text
                            ]
                        else:
                            # LoRA不使用時はLoRA関連UIのみ非表示（FP8最適化は表示したまま）
                            return [
                                gr.update(visible=False),  # lora_mode
                                gr.update(visible=False),  # lora_upload_group
                                gr.update(visible=False),  # lora_dropdown_group
                                gr.update(visible=False),  # lora_scales_text
                            ]
                    
                    # LoRA読み込み方式に応じて表示を切り替える関数
                    def toggle_lora_mode(mode):
                        # 前回のモードを更新
                        global previous_lora_mode
                        previous_lora_mode = mode
                        print(translate("[DEBUG] LoRAモードを変更: {0}").format(mode))
                        
                        if mode == translate("ディレクトリから選択"):
                            # ディレクトリから選択モードの場合
                            # 最初にディレクトリをスキャン
                            choices = scan_lora_directory()
                            
                            # 選択肢が確実に更新されるようにする
                            return [
                                gr.update(visible=False),                                # lora_upload_group
                                gr.update(visible=True),                                 # lora_dropdown_group
                                gr.update(choices=choices, value=choices[0]),            # lora_dropdown1
                                gr.update(choices=choices, value=choices[0]),            # lora_dropdown2
                                gr.update(choices=choices, value=choices[0])             # lora_dropdown3
                            ]
                        else:  # ファイルアップロード
                            # ファイルアップロード方式の場合、ドロップダウンの値は更新しない
                            return [
                                gr.update(visible=True),   # lora_upload_group
                                gr.update(visible=False),  # lora_dropdown_group
                                gr.update(),               # lora_dropdown1 - 変更なし
                                gr.update(),               # lora_dropdown2 - 変更なし
                                gr.update()                # lora_dropdown3 - 変更なし
                            ]
                    
                    # スキャンボタンの処理関数
                    def update_lora_dropdowns():
                        choices = scan_lora_directory()
                        # 各ドロップダウンを更新
                        return [
                            gr.update(choices=choices, value=choices[0]),  # lora_dropdown1
                            gr.update(choices=choices, value=choices[0]),  # lora_dropdown2
                            gr.update(choices=choices, value=choices[0]),  # lora_dropdown3
                        ]
                    
                    # LoRA使用チェックボックスの切り替え後にドロップダウンを更新する統合関数
                    def toggle_lora_full_update(use_lora_val):
                        global previous_lora_mode
                        
                        # まずLoRA設定全体の表示/非表示を切り替え
                        mode_updates = toggle_lora_settings(use_lora_val)
                        
                        # LoRAが有効で、かつディレクトリ選択モード時にドロップダウンを更新
                        if use_lora_val and previous_lora_mode == translate("ディレクトリから選択"):
                            choices = scan_lora_directory()
                            # ドロップダウン更新
                            dropdown_updates = [
                                gr.update(choices=choices, value=choices[0]),  # lora_dropdown1
                                gr.update(choices=choices, value=choices[0]),  # lora_dropdown2
                                gr.update(choices=choices, value=choices[0])   # lora_dropdown3
                            ]
                            return mode_updates + dropdown_updates
                        
                        # それ以外の場合は変更なし
                        return mode_updates + [gr.update(), gr.update(), gr.update()]
                    
                    # チェックボックスの変更イベントに統合関数を紐づけ
                    use_lora.change(
                        fn=toggle_lora_full_update,
                        inputs=[use_lora],
                        outputs=[lora_mode, lora_upload_group, lora_dropdown_group,
                                 lora_scales_text,
                                 lora_dropdown1, lora_dropdown2, lora_dropdown3]
                    )
                    
                    # LoRA読み込み方式の変更イベントに表示切替関数を紐づけ
                    lora_mode.change(
                        fn=toggle_lora_mode,
                        inputs=[lora_mode],
                        outputs=[lora_upload_group, lora_dropdown_group, lora_dropdown1, lora_dropdown2, lora_dropdown3]
                    )
                    
                    # スキャンボタンの処理を紐づけ
                    lora_scan_button.click(
                        fn=update_lora_dropdowns,
                        inputs=[],
                        outputs=[lora_dropdown1, lora_dropdown2, lora_dropdown3]
                    )

                    # UIロード後に自動的に初期化ボタンをクリックするJavaScriptを追加
                    js_init_code = """
                    function initLoraDropdowns() {
                        // UIロード後、少し待ってからボタンをクリック
                        setTimeout(function() {
                            // LoRAフォルダを再スキャンボタンを探して自動クリック
                            var scanBtns = document.querySelectorAll('button');
                            var scanBtn = null;
                            
                            for (var i = 0; i < scanBtns.length; i++) {
                                if (scanBtns[i].textContent.includes('LoRAフォルダを再スキャン')) {
                                    scanBtn = scanBtns[i];
                                    break;
                                }
                            }
                            
                            if (scanBtn) {
                                console.log('LoRAドロップダウン初期化ボタンを自動実行します');
                                scanBtn.click();
                            }
                        }, 1000); // 1秒待ってから実行
                    }
                    
                    // ページロード時に初期化関数を呼び出し
                    window.addEventListener('load', initLoraDropdowns);
                    """
                    
                    # JavaScriptコードをUIに追加
                    gr.HTML(f"<script>{js_init_code}</script>")

                    # LoRAサポートが無効の場合のメッセージ
                    if not has_lora_support:
                        gr.Markdown(translate("LoRAサポートは現在無効です。lora_utilsモジュールが必要です。"))
            else:
                # LoRAサポートが無効の場合はダミー変数を作成
                use_lora = gr.Checkbox(visible=False, value=False)
                lora_mode = gr.Radio(visible=False, value=translate("ディレクトリから選択"))
                lora_upload_group = gr.Group(visible=False)
                lora_dropdown_group = gr.Group(visible=False)
                lora_files = gr.File(visible=False)
                lora_files2 = gr.File(visible=False)
                lora_files3 = gr.File(visible=False)
                lora_dropdown1 = gr.Dropdown(visible=False)
                lora_dropdown2 = gr.Dropdown(visible=False)
                lora_dropdown3 = gr.Dropdown(visible=False)
                lora_scales_text = gr.Textbox(visible=False, value="0.8,0.8,0.8")

            # LoRA設定の下に埋め込みプロンプトおよびシードを複写するチェックボックスを表示
            # 埋め込みプロンプトおよびシードを複写するチェックボックス（LoRA設定の下に表示）
            copy_metadata_visible = gr.Checkbox(
                label=translate("埋め込みプロンプトおよびシードを複写する"),
                value=False,
                info=translate("チェックをオンにすると、画像のメタデータからプロンプトとシードを自動的に取得します")
            )

            # 表示用チェックボックスと実際の処理用チェックボックスを同期
            copy_metadata_visible.change(
                fn=lambda x: x,
                inputs=[copy_metadata_visible],
                outputs=[copy_metadata]
            )

            # 元のチェックボックスが変更されたときも表示用を同期
            copy_metadata.change(
                fn=lambda x: x,
                inputs=[copy_metadata],
                outputs=[copy_metadata_visible],
                queue=False  # 高速化のためキューをスキップ
            )

            # プロンプト入力
            prompt = gr.Textbox(label=translate("プロンプト"), value=get_default_startup_prompt(), lines=6)
            n_prompt = gr.Textbox(label=translate("ネガティブプロンプト"), value='')
            
            # プロンプト管理パネル
            with gr.Group(visible=True) as prompt_management:
                gr.Markdown(f"### " + translate("プロンプト管理"))
                
                # 編集画面を常時表示する
                with gr.Group(visible=True):
                    # 起動時デフォルトの初期表示用に取得
                    default_prompt = ""
                    default_name = ""
                    for preset in load_presets()["presets"]:
                        if preset.get("is_startup_default", False):
                            default_prompt = preset["prompt"]
                            default_name = preset["name"]
                            break
                    
                    with gr.Row():
                        edit_name = gr.Textbox(label=translate("プリセット名"), placeholder=translate("名前を入力..."), value=default_name)
                    
                    edit_prompt = gr.Textbox(label=translate("プロンプト"), lines=5, value=default_prompt)
                    
                    with gr.Row():
                        # 起動時デフォルトをデフォルト選択に設定
                        default_preset = translate("起動時デフォルト")
                        # プリセットデータから全プリセット名を取得
                        presets_data = load_presets()
                        choices = [preset["name"] for preset in presets_data["presets"]]
                        default_presets = [name for name in choices if any(p["name"] == name and p.get("is_default", False) for p in presets_data["presets"])]
                        user_presets = [name for name in choices if name not in default_presets]
                        sorted_choices = [(name, name) for name in sorted(default_presets) + sorted(user_presets)]
                        preset_dropdown = gr.Dropdown(label=translate("プリセット"), choices=sorted_choices, value=default_preset, type="value")
                    
                    with gr.Row():
                        save_btn = gr.Button(value=translate("保存"), variant="primary")
                        apply_preset_btn = gr.Button(value=translate("反映"), variant="primary")
                        clear_btn = gr.Button(value=translate("クリア"))
                        delete_preset_btn = gr.Button(value=translate("削除"))
                
                # メッセージ表示用
                result_message = gr.Markdown("")
                
        with gr.Column(scale=1):
            # 右カラム - 生成結果と設定
            result_image = gr.Image(label=translate("生成結果"), height=512)
            preview_image = gr.Image(label=translate("処理中のプレビュー"), height=200, visible=False)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            
            # endframe_ichiと同じ順序で設定項目を配置
            # TeaCacheとランダムシード設定
            use_teacache = gr.Checkbox(label=translate('Use TeaCache'), value=True, info=translate('Faster speed, but often makes hands and fingers slightly worse.'))
            
            # Use Random Seedの初期値
            use_random_seed = gr.Checkbox(label=translate("Use Random Seed"), value=use_random_seed_default)
            seed = gr.Number(label=translate("Seed"), value=seed_default, precision=0)
            
            # ステップ数などの設定を右カラムに配置
            steps = gr.Slider(label=translate("ステップ数"), minimum=1, maximum=100, value=25, step=1, info=translate('この値の変更は推奨されません'))
            gs = gr.Slider(label=translate("蒸留CFGスケール"), minimum=1.0, maximum=32.0, value=10.0, step=0.01, info=translate('この値の変更は推奨されません'))
            
            # 非表示設定
            cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
            rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
            
            # GPU設定
            gpu_memory_preservation = gr.Slider(
                label=translate("GPUメモリ保持 (GB)"), 
                minimum=6, maximum=128, value=6, step=0.1, 
                info=translate("OOMが発生する場合は値を大きくしてください。値が大きいほど速度が遅くなります。")
            )
            
            # GPUメモリ保持部分のみ残し、詳細設定アコーディオンは削除（すでに移動済み）

            # 出力設定はシード値の下に配置 - endframe_ichiと同じ実装にする
            with gr.Group():
                gr.Markdown(f"### " + translate("出力設定"))
                
                with gr.Row(equal_height=True):
                    with gr.Column(scale=4):
                        # フォルダ名だけを入力欄に設定
                        output_dir = gr.Textbox(
                            label=translate("出力フォルダ名"),
                            value=output_folder_name,  # 設定から読み込んだ値を使用
                            info=translate("生成画像の保存先フォルダ名"),
                            placeholder="outputs"
                        )
                    with gr.Column(scale=1, min_width=100):
                        open_folder_btn = gr.Button(value=translate("📂 保存および出力フォルダを開く"), size="sm")

                # 実際の出力パスを表示
                with gr.Row(visible=False):
                    path_display = gr.Textbox(
                        label=translate("出力フォルダの完全パス"),
                        value=os.path.join(base_path, output_folder_name),
                        interactive=False
                    )
    
    # シードのランダム化機能
    def set_random_seed(is_checked):
        if is_checked:
            return random.randint(0, 2**32 - 1)
        return gr.update()
    
    # チェックボックス変更時にランダムシードを生成
    use_random_seed.change(fn=set_random_seed, inputs=use_random_seed, outputs=seed)
    
    # バッチ処理回数に応じてランダムシードを扱う処理
    def randomize_seed_if_needed(use_random, batch_num=1):
        """バッチ処理用のシード設定関数"""
        if use_random:
            # ランダムシードの場合はバッチごとに異なるシードを生成
            random_seeds = [random.randint(0, 2**32 - 1) for _ in range(batch_num)]
            return random_seeds[0]  # 最初のシードを返す（表示用）
        return gr.update()  # ランダムシードでない場合は何もしない
    
    # プリセット管理に関するイベントハンドラ
    # プリセット保存ボタンのイベント
    def save_button_click_handler(name, prompt_text):
        """保存ボタンクリック時のハンドラ関数"""
        # 重複チェックと正規化
        if "A character" in prompt_text and prompt_text.count("A character") > 1:
            sentences = prompt_text.split(".")
            if len(sentences) > 0:
                prompt_text = sentences[0].strip() + "."
                # 重複を検出したため正規化

        # プリセット保存
        result_msg = save_preset(name, prompt_text)

        # プリセットデータを取得してドロップダウンを更新
        presets_data = load_presets()
        choices = [preset["name"] for preset in presets_data["presets"]]
        default_presets = [n for n in choices if any(p["name"] == n and p.get("is_default", False) for p in presets_data["presets"])]
        user_presets = [n for n in choices if n not in default_presets]
        sorted_choices = [(n, n) for n in sorted(default_presets) + sorted(user_presets)]

        # メインプロンプトは更新しない（保存のみを行う）
        return result_msg, gr.update(choices=sorted_choices), gr.update()

    # クリアボタン処理
    def clear_fields():
        return gr.update(value=""), gr.update(value="")

    # プリセット読込処理
    def load_preset_handler(preset_name):
        # プリセット選択時に編集欄のみを更新
        for preset in load_presets()["presets"]:
            if preset["name"] == preset_name:
                return gr.update(value=preset_name), gr.update(value=preset["prompt"])
        return gr.update(), gr.update()

    # プリセット選択時に編集欄に反映
    def load_preset_handler_wrapper(preset_name):
        # プリセット名がタプルの場合も処理する
        if isinstance(preset_name, tuple) and len(preset_name) == 2:
            preset_name = preset_name[1]  # 値部分を取得
        return load_preset_handler(preset_name)

    # 反映ボタン処理 - 編集画面の内容をメインプロンプトに反映
    def apply_to_prompt(edit_text):
        """編集画面の内容をメインプロンプトに反映する関数"""
        # 編集画面のプロンプトをメインに適用
        return gr.update(value=edit_text)

    # プリセット削除処理
    def delete_preset_handler(preset_name):
        # プリセット名がタプルの場合も処理する
        if isinstance(preset_name, tuple) and len(preset_name) == 2:
            preset_name = preset_name[1]  # 値部分を取得

        result = delete_preset(preset_name)

        # プリセットデータを取得してドロップダウンを更新
        presets_data = load_presets()
        choices = [preset["name"] for preset in presets_data["presets"]]
        default_presets = [name for name in choices if any(p["name"] == name and p.get("is_default", False) for p in presets_data["presets"])]
        user_presets = [name for name in choices if name not in default_presets]
        sorted_names = sorted(default_presets) + sorted(user_presets)
        updated_choices = [(name, name) for name in sorted_names]

        return result, gr.update(choices=updated_choices)
    
    # 保存ボタンのクリックイベントを接続
    save_btn.click(
        fn=save_button_click_handler,
        inputs=[edit_name, edit_prompt],
        outputs=[result_message, preset_dropdown, prompt]
    )
    
    # クリアボタンのクリックイベントを接続
    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[edit_name, edit_prompt]
    )
    
    # プリセット選択時のイベントを接続
    preset_dropdown.change(
        fn=load_preset_handler_wrapper,
        inputs=[preset_dropdown],
        outputs=[edit_name, edit_prompt]
    )
    
    # 反映ボタンのクリックイベントを接続
    apply_preset_btn.click(
        fn=apply_to_prompt,
        inputs=[edit_prompt],
        outputs=[prompt]
    )
    
    # 削除ボタンのクリックイベントを接続
    delete_preset_btn.click(
        fn=delete_preset_handler,
        inputs=[preset_dropdown],
        outputs=[result_message, preset_dropdown]
    )
    
    # 画像変更時にメタデータを抽出するイベント設定
    input_image.change(
        fn=update_from_image_metadata,
        inputs=[input_image, copy_metadata],
        outputs=[prompt, seed]
    )
    
    # チェックボックス変更時にメタデータを抽出するイベント設定
    copy_metadata.change(
        fn=check_metadata_on_checkbox_change,
        inputs=[copy_metadata, input_image],
        outputs=[prompt, seed]
    )
    
    # フォルダを開くボタンのイベント
    open_folder_btn.click(
        fn=handle_open_folder_btn,
        inputs=[output_dir],
        outputs=[output_dir, path_display]
    )
    
    # 生成開始・中止のイベント
    ips = [input_image, prompt, n_prompt, seed, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache,
           lora_files, lora_files2, lora_scales_text, use_lora, fp8_optimization, resolution, output_dir,
           batch_count, use_random_seed, latent_window_size, latent_index,
           use_clean_latents_2x, use_clean_latents_4x, use_clean_latents_post,
           lora_mode, lora_dropdown1, lora_dropdown2, lora_dropdown3, lora_files3, use_rope_batch,
           use_queue, prompt_queue_file]  # キュー機能パラメータを追加
    start_button.click(fn=process, inputs=ips, outputs=[result_image, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process, outputs=[end_button])
    
    gr.HTML(f'<div style="text-align:center; margin-top:20px;">{translate("FramePack 単一フレーム生成版")}</div>')

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
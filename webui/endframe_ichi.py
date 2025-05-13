import os
import sys
sys.path.append(os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './submodules/FramePack'))))

from diffusers_helper.hf_login import login

# VAEキャッシュ機能のインポート
from eichi_utils.vae_cache import vae_decode_cache

# グローバル変数の設定
vae_cache_enabled = False  # VAEキャッシュのチェックボックス状態を保持
current_prompt = None      # キューから読み込まれた現在のプロンプト

import os
import random
import time
import subprocess
import traceback  # デバッグログ出力用
# クロスプラットフォーム対応のための条件付きインポート
import yaml
import zipfile

import argparse

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
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--lang", type=str, default='ja', help="Language: ja, zh-tw, en")
args = parser.parse_args()

# Load translations from JSON files
from locales.i18n_extended import (set_lang, translate)
set_lang(args.lang)

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False
import json
import traceback
from datetime import datetime, timedelta

if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
    print(translate("HF_HOMEを設定: {0}").format(os.environ['HF_HOME']))
else:
    print(translate("既存のHF_HOMEを使用: {0}").format(os.environ['HF_HOME']))

temp_dir = "./temp_for_zip_section_info"

# LoRAサポートの確認
has_lora_support = False
try:
    import lora_utils
    has_lora_support = True
    print(translate("LoRAサポートが有効です"))
except ImportError:
    print(translate("LoRAサポートが無効です（lora_utilsモジュールがインストールされていません）"))

# 設定モジュールをインポート（ローカルモジュール）
import os.path
from eichi_utils.video_mode_settings import (
    VIDEO_MODE_SETTINGS, get_video_modes, get_video_seconds, get_important_keyframes,
    get_copy_targets, get_max_keyframes_count, get_total_sections, generate_keyframe_guide_html,
    handle_mode_length_change, process_keyframe_change, MODE_TYPE_NORMAL, MODE_TYPE_LOOP
)

# 設定管理モジュールをインポート
from eichi_utils.settings_manager import (
    get_settings_file_path,
    get_output_folder_path,
    initialize_settings,
    load_settings,
    save_settings,
    open_output_folder
)

# プリセット管理モジュールをインポート
from eichi_utils.preset_manager import (
    initialize_presets,
    load_presets,
    get_default_startup_prompt,
    save_preset,
    delete_preset
)

# キーフレーム処理モジュールをインポート
from eichi_utils.keyframe_handler import (
    ui_to_code_index,
    code_to_ui_index,
    unified_keyframe_change_handler,
    unified_input_image_change_handler,
    print_keyframe_debug_info
)

# セクション情報の一括管理モジュールをインポート
from eichi_utils.section_manager import upload_zipfile_handler, download_zipfile_handler

# 拡張キーフレーム処理モジュールをインポート
from eichi_utils.keyframe_handler_extended import extended_mode_length_change_handler
import gradio as gr
# UI関連モジュールのインポート
from eichi_utils.ui_styles import get_app_css
import torch
import einops
import safetensors.torch as sf
import numpy as np
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_encode, vae_decode_fake, vae_decode
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
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

# グローバルなモデル状態管理インスタンスを作成
# 通常モードではuse_f1_model=Falseを指定（デフォルト値なので省略可）
transformer_manager = TransformerManager(device=gpu, high_vram_mode=high_vram, use_f1_model=False)
text_encoder_manager = TextEncoderManager(device=gpu, high_vram_mode=high_vram)

try:
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

    # text_encoderとtext_encoder_2の初期化
    if not text_encoder_manager.ensure_text_encoder_state():
        raise Exception(translate("text_encoderとtext_encoder_2の初期化に失敗しました"))
    text_encoder, text_encoder_2 = text_encoder_manager.get_text_encoders()

    # transformerの初期化
    transformer_manager.ensure_download_models()
    transformer = transformer_manager.get_transformer()  # 仮想デバイス上のtransformerを取得

    # 他のモデルの読み込み
    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
except Exception as e:
    print(translate("モデル読み込みエラー: {0}").format(e))
    print(translate("プログラムを終了します..."))
    import sys
    sys.exit(1)

vae.eval()
image_encoder.eval()

# VAE設定を適用（カスタム設定またはデフォルト設定）
from eichi_utils import apply_vae_settings, load_vae_settings

# VAE設定を適用
vae = apply_vae_settings(vae)

# 低VRAMモードでカスタム設定が無効な場合はデフォルトの設定を適用
vae_settings = load_vae_settings()
if not high_vram and not vae_settings.get('custom_vae_settings', False):
    vae.enable_slicing()
    vae.enable_tiling()

vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)

vae.requires_grad_(False)
image_encoder.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu) # クラスを操作するので仮想デバイス上のtransformerでもOK
else:
    image_encoder.to(gpu)
    vae.to(gpu)

# グローバル変数
batch_stopped = False  # バッチ処理の停止フラグ
queue_enabled = False  # キュー機能の有効/無効フラグ
queue_type = "prompt"  # キューのタイプ（"prompt" または "image"）
prompt_queue_file_path = None  # プロンプトキューファイルのパス
vae_cache_enabled = False  # VAEキャッシュの有効/無効フラグ
image_queue_files = []  # イメージキューのファイルリスト
input_folder_name_value = "inputs"  # 入力フォルダ名（デフォルト値）

# イメージキューのための画像ファイルリストを取得する関数（グローバル関数）
def get_image_queue_files():
    global image_queue_files, input_folder_name_value
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_folder_name_value)

    # 入力ディレクトリが存在するかチェック（ボタン押下時のみ作成するため、ここでは作成しない）
    if not os.path.exists(input_dir):
        print(translate("入力ディレクトリが存在しません: {0}（保存及び入力フォルダを開くボタンを押すと作成されます）").format(input_dir))
        return []

    # 画像ファイル（png, jpg, jpeg）のみをリスト
    image_files = []
    for file in sorted(os.listdir(input_dir)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(input_dir, file))

    print(translate("入力ディレクトリから画像ファイル{0}個を読み込みました").format(len(image_files)))
    image_queue_files = image_files
    return image_files

stream = AsyncStream()

# 設定管理モジュールをインポート
from eichi_utils.settings_manager import (
    get_settings_file_path,
    get_output_folder_path,
    initialize_settings,
    load_settings,
    save_settings,
    open_output_folder
)

# フォルダ構造を先に定義
webui_folder = os.path.dirname(os.path.abspath(__file__))

# 設定保存用フォルダの設定
settings_folder = os.path.join(webui_folder, 'settings')
os.makedirs(settings_folder, exist_ok=True)

# 設定ファイル初期化
initialize_settings()

# ベースパスを定義
base_path = os.path.dirname(os.path.abspath(__file__))

# 設定から出力フォルダを取得
app_settings = load_settings()
output_folder_name = app_settings.get('output_folder', 'outputs')
print(translate("設定から出力フォルダを読み込み: {0}").format(output_folder_name))

# 入力フォルダ名も設定から取得
input_folder_name_value = app_settings.get('input_folder', 'inputs')
print(translate("設定から入力フォルダを読み込み: {0}").format(input_folder_name_value))

# 出力フォルダのフルパスを生成
outputs_folder = get_output_folder_path(output_folder_name)
os.makedirs(outputs_folder, exist_ok=True)

# 入力フォルダも存在確認して作成
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_folder_name_value)
os.makedirs(input_dir, exist_ok=True)

# キーフレーム処理関数は keyframe_handler.py に移動済み

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf=16, all_padding_value=1.0, end_frame=None, end_frame_strength=1.0, keep_section_videos=False, lora_files=None, lora_files2=None, lora_files3=None, lora_scales_text="0.8,0.8,0.8", output_dir=None, save_section_frames=False, section_settings=None, use_all_padding=False, use_lora=False, lora_mode=None, lora_dropdown1=None, lora_dropdown2=None, lora_dropdown3=None, save_tensor_data=False, tensor_data_input=None, fp8_optimization=False, resolution=640, batch_index=None, save_latent_frames=False, save_last_section_frames=False, use_vae_cache=False, use_queue=False, prompt_queue_file=None):
    # グローバル変数を使用
    global vae_cache_enabled, current_prompt
    # パラメータ経由の値とグローバル変数の値を確認
    print(f"worker関数でのVAEキャッシュ設定: パラメータ={use_vae_cache}, グローバル変数={vae_cache_enabled}")

    # キュー処理
    current_prompt = prompt
    current_image = input_image

    # キュー状態のチェック - バッチ処理開始時に保存された値を使用
    # これにより、バッチ処理中のUIでの変更の影響を受けない
    use_queue_flag = bool(use_queue)  # パラメータから直接使用
    queue_type_flag = queue_type  # グローバル変数を使用

    print(f"worker: キュー状態: {use_queue_flag}, タイプ: {queue_type_flag}")

    # パラメータも診断のため確認
    if isinstance(use_queue, bool):
        param_queue = use_queue
        print(f"worker: パラメータ値(ブール型): {param_queue}")
    elif hasattr(use_queue, 'value'):
        param_queue = bool(use_queue.value)
        print(f"worker: パラメータ値(Gradioオブジェクト): {param_queue}")
    else:
        print(f"worker: パラメータ値が無効な形式: {type(use_queue).__name__}")

    # キュー処理の開始
    if use_queue_flag:
        # キュータイプに応じた処理
        if queue_type_flag == "prompt":
            # プロンプトキュー処理
            print(f"worker: プロンプトキューを使用")

            # パラメータからファイルパス情報を確認
            print(f"worker: prompt_queue_file パラメータの型: {type(prompt_queue_file).__name__}")
            if hasattr(prompt_queue_file, 'name'):
                print(f"worker: 【ファイル診断】prompt_queue_file.name: {prompt_queue_file.name}")

            # バッチ処理開始時に保存された値を使用
            if prompt_queue_file_path is not None:
                queue_file_path = prompt_queue_file_path
                print(f"worker: プロンプトキューファイルを使用: {queue_file_path}")

                try:
                    # ファイルが存在し、読み込める場合
                    if queue_file_path and os.path.exists(queue_file_path):
                        print(f"プロンプトキューファイルを読み込みます: {queue_file_path}")

                        with open(queue_file_path, 'r', encoding='utf-8') as f:
                            # 空行を除去して整形
                            prompt_lines = [line.strip() for line in f.readlines() if line.strip()]
                            print(f"読み込まれたプロンプト行数: {len(prompt_lines)}")

                        # バッチインデックスに応じたプロンプトを選択
                        if prompt_lines and batch_index is not None and 0 <= batch_index < len(prompt_lines):
                            current_prompt = prompt_lines[batch_index]
                            print(f"バッチ {batch_index + 1}/{len(prompt_lines)} のプロンプト: {current_prompt[:50]}...")
                            print(translate("◆ プロンプトキュー行を優先: {0}/{1} 行目を処理中").format(batch_index + 1, len(prompt_lines)))
                        else:
                            # バッチインデックスが不正な場合
                            if batch_index is None:
                                print("バッチインデックスが未指定です")
                            elif batch_index < 0 or (prompt_lines and batch_index >= len(prompt_lines)):
                                print(f"バッチインデックスが範囲外です: {batch_index}, プロンプト数: {len(prompt_lines) if prompt_lines else 0}")
                            elif not prompt_lines:
                                print("プロンプトファイルに有効な行がありません")

                            print(f"元のプロンプトを使用します: {prompt[:50]}...")
                    else:
                        print(f"プロンプトキューファイルが存在しないか無効です: {queue_file_path}")
                except Exception as e:
                    print(f"プロンプトキュー処理エラー: {str(e)}")
                    import traceback
                    traceback.print_exc()

        elif queue_type_flag == "image":
            # イメージキュー処理
            print(f"worker: イメージキューを使用")

            # batch_indexが0（最初の実行）なら入力画像を使用
            if batch_index == 0 or batch_index is None:
                print(f"イメージキュー: 最初の実行のため入力画像を使用")
                # 現在の入力画像をそのまま使用
                current_image = input_image
            else:
                # 2回目以降はinputsフォルダの画像を使用
                global image_queue_files

                # 画像ファイルリストを取得（初回または未取得の場合）
                if not image_queue_files:
                    get_image_queue_files()

                # イメージキューの詳細ログ出力（デバッグ用）
                print(f"worker: イメージキュー詳細: 画像数={len(image_queue_files)}, batch_index={batch_index}")
                for i, img in enumerate(image_queue_files):
                    print(f"    worker: 画像{i+1}: {os.path.basename(img)}")

                # 入力画像 + 画像キュー
                if image_queue_files and batch_index - 1 < len(image_queue_files):
                    # 画像ファイルがある場合はそれを使用（batch_index-1の理由は1回目は入力画像を使うため）
                    queue_index = batch_index - 1
                    img_path = image_queue_files[queue_index]
                    print(f"イメージキュー: {batch_index}回目の実行、画像ファイル使用: {os.path.basename(img_path)} (インデックス: {queue_index})")
                    current_image = img_path
                else:
                    # 画像ファイルが足りない場合は入力画像に戻る
                    print(f"イメージキュー: 画像ファイルが足りないため入力画像を再使用")
                    current_image = input_image
    # グローバル変数の値を優先
    use_vae_cache = vae_cache_enabled or use_vae_cache

    # フレーム保存フラグのタイプと値を確認（必ずブール値であるべき）
    print(translate("[DEBUG] worker関数に渡されたフラグ - save_latent_frames型: {0}, 値: {1}").format(type(save_latent_frames).__name__, save_latent_frames))
    print(translate("[DEBUG] worker関数に渡されたフラグ - save_last_section_frames型: {0}, 値: {1}").format(type(save_last_section_frames).__name__, save_last_section_frames))
    
    # 万が一文字列が渡された場合の防御コード
    if isinstance(save_latent_frames, str):
        # 文字列の場合は、条件判定して適切なブール値に変換
        if save_latent_frames == translate("全フレーム画像保存"):
            save_latent_frames = True
        else:
            save_latent_frames = False
        print(translate("[WARN] save_latent_framesが文字列でした。ブール値に変換: {0}").format(save_latent_frames))
    
    if isinstance(save_last_section_frames, str):
        # 文字列の場合は、条件判定して適切なブール値に変換
        # 注意: UIでは「最終セクションのみ全フレーム画像保存」という表記を使っている
        if save_last_section_frames == translate("最終セクションのみフレーム画像保存") or save_last_section_frames == translate("最終セクションのみ全フレーム画像保存"):
            save_last_section_frames = True
        else:
            save_last_section_frames = False
        print(translate("[WARN] save_last_section_framesが文字列でした。ブール値に変換: {0}").format(save_last_section_frames))
    
    # 最終的に必ずブール型に変換しておく
    save_latent_frames = bool(save_latent_frames)
    save_last_section_frames = bool(save_last_section_frames)
    
    print(translate("[DEBUG] 最終変換後のフラグ - save_latent_frames: {0}, save_last_section_frames: {1}").format(save_latent_frames, save_last_section_frames))

    # 入力画像または表示されている最後のキーフレーム画像のいずれかが存在するか確認
    print(translate("[DEBUG] worker内 input_imageの型: {0}").format(type(input_image)))
    if isinstance(input_image, str):
        print(translate("[DEBUG] input_imageはファイルパスです: {0}").format(input_image))
        has_any_image = (input_image is not None)
    else:
        print(translate("[DEBUG] input_imageはファイルパス以外です"))
        has_any_image = (input_image is not None)
    last_visible_section_image = None
    last_visible_section_num = -1

    if not has_any_image and section_settings is not None:
        # 現在の動画長設定から表示されるセクション数を計算
        total_display_sections = None
        try:
            # 動画長を秒数で取得
            seconds = get_video_seconds(total_second_length)

            # フレームサイズ設定からlatent_window_sizeを計算
            current_latent_window_size = 4.5 if frame_size_setting == "0.5秒 (17フレーム)" else 9
            frame_count = current_latent_window_size * 4 - 3

            # セクション数を計算
            total_frames = int(seconds * 30)
            total_display_sections = int(max(round(total_frames / frame_count), 1))
            print(translate("[DEBUG] worker内の現在の設定によるセクション数: {0}").format(total_display_sections))
        except Exception as e:
            print(translate("[ERROR] worker内のセクション数計算エラー: {0}").format(e))

        # 有効なセクション番号を収集
        valid_sections = []
        for section in section_settings:
            if section and len(section) > 1 and section[0] is not None and section[1] is not None:
                try:
                    section_num = int(section[0])
                    # 表示セクション数が計算されていれば、それ以下のセクションのみ追加
                    if total_display_sections is None or section_num < total_display_sections:
                        valid_sections.append((section_num, section[1]))
                except (ValueError, TypeError):
                    continue

        # 有効なセクションがあれば、最大の番号（最後のセクション）を探す
        if valid_sections:
            # 番号でソート
            valid_sections.sort(key=lambda x: x[0])
            # 最後のセクションを取得
            last_visible_section_num, last_visible_section_image = valid_sections[-1]
            print(translate("[DEBUG] worker内の最後のキーフレーム確認: セクション{0} (画像あり)").format(last_visible_section_num))

    has_any_image = has_any_image or (last_visible_section_image is not None)
    if not has_any_image:
        # UIに直接エラーメッセージを表示
        stream.output_queue.push(('progress', (None, translate("❗️ 画像が選択されていません\n生成を開始する前に「Image」欄または表示されている最後のキーフレーム画像に画像をアップロードしてください。これはあまねく叡智の始発点となる重要な画像です。"), make_progress_bar_html(0, translate('エラー')))))
        # 処理を終了
        stream.output_queue.push(('end', None))
        return

    # 入力画像がない場合はキーフレーム画像を使用
    if input_image is None and last_visible_section_image is not None:
        print(translate("[INFO] 入力画像が指定されていないため、セクション{0}のキーフレーム画像を使用します").format(last_visible_section_num))
        input_image = last_visible_section_image

    # 出力フォルダの設定
    global outputs_folder
    global output_folder_name
    if output_dir and output_dir.strip():
        # 出力フォルダパスを取得
        outputs_folder = get_output_folder_path(output_dir)
        print(translate("出力フォルダを設定: {0}").format(outputs_folder))

        # フォルダ名が現在の設定と異なる場合は設定ファイルを更新
        if output_dir != output_folder_name:
            settings = load_settings()
            settings['output_folder'] = output_dir
            if save_settings(settings):
                output_folder_name = output_dir
                print(translate("出力フォルダ設定を保存しました: {0}").format(output_dir))
    else:
        # デフォルト設定を使用
        outputs_folder = get_output_folder_path(output_folder_name)
        print(translate("デフォルト出力フォルダを使用: {0}").format(outputs_folder))

    # フォルダが存在しない場合は作成
    os.makedirs(outputs_folder, exist_ok=True)

    # 処理時間計測の開始
    process_start_time = time.time()

    # グローバル変数で状態管理しているモデル変数を宣言する
    global transformer, text_encoder, text_encoder_2

    # text_encoderとtext_encoder_2を確実にロード
    if not text_encoder_manager.ensure_text_encoder_state():
        raise Exception(translate("text_encoderとtext_encoder_2の初期化に失敗しました"))
    text_encoder, text_encoder_2 = text_encoder_manager.get_text_encoders()

    # 既存の計算方法を保持しつつ、設定からセクション数も取得する
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    # 現在のモードを取得（UIから渡された情報から）
    # セクション数を全セクション数として保存
    total_sections = total_latent_sections

    # 現在のバッチ番号が指定されていれば使用する
    batch_suffix = f"_batch{batch_index+1}" if batch_index is not None else ""
    job_id = generate_timestamp() + batch_suffix

    # セクション处理の詳細ログを出力
    if use_all_padding:
        # オールパディングが有効な場合、すべてのセクションで同じ値を使用
        padding_value = round(all_padding_value, 1)  # 小数点1桁に固定（小数点対応）
        latent_paddings = [padding_value] * total_latent_sections
        print(translate("オールパディングを有効化: すべてのセクションにパディング値 {0} を適用").format(padding_value))
    else:
        # 通常のパディング値計算
        latent_paddings = reversed(range(total_latent_sections))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

    # 全セクション数を事前に計算して保存（イテレータの消費を防ぐため）
    latent_paddings_list = list(latent_paddings)
    total_sections = len(latent_paddings_list)
    latent_paddings = latent_paddings_list  # リストに変換したものを使用

    print(translate("\u25a0 セクション生成詳細:"))
    print(translate("  - 生成予定セクション: {0}").format(latent_paddings))
    frame_count = latent_window_size * 4 - 3
    print(translate("  - 各セクションのフレーム数: 約{0}フレーム (latent_window_size: {1})").format(frame_count, latent_window_size))
    print(translate("  - 合計セクション数: {0}").format(total_sections))

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # セクション設定の前処理
        def get_section_settings_map(section_settings):
            """
            section_settings: DataFrame形式のリスト [[番号, 画像, プロンプト], ...]
            → {セクション番号: (画像, プロンプト)} のdict
            プロンプトやセクション番号のみの設定も許可する
            """
            result = {}
            if section_settings is not None:
                for row in section_settings:
                    if row and len(row) > 0 and row[0] is not None:
                        # セクション番号を取得
                        sec_num = int(row[0])

                        # セクションプロンプトを取得
                        prm = row[2] if len(row) > 2 and row[2] is not None else ""

                        # 画像を取得（ない場合はNone）
                        img = row[1] if len(row) > 1 and row[1] is not None else None

                        # プロンプトまたは画像のどちらかがあればマップに追加
                        if img is not None or (prm is not None and prm.strip() != ""):
                            result[sec_num] = (img, prm)
            return result

        section_map = get_section_settings_map(section_settings)
        section_numbers_sorted = sorted(section_map.keys()) if section_map else []

        def get_section_info(i_section):
            """
            i_section: int
            section_map: {セクション番号: (画像, プロンプト)}
            指定がなければ次のセクション、なければNone
            """
            if not section_map:
                return None, None, None
            # i_section以降で最初に見つかる設定
            for sec in range(i_section, max(section_numbers_sorted)+1):
                if sec in section_map:
                    img, prm = section_map[sec]
                    return sec, img, prm
            return None, None, None

        # セクション固有のプロンプト処理を行う関数
        def process_section_prompt(i_section, section_map, llama_vec, clip_l_pooler, llama_attention_mask, embeddings_cache=None):
            """セクションに固有のプロンプトがあればエンコードまたはキャッシュから取得して返す
            なければメインプロンプトのエンコード結果を返す
            返り値: (llama_vec, clip_l_pooler, llama_attention_mask)
            """
            if not isinstance(llama_vec, torch.Tensor) or not isinstance(llama_attention_mask, torch.Tensor):
                print(translate("[ERROR] メインプロンプトのエンコード結果またはマスクが不正です"))
                return llama_vec, clip_l_pooler, llama_attention_mask

            # embeddings_cacheがNoneの場合は空の辞書で初期化
            embeddings_cache = embeddings_cache or {}

            # セクション固有のプロンプトがあるか確認
            section_info = None
            section_num = None
            
            # セクション固有のプロンプトをチェック - 各セクションのプロンプトはそのセクションでのみ有効
            if section_map and i_section in section_map:
                section_num = i_section
                section_info = section_map[section_num]

            # セクション固有のプロンプトがあれば使用（キーフレーム画像の有無に関わらず）
            if section_info:
                img, section_prompt = section_info
                if section_prompt and section_prompt.strip():
                    # 事前にエンコードされたプロンプト埋め込みをキャッシュから取得
                    if section_num in embeddings_cache:
                        print(translate("[section_prompt] セクション{0}の専用プロンプトをキャッシュから取得: {1}...").format(i_section, section_prompt[:30]))
                        # キャッシュからデータを取得
                        cached_llama_vec, cached_clip_l_pooler, cached_llama_attention_mask = embeddings_cache[section_num]

                        # データ型を明示的にメインプロンプトと合わせる（2回目のチェック）
                        cached_llama_vec = cached_llama_vec.to(dtype=llama_vec.dtype, device=llama_vec.device)
                        cached_clip_l_pooler = cached_clip_l_pooler.to(dtype=clip_l_pooler.dtype, device=clip_l_pooler.device)
                        cached_llama_attention_mask = cached_llama_attention_mask.to(dtype=llama_attention_mask.dtype, device=llama_attention_mask.device)

                        return cached_llama_vec, cached_clip_l_pooler, cached_llama_attention_mask

                    print(translate("[section_prompt] セクション{0}の専用プロンプトを処理: {1}...").format(i_section, section_prompt[:30]))

                    try:
                        # プロンプト処理
                        section_llama_vec, section_clip_l_pooler = encode_prompt_conds(
                            section_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
                        )

                        # マスクの作成
                        section_llama_vec, section_llama_attention_mask = crop_or_pad_yield_mask(
                            section_llama_vec, length=512
                        )

                        # データ型を明示的にメインプロンプトと合わせる
                        section_llama_vec = section_llama_vec.to(
                            dtype=llama_vec.dtype, device=llama_vec.device
                        )
                        section_clip_l_pooler = section_clip_l_pooler.to(
                            dtype=clip_l_pooler.dtype, device=clip_l_pooler.device
                        )
                        section_llama_attention_mask = section_llama_attention_mask.to(
                            device=llama_attention_mask.device
                        )

                        return section_llama_vec, section_clip_l_pooler, section_llama_attention_mask
                    except Exception as e:
                        print(translate("[ERROR] セクションプロンプト処理エラー: {0}").format(e))

            # 共通プロンプトを使用
            # プロンプトキューを使用している場合は、その情報も表示
            global current_prompt
            if 'current_prompt' in globals() and current_prompt != prompt:
                print(translate("[section_prompt] セクション{0}は共通プロンプトを使用します（プロンプトキュー: {1}...）").format(i_section, current_prompt[:30]))
            else:
                print(translate("[section_prompt] セクション{0}は共通プロンプトを使用します").format(i_section))
            return llama_vec, clip_l_pooler, llama_attention_mask


        # Clean GPU
        if not high_vram:
            # モデルをCPUにアンロード
            unload_complete_models(
                image_encoder, vae
            )

        # Text encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Text encoding ...")))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        # プロンプトキューから選択されたプロンプトを使用
        # フラグが設定されていなくてもcurrent_promptを使うことで、
        # バッチ処理中で既にプロンプトが上書きされていた場合でも対応
        llama_vec, clip_l_pooler = encode_prompt_conds(current_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # セクションプロンプトを事前にエンコードしておく
        section_prompt_embeddings = {}
        if section_map:
            print(translate("セクションプロンプトを事前にエンコードしています..."))
            for sec_num, (_, sec_prompt) in section_map.items():
                if sec_prompt and sec_prompt.strip():
                    try:
                        # セクションプロンプトをエンコード
                        print(translate("[section_prompt] セクション{0}の専用プロンプトを事前エンコード: {1}...").format(sec_num, sec_prompt[:30]))
                        sec_llama_vec, sec_clip_l_pooler = encode_prompt_conds(sec_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                        sec_llama_vec, sec_llama_attention_mask = crop_or_pad_yield_mask(sec_llama_vec, length=512)

                        # データ型を明示的にメインプロンプトと合わせる
                        sec_llama_vec = sec_llama_vec.to(dtype=llama_vec.dtype, device=llama_vec.device)
                        sec_clip_l_pooler = sec_clip_l_pooler.to(dtype=clip_l_pooler.dtype, device=clip_l_pooler.device)
                        sec_llama_attention_mask = sec_llama_attention_mask.to(dtype=llama_attention_mask.dtype, device=llama_attention_mask.device)

                        # 結果を保存
                        section_prompt_embeddings[sec_num] = (sec_llama_vec, sec_clip_l_pooler, sec_llama_attention_mask)
                        print(translate("[section_prompt] セクション{0}のプロンプトエンコード完了").format(sec_num))
                    except Exception as e:
                        print(translate("[ERROR] セクション{0}のプロンプトエンコードに失敗: {1}").format(sec_num, e))
                        traceback.print_exc()

        # これ以降の処理は text_encoder, text_encoder_2 は不要なので、メモリ解放してしまって構わない
        if not high_vram:
            text_encoder, text_encoder_2 = None, None
            text_encoder_manager.dispose_text_encoders()

        # テンソルデータのアップロードがあれば読み込み
        uploaded_tensor = None
        if tensor_data_input is not None:
            try:
                tensor_path = tensor_data_input.name
                print(translate("テンソルデータを読み込み: {0}").format(os.path.basename(tensor_path)))
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate('Loading tensor data ...')))))

                # safetensorsからテンソルを読み込み
                tensor_dict = sf.load_file(tensor_path)

                # テンソルに含まれているキーとシェイプを確認
                print(translate("テンソルデータの内容:"))
                for key, tensor in tensor_dict.items():
                    print(f"  - {key}: shape={tensor.shape}, dtype={tensor.dtype}")

                # history_latentsと呼ばれるキーが存在するか確認
                if "history_latents" in tensor_dict:
                    uploaded_tensor = tensor_dict["history_latents"]

                    # 削除するフレームサイズを計算
                    uploaded_tensor_latent_size = uploaded_tensor.shape[2]
                    trim_start_latent_size = tensor_trim_start_latents.value

                    if uploaded_tensor_latent_size > trim_start_latent_size:
                        # テンソルデータの先頭フレームを削除
                        if trim_start_latent_size > 0:
                            uploaded_tensor = uploaded_tensor[:, :, trim_start_latent_size:, :, :].clone()
                            print(translate("アップロードされたテンソルデータの先頭フレームを削除しました。削除数: {0}/{1}").format(trim_start_latent_size, uploaded_tensor_latent_size))
                    else:
                        print(translate("警告: テンソルデータのフレーム数よりも、先頭フレーム削除数が大きく指定されているため、先頭フレーム削除は実施しません。"))

                    print(translate("テンソルデータ読み込み成功: shape={0}, dtype={1}").format(uploaded_tensor.shape, uploaded_tensor.dtype))
                    stream.output_queue.push(('progress', (None, translate('Tensor data loaded successfully!'), make_progress_bar_html(10, translate('Tensor data loaded successfully!')))))
                else:
                    print(translate("警告: テンソルデータに 'history_latents' キーが見つかりません"))
            except Exception as e:
                print(translate("テンソルデータ読み込みエラー: {0}").format(e))
                traceback.print_exc()

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Image processing ...")))))

        def preprocess_image(img_path_or_array, resolution=640):
            """Pathまたは画像配列を処理して適切なサイズに変換する"""
            print(translate("[DEBUG] preprocess_image: img_path_or_array型 = {0}").format(type(img_path_or_array)))

            if img_path_or_array is None:
                # 画像がない場合は指定解像度の黒い画像を生成
                img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
                height = width = resolution
                return img, img, height, width

            # TensorからNumPyへ変換する必要があれば行う
            if isinstance(img_path_or_array, torch.Tensor):
                img_path_or_array = img_path_or_array.cpu().numpy()

            # Pathの場合はPILで画像を開く
            if isinstance(img_path_or_array, str) and os.path.exists(img_path_or_array):
                # print(translate("[DEBUG] ファイルから画像を読み込み: {0}").format(img_path_or_array))
                img = np.array(Image.open(img_path_or_array).convert('RGB'))
            else:
                # NumPy配列の場合はそのまま使う
                img = img_path_or_array

            H, W, C = img.shape
            # 解像度パラメータを使用してサイズを決定
            height, width = find_nearest_bucket(H, W, resolution=resolution)
            img_np = resize_and_center_crop(img, target_width=width, target_height=height)
            img_pt = torch.from_numpy(img_np).float() / 127.5 - 1
            img_pt = img_pt.permute(2, 0, 1)[None, :, None]
            return img_np, img_pt, height, width

        input_image_np, input_image_pt, height, width = preprocess_image(current_image, resolution=resolution)
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))
        # 入力画像にメタデータを埋め込んで保存
        initial_image_path = os.path.join(outputs_folder, f'{job_id}.png')
        Image.fromarray(input_image_np).save(initial_image_path)

        # メタデータの埋め込み
        # print(translate("\n[DEBUG] 入力画像へのメタデータ埋め込み開始: {0}").format(initial_image_path))
        # print(f"[DEBUG] prompt: {prompt}")
        # print(f"[DEBUG] seed: {seed}")
        metadata = {
            PROMPT_KEY: prompt,
            SEED_KEY: seed
        }
        # print(translate("[DEBUG] 埋め込むメタデータ: {0}").format(metadata))
        embed_metadata_to_png(initial_image_path, metadata)

        # VAE encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("VAE encoding ...")))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        # アップロードされたテンソルがあっても、常に入力画像から通常のエンコーディングを行う
        # テンソルデータは後で後付けとして使用するために保持しておく
        if uploaded_tensor is not None:
            print(translate("アップロードされたテンソルデータを検出: 動画生成後に後方に結合します"))
            # 入力画像がNoneの場合、テンソルからデコードして表示画像を生成
            if input_image is None:
                try:
                    # テンソルの最初のフレームから画像をデコードして表示用に使用
                    preview_latent = uploaded_tensor[:, :, 0:1, :, :].clone()
                    if preview_latent.device != torch.device('cpu'):
                        preview_latent = preview_latent.cpu()
                    if preview_latent.dtype != torch.float16:
                        preview_latent = preview_latent.to(dtype=torch.float16)

                    # VAEキャッシュ設定に応じてデコード関数を切り替え
                    if use_vae_cache:
                        print(translate("[INFO] VAEキャッシュを使用: プレビュー画像"))
                        decoded_image = vae_decode_cache(preview_latent, vae)
                    else:
                        decoded_image = vae_decode(preview_latent, vae)
                    decoded_image = (decoded_image[0, :, 0] * 127.5 + 127.5).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
                    # デコードした画像を保存
                    Image.fromarray(decoded_image).save(os.path.join(outputs_folder, f'{job_id}_tensor_preview.png'))
                    # デコードした画像を入力画像として設定
                    input_image = decoded_image
                    # 前処理用のデータも生成
                    input_image_np, input_image_pt, height, width = preprocess_image(current_image)
                    print(translate("テンソルからデコードした画像を生成しました: {0}x{1}").format(height, width))
                except Exception as e:
                    print(translate("テンソルからのデコード中にエラーが発生しました: {0}").format(e))
                    # デコードに失敗した場合は通常の処理を続行

            # UI上でテンソルデータの情報を表示
            tensor_info = translate("テンソルデータ ({0}フレーム) を検出しました。動画生成後に後方に結合します。").format(uploaded_tensor.shape[2])
            stream.output_queue.push(('progress', (None, tensor_info, make_progress_bar_html(10, translate('テンソルデータを後方に結合')))))

        # 常に入力画像から通常のエンコーディングを行う
        start_latent = vae_encode(input_image_pt, vae)
        # end_frameも同じタイミングでencode
        if end_frame is not None:
            end_frame_np, end_frame_pt, _, _ = preprocess_image(end_frame, resolution=resolution)
            end_frame_latent = vae_encode(end_frame_pt, vae)
        else:
            end_frame_latent = None

        # create section_latents here
        section_latents = None
        if section_map:
            section_latents = {}
            for sec_num, (img, prm) in section_map.items():
                if img is not None:
                    # 画像をVAE encode
                    img_np, img_pt, _, _ = preprocess_image(img, resolution=resolution)
                    section_latents[sec_num] = vae_encode(img_pt, vae)

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("CLIP Vision encoding ...")))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Start sampling ...")))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        # latent_window_sizeが4.5の場合は特別に17フレームとする
        if latent_window_size == 4.5:
            num_frames = 17  # 5 * 4 - 3 = 17
        else:
            num_frames = int(latent_window_size * 4 - 3)

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        # ここでlatent_paddingsを再定義していたのが原因だったため、再定義を削除します

        # -------- LoRA 設定 START ---------

        # UI設定のuse_loraフラグ値を保存
        original_use_lora = use_lora
        print(f"[DEBUG] UI設定のuse_loraフラグの値: {original_use_lora}")

        # LoRAの環境変数設定（PYTORCH_CUDA_ALLOC_CONF）
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            old_env = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            print(translate("CUDA環境変数設定: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (元の値: {0})").format(old_env))

        # 次回のtransformer設定を更新
        current_lora_paths = []
        current_lora_scales = []
        
        # ディレクトリモードのLoRA選択がある場合は強制的に有効にする
        if lora_mode == translate("ディレクトリから選択") and has_lora_support:
            # ディレクトリからドロップダウンで選択されたLoRAが1つでもあるか確認
            has_selected_lora = False
            for dropdown in [lora_dropdown1, lora_dropdown2, lora_dropdown3]:
                dropdown_value = dropdown.value if hasattr(dropdown, 'value') else dropdown
                
                # 通常の値が0や0.0などの数値の場合の特別処理（GradioのUIの問題によるもの）
                if dropdown_value == 0 or dropdown_value == "0" or dropdown_value == 0.0:
                    # 数値の0を"なし"として扱う
                    dropdown_value = translate("なし")
                
                # 型チェックと文字列変換を追加
                if not isinstance(dropdown_value, str) and dropdown_value is not None:
                    dropdown_value = str(dropdown_value)
                
                if dropdown_value and dropdown_value != translate("なし"):
                    has_selected_lora = True
                    break
            
            # LoRA選択があれば強制的に有効にする
            if has_selected_lora:
                use_lora = True
                print(translate("[INFO] workerプロセス: ディレクトリでLoRAが選択されているため、LoRA使用を有効にしました"))
        
        # ファイルアップロードモードでLoRAファイルが選択されている場合も強制的に有効に
        elif not use_lora and has_lora_support:
            if ((lora_files is not None and hasattr(lora_files, 'name')) or 
                (lora_files2 is not None and hasattr(lora_files2, 'name')) or 
                (lora_files3 is not None and hasattr(lora_files3, 'name'))):
                use_lora = True
                print(translate("[INFO] workerプロセス: LoRAファイルが選択されているため、LoRA使用を有効にしました"))
        
        if use_lora and has_lora_support:
            # LoRAの読み込み方式によって処理を分岐
            lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
            
            # UI状態からモードを検出 (lora_modeがgr.Radioオブジェクトの場合、valueプロパティを使用)
            # 重要: worker関数内では、直接引数として渡されたlora_modeの値を尊重する
            lora_mode_value = lora_mode
            if hasattr(lora_mode, 'value'):
                try:
                    # valueプロパティが存在する場合は、それを使用
                    temp_value = lora_mode.value
                    if temp_value and isinstance(temp_value, str):
                        lora_mode_value = temp_value
                except:
                    # エラーが発生した場合は引数の値を直接使用
                    pass
            print(translate("[DEBUG] lora_mode_value 型: {0}, 値: {1}").format(type(lora_mode_value).__name__, lora_mode_value))
            
            # ドロップダウン値のデバッグ情報を出力
            print(translate("[DEBUG] worker内のLoRAドロップダウン値（元の値）:"))
            print(translate("  - lora_dropdown1: {0}, 型: {1}").format(lora_dropdown1, type(lora_dropdown1).__name__))
            print(translate("  - lora_dropdown2: {0}, 型: {1}").format(lora_dropdown2, type(lora_dropdown2).__name__))
            print(translate("  - lora_dropdown3: {0}, 型: {1}").format(lora_dropdown3, type(lora_dropdown3).__name__))
            if lora_mode_value and lora_mode_value == translate("ディレクトリから選択"):
                # ディレクトリから選択モード
                print(translate("[INFO] LoRA読み込み方式: ディレクトリから選択"))
                
                # ドロップダウンから選択されたファイルを処理
                # 渡されたままの値を保存
                # lora_dropdown2が数値0の場合、UI上で選択されたはずの値がおかしくなっている可能性あり
                if lora_dropdown2 == 0:
                    print(translate("[DEBUG] lora_dropdown2が数値0になっています。特別処理を実行します"))
                    
                    # loraディレクトリ内の実際のファイルを確認（デバッグ用）
                    lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
                    if os.path.exists(lora_dir):
                        lora_file_listing = []
                        for filename in os.listdir(lora_dir):
                            if filename.endswith(('.safetensors', '.pt', '.bin')):
                                lora_file_listing.append(filename)
                        print(translate("[DEBUG] ディレクトリ内LoRAファイル数: {0}").format(len(lora_file_listing)))
                
                original_dropdowns = {
                    "LoRA1": lora_dropdown1,
                    "LoRA2": lora_dropdown2,
                    "LoRA3": lora_dropdown3
                }
                
                print(translate("[DEBUG] ドロップダウン詳細オリジナル値:"))
                print(f"  lora_dropdown1 = {lora_dropdown1}, 型: {type(lora_dropdown1).__name__}")
                print(f"  lora_dropdown2 = {lora_dropdown2}, 型: {type(lora_dropdown2).__name__}")
                print(f"  lora_dropdown3 = {lora_dropdown3}, 型: {type(lora_dropdown3).__name__}")
                
                # 渡されたドロップダウン値をそのまま使用する（Gradioオブジェクトを避けるため）
                # これは引数として渡された値をそのまま使うアプローチで、Gradioの複雑な内部構造と型の変換を回避
                
                # ドロップダウンの値の問題を特別に処理
                # 問題が起きやすい2番目の値に詳細ログを出力
                print(translate("[DEBUG] 詳細ログ: LoRA2の値={0!r}, 型={1}").format(lora_dropdown2, type(lora_dropdown2).__name__))
                
                # Gradioのバグ対応: 特に2番目の値が数値0になりやすい
                if lora_dropdown2 == 0:
                    print(translate("[DEBUG] LoRA2の値が0になっているため、詳細な状態を確認"))
                    
                    # loraディレクトリの内容を確認
                    lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
                    if os.path.exists(lora_dir):
                        print(translate("[DEBUG] LoRAディレクトリ内容:"))
                        directory_files = []
                        for filename in os.listdir(lora_dir):
                            if filename.endswith(('.safetensors', '.pt', '.bin')):
                                directory_files.append(filename)
                        
                        # 最初の何個かのファイルを表示
                        for i, file in enumerate(directory_files[:5]):
                            print(f"  {i+1}. {file}")
                
                dropdown_direct_values = {
                    "dropdown1": original_dropdowns["LoRA1"],
                    "dropdown2": original_dropdowns["LoRA2"],
                    "dropdown3": original_dropdowns["LoRA3"]
                }
                
                # 各ドロップダウンを処理
                for dropdown_name, dropdown_direct_value in dropdown_direct_values.items():
                    # ドロップダウンの値を直接使用
                    print(translate("[DEBUG] ドロップダウン{0}処理直接使用: 値={1}, 型={2}, 数値として表示={3}").format(
                        dropdown_name, 
                        repr(dropdown_direct_value), 
                        type(dropdown_direct_value).__name__,
                        "0" if dropdown_direct_value == 0 or dropdown_direct_value == 0.0 else "非0またはNone"
                    ))
                    
                    # 特に第2ドロップダウンの処理を強化（問題が最も頻繁に発生している場所）
                    if dropdown_name == "dropdown2" and dropdown_direct_value == 0:
                        print(translate("[DEBUG] dropdown2の特別処理: 数値0が検出されました。元の値: {0}").format(lora_dropdown2))
                        if isinstance(lora_dropdown2, str) and lora_dropdown2 != "0" and lora_dropdown2 != translate("なし"):
                            # 元の引数の値が文字列で、有効な値なら使用
                            dropdown_direct_value = lora_dropdown2
                            print(translate("[DEBUG] dropdown2の値を元の引数から復元: {0}").format(dropdown_direct_value))
                    
                    # 処理用の変数にコピー
                    dropdown_value = dropdown_direct_value
                    
                    # 通常の値が0や0.0などの数値の場合の特別処理（GradioのUIの問題によるもの）
                    # 先に数値0かどうかをチェック（文字列変換前）
                    if dropdown_value == 0 or dropdown_value == 0.0 or dropdown_value == "0":
                        # 数値の0を"なし"として扱う
                        print(translate("[DEBUG] {name}の値が数値0として検出されました。'なし'として扱います").format(name=dropdown_name))
                        dropdown_value = translate("なし")
                    # この段階で文字列変換を強制的に行う（Gradioの型が入り乱れる問題に対処）
                    elif dropdown_value is not None and not isinstance(dropdown_value, str):
                        print(translate("[DEBUG] {name}の前処理: 非文字列値が検出されたため文字列変換を実施: 値={1}, 型={2}").format(
                            dropdown_name, dropdown_value, type(dropdown_value).__name__
                        ))
                        dropdown_value = str(dropdown_value)
                    
                    # 最終的な型チェック - 万一文字列になっていない場合の保険
                    if dropdown_value is not None and not isinstance(dropdown_value, str):
                        print(translate("[DEBUG] {name}の値のタイプが依然として非文字列です: {type}").format(name=dropdown_name, type=type(dropdown_value).__name__))
                        dropdown_value = str(dropdown_value)
                    
                    # コード実行前の最終状態を記録
                    print(translate("[DEBUG] {name}の最終値 (loading前): 値={value!r}, 型={type}, 'なし'と比較={is_none}").format(
                        name=dropdown_name, 
                        value=dropdown_value, 
                        type=type(dropdown_value).__name__,
                        is_none="True" if dropdown_value == translate("なし") else "False"
                    ))
                    
                    if dropdown_value and dropdown_value != translate("なし"):
                        lora_path = os.path.join(lora_dir, dropdown_value)
                        print(translate("[DEBUG] {name}のロード試行: パス={path}").format(name=dropdown_name, path=lora_path))
                        if os.path.exists(lora_path):
                            current_lora_paths.append(lora_path)
                            print(translate("[INFO] {name}を選択: {path}").format(name=dropdown_name, path=lora_path))
                        else:
                            # パスを修正して再試行（単なるファイル名の場合）
                            if os.path.dirname(lora_path) == lora_dir and not os.path.isabs(dropdown_value):
                                # すでに正しく構築されているので再試行不要
                                print(translate("[WARN] 選択された{name}が見つかりません: {file}").format(name=dropdown_name, file=dropdown_value))
                            else:
                                # 直接ファイル名だけで試行
                                lora_path_retry = os.path.join(lora_dir, os.path.basename(str(dropdown_value)))
                                print(translate("[DEBUG] {name}を再試行: {path}").format(name=dropdown_name, path=lora_path_retry))
                                if os.path.exists(lora_path_retry):
                                    current_lora_paths.append(lora_path_retry)
                                    print(translate("[INFO] {name}を選択 (パス修正後): {path}").format(name=dropdown_name, path=lora_path_retry))
                                else:
                                    print(translate("[WARN] 選択された{name}が見つかりません: {file}").format(name=dropdown_name, file=dropdown_value))
            else:
                # ファイルアップロードモード
                print(translate("[INFO] LoRA読み込み方式: ファイルアップロード"))
                
                # 1つ目のLoRAファイルを処理
                if lora_files is not None:
                    # デバッグ情報を出力
                    print(f"[DEBUG] lora_filesの型: {type(lora_files)}, 値: {lora_files}")
                    
                    if isinstance(lora_files, list):
                        # 複数のLoRAファイル（将来のGradioバージョン用）
                        # 各ファイルがnameプロパティを持っているか確認
                        for file in lora_files:
                            if hasattr(file, 'name') and file.name:
                                current_lora_paths.append(file.name)
                            else:
                                print(translate("[WARN] LoRAファイル1のリスト内に無効なファイルがあります"))
                    else:
                        # 単一のLoRAファイル
                        # nameプロパティがあるか確認
                        if hasattr(lora_files, 'name') and lora_files.name:
                            current_lora_paths.append(lora_files.name)
                        else:
                            print(translate("[WARN] 1つ目のLoRAファイルは無効か選択されていません"))
                
                # 2つ目のLoRAファイルがあれば追加
                if lora_files2 is not None:
                    # デバッグ情報を出力
                    print(f"[DEBUG] lora_files2の型: {type(lora_files2)}, 値: {lora_files2}")
                    
                    if isinstance(lora_files2, list):
                        # 複数のLoRAファイル（将来のGradioバージョン用）
                        # 各ファイルがnameプロパティを持っているか確認
                        for file in lora_files2:
                            if hasattr(file, 'name') and file.name:
                                current_lora_paths.append(file.name)
                            else:
                                print(translate("[WARN] LoRAファイル2のリスト内に無効なファイルがあります"))
                    else:
                        # 単一のLoRAファイル
                        # nameプロパティがあるか確認
                        if hasattr(lora_files2, 'name') and lora_files2.name:
                            current_lora_paths.append(lora_files2.name)
                        else:
                            print(translate("[WARN] 2つ目のLoRAファイルは無効か選択されていません"))
                
                # 3つ目のLoRAファイルがあれば追加
                if lora_files3 is not None:
                    # デバッグ情報を出力
                    print(f"[DEBUG] lora_files3の型: {type(lora_files3)}, 値: {lora_files3}")
                    
                    if isinstance(lora_files3, list):
                        # 複数のLoRAファイル（将来のGradioバージョン用）
                        # 各ファイルがnameプロパティを持っているか確認
                        for file in lora_files3:
                            if hasattr(file, 'name') and file.name:
                                current_lora_paths.append(file.name)
                            else:
                                print(translate("[WARN] LoRAファイル3のリスト内に無効なファイルがあります"))
                    else:
                        # 単一のLoRAファイル
                        # nameプロパティがあるか確認
                        if hasattr(lora_files3, 'name') and lora_files3.name:
                            current_lora_paths.append(lora_files3.name)
                        else:
                            print(translate("[WARN] 3つ目のLoRAファイルは無効か選択されていません"))
            
            # スケール値をテキストから解析
            if current_lora_paths:  # LoRAパスがある場合のみ解析
                try:
                    scales_text = lora_scales_text.strip()
                    if scales_text:
                        # カンマ区切りのスケール値を解析
                        scales = [float(scale.strip()) for scale in scales_text.split(',')]
                        current_lora_scales = scales
                    else:
                        # スケール値が指定されていない場合は全て0.8を使用
                        current_lora_scales = [0.8] * len(current_lora_paths)
                except Exception as e:
                    print(translate("LoRAスケール解析エラー: {0}").format(e))
                    print(translate("デフォルトスケール 0.8 を使用します"))
                    current_lora_scales = [0.8] * len(current_lora_paths)
                
                # スケール値の数がLoRAパスの数と一致しない場合は調整
                if len(current_lora_scales) < len(current_lora_paths):
                    # 足りない分は0.8で埋める
                    current_lora_scales.extend([0.8] * (len(current_lora_paths) - len(current_lora_scales)))
                elif len(current_lora_scales) > len(current_lora_paths):
                    # 余分は切り捨て
                    current_lora_scales = current_lora_scales[:len(current_lora_paths)]

        # UIでLoRA使用が有効になっていた場合、ファイル選択に関わらず強制的に有効化
        if original_use_lora:
            use_lora = True
            print(translate("[INFO] UIでLoRA使用が有効化されているため、LoRA使用を有効にします"))

        print(f"[DEBUG] 最終的なuse_loraフラグ: {use_lora}")

        # LoRA設定を更新（リロードは行わない）
        transformer_manager.set_next_settings(
            lora_paths=current_lora_paths,
            lora_scales=current_lora_scales,
            fp8_enabled=fp8_optimization,  # fp8_enabledパラメータを追加
            high_vram_mode=high_vram,
            force_dict_split=True  # 常に辞書分割処理を行う
        )

        # -------- LoRA 設定 END ---------

        # -------- FP8 設定 START ---------
        # FP8設定（既にLoRA設定に含めたので不要）
        # この行は削除しても問題ありません
        # -------- FP8 設定 END ---------

        # セクション処理開始前にtransformerの状態を確認
        print(translate("\nセクション処理開始前のtransformer状態チェック..."))
        try:
            # transformerの状態を確認し、必要に応じてリロード
            if not transformer_manager.ensure_transformer_state():
                raise Exception(translate("transformer状態の確認に失敗しました"))

            # 最新のtransformerインスタンスを取得
            transformer = transformer_manager.get_transformer()
            print(translate("transformer状態チェック完了"))
        except Exception as e:
            print(translate("transformer状態チェックエラー: {0}").format(e))
            traceback.print_exc()
            raise e

        for i_section, latent_padding in enumerate(latent_paddings):
            # 先に変数を定義
            is_first_section = i_section == 0

            # オールパディングの場合の特別処理
            if use_all_padding:
                # 最後のセクションの判定
                is_last_section = i_section == len(latent_paddings) - 1

                # 内部処理用に元の値を保存
                orig_padding_value = latent_padding

                # 最後のセクションが0より大きい場合は警告と強制変換
                if is_last_section and float(latent_padding) > 0:
                    print(translate("警告: 最後のセクションのパディング値は内部計算のために0に強制します。"))
                    latent_padding = 0
                elif isinstance(latent_padding, float):
                    # 浮動小数点の場合はそのまま使用（小数点対応）
                    # 小数点1桁に固定のみ行い、丸めは行わない
                    latent_padding = round(float(latent_padding), 1)

                # 値が変更された場合にデバッグ情報を出力
                if float(orig_padding_value) != float(latent_padding):
                    print(translate("パディング値変換: セクション{0}の値を{1}から{2}に変換しました").format(i_section, orig_padding_value, latent_padding))
            else:
                # 通常モードの場合
                is_last_section = latent_padding == 0

            use_end_latent = is_last_section and end_frame is not None
            latent_padding_size = int(latent_padding * latent_window_size)

            # 定義後にログ出力
            padding_info = translate("設定パディング値: {0}").format(all_padding_value) if use_all_padding else translate("パディング値: {0}").format(latent_padding)
            print(translate("\n■ セクション{0}の処理開始 ({1})").format(i_section, padding_info))
            print(translate("  - 現在の生成フレーム数: {0}フレーム").format(total_generated_latent_frames * 4 - 3))
            print(translate("  - 生成予定フレーム数: {0}フレーム").format(num_frames))
            print(translate("  - 最初のセクション?: {0}").format(is_first_section))
            print(translate("  - 最後のセクション?: {0}").format(is_last_section))
            # set current_latent here
            # セクションごとのlatentを使う場合
            if section_map and section_latents is not None and len(section_latents) > 0:
                # i_section以上で最小のsection_latentsキーを探す
                valid_keys = [k for k in section_latents.keys() if k >= i_section]
                if valid_keys:
                    use_key = min(valid_keys)
                    current_latent = section_latents[use_key]
                    print(translate("[section_latent] section {0}: use section {1} latent (section_map keys: {2})").format(i_section, use_key, list(section_latents.keys())))
                    print(translate("[section_latent] current_latent id: {0}, min: {1:.4f}, max: {2:.4f}, mean: {3:.4f}").format(id(current_latent), current_latent.min().item(), current_latent.max().item(), current_latent.mean().item()))
                else:
                    current_latent = start_latent
                    print(translate("[section_latent] section {0}: use start_latent (no section_latent >= {1})").format(i_section, i_section))
                    print(translate("[section_latent] current_latent id: {0}, min: {1:.4f}, max: {2:.4f}, mean: {3:.4f}").format(id(current_latent), current_latent.min().item(), current_latent.max().item(), current_latent.mean().item()))
            else:
                current_latent = start_latent
                print(translate("[section_latent] section {0}: use start_latent (no section_latents)").format(i_section))
                print(translate("[section_latent] current_latent id: {0}, min: {1:.4f}, max: {2:.4f}, mean: {3:.4f}").format(id(current_latent), current_latent.min().item(), current_latent.max().item(), current_latent.mean().item()))

            if is_first_section and end_frame_latent is not None:
                # EndFrame影響度設定を適用（デフォルトは1.0=通常の影響）
                if end_frame_strength != 1.0:
                    # 影響度を適用した潜在表現を生成
                    # 値が小さいほど影響が弱まるように単純な乗算を使用
                    # end_frame_strength=1.0のときは1.0倍（元の値）
                    # end_frame_strength=0.01のときは0.01倍（影響が非常に弱い）
                    modified_end_frame_latent = end_frame_latent * end_frame_strength
                    print(translate("EndFrame影響度を{0}に設定（最終フレームの影響が{1}倍）").format(f"{end_frame_strength:.2f}", f"{end_frame_strength:.2f}"))
                    try:
                        history_latents[:, :, 0:1, :, :] = modified_end_frame_latent
                    except RuntimeError as e:
                        # テンソルサイズの不一致エラーを検出して、わかりやすいエラーメッセージを表示
                        error_msg = str(e)
                        if "The expanded size of the tensor" in error_msg and "must match the existing size" in error_msg:
                            # エラーメッセージのデバッグ出力
                            print(translate("解像度不一致エラー検出: {0}").format(error_msg))
                            # 入力画像と最終フレーム画像のサイズを表示
                            print(translate("入力画像サイズ: {0}x{1}").format(width, height))
                            
                            # コンソールにエラーを出力するだけのシンプルな方法
                            print("\n" + "*" * 80)
                            print(translate("❗️ 入力画像と最終フレーム画像の解像度が一致していません"))
                            print(translate("同じ解像度の画像を使用してください"))
                            print("*" * 80 + "\n")
                            # 処理を終了
                            stream.output_queue.push(('end', None))
                            return
                        else:
                            # その他のエラーはそのまま再発生
                            raise
                else:
                    # 通常の処理（通常の影響）
                    try:
                        history_latents[:, :, 0:1, :, :] = end_frame_latent
                    except RuntimeError as e:
                        # テンソルサイズの不一致エラーを検出して、わかりやすいエラーメッセージを表示
                        error_msg = str(e)
                        if "The expanded size of the tensor" in error_msg and "must match the existing size" in error_msg:
                            # エラーメッセージのデバッグ出力
                            print(translate("解像度不一致エラー検出: {0}").format(error_msg))
                            # 入力画像と最終フレーム画像のサイズを表示
                            print(translate("入力画像サイズ: {0}x{1}").format(width, height))
                            
                            # コンソールにエラーを出力するだけのシンプルな方法
                            print("\n" + "*" * 80)
                            print(translate("❗️ 入力画像と最終フレーム画像の解像度が一致していません"))
                            print(translate("同じ解像度の画像を使用してください"))
                            print("*" * 80 + "\n")
                            # 処理を終了
                            stream.output_queue.push(('end', None))
                            return
                        else:
                            # その他のエラーはそのまま再発生
                            raise

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            # セクション固有のプロンプトがあれば使用する（事前にエンコードしたキャッシュを使用）
            current_llama_vec, current_clip_l_pooler, current_llama_attention_mask = process_section_prompt(i_section, section_map, llama_vec, clip_l_pooler, llama_attention_mask, section_prompt_embeddings)

            print(translate('latent_padding_size = {0}, is_last_section = {1}').format(latent_padding_size, is_last_section))


            # COMMENTED OUT: セクション処理前のメモリ解放（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()

            # latent_window_sizeが4.5の場合は特別に5を使用
            effective_window_size = 5 if latent_window_size == 4.5 else int(latent_window_size)
            indices = torch.arange(0, sum([1, latent_padding_size, effective_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, effective_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = current_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                # GPUメモリ保存値を明示的に浮動小数点に変換
                preserved_memory = float(gpu_memory_preservation) if gpu_memory_preservation is not None else 6.0
                print(translate('Setting transformer memory preservation to: {0} GB').format(preserved_memory))
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=preserved_memory)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = translate('Sampling {0}/{1}').format(current_step, steps)
                # セクション情報を追加（現在のセクション/全セクション）
                section_info = translate('セクション: {0}/{1}').format(i_section+1, total_sections)
                desc = f"{section_info} " + translate('生成フレーム数: {total_generated_latent_frames}, 動画長: {video_length:.2f} 秒 (FPS-30). 動画が生成中です ...').format(section_info=section_info, total_generated_latent_frames=int(max(0, total_generated_latent_frames * 4 - 3)), video_length=max(0, (total_generated_latent_frames * 4 - 3) / 30))
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=current_llama_vec,  # セクションごとのプロンプトを使用
                prompt_embeds_mask=current_llama_attention_mask,  # セクションごとのマスクを使用
                prompt_poolers=current_clip_l_pooler,  # セクションごとのプロンプトを使用
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_first_section:
                # 末尾から削除するフレームサイズを計算
                generated_latent_size = generated_latents.shape[2]
                trim_end_latent_size = current_trim_end_latents.value
                if generated_latent_size > trim_end_latent_size:
                    if trim_end_latent_size > 0:
                        generated_latents = generated_latents[:, :, :-trim_end_latent_size, :, :].clone()
                        print(translate("生成されたデータの末尾フレームを削除しました。削除数: {0}/{1}").format(trim_end_latent_size, generated_latent_size))
                else:
                    print(translate("警告: 生成されたデータのフレーム数よりも、末尾フレーム削除数が大きく指定されているため、末尾フレーム削除は実施しません。"))

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                # 減圧時に使用するGPUメモリ値も明示的に浮動小数点に設定
                preserved_memory_offload = 8.0  # こちらは固定値のまま
                print(translate('Offloading transformer with memory preservation: {0} GB').format(preserved_memory_offload))
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=preserved_memory_offload)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            # COMMENTED OUT: VAEデコード前のメモリクリア（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()
            #     print(translate("VAEデコード前メモリ: {memory_allocated:.2f}GB").format(memory_allocated=torch.cuda.memory_allocated()/1024**3))

            if history_pixels is None:
                # VAEキャッシュ設定に応じてデコード関数を切り替え
                print(f"[DEBUG] 履歴フレームデコード前のVAEキャッシュ設定: {use_vae_cache}, 型: {type(use_vae_cache)}")
                if use_vae_cache:
                    print(translate("[INFO] VAEキャッシュを使用: 履歴フレーム"))
                    history_pixels = vae_decode_cache(real_history_latents, vae).cpu()
                else:
                    print("[INFO] 通常デコード使用: 履歴フレーム")
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                
                # 最初のセクションで全フレーム画像を保存
                # 「全フレーム画像保存」または「最終セクションのみ全フレーム画像保存かつ最終セクション」が有効な場合
                # 最終セクションかどうかを判断
                is_last_section = i_section == total_sections - 1
                print(translate("\n[DEBUG] 現在のセクション: {0}, 総セクション数: {1}, 最終セクションと判定: {2}").format(i_section, total_sections, is_last_section))
                
                # save_latent_frames と save_last_section_frames の値をcopy
                # ループ内の変数を変更してもグローバルな値は変わらないため
                # 注意：既にここに来る前に万が一の文字列→ブール変換処理が済んでいるはず
                
                # デバッグ情報を追加：実際に使用されるフラグの値を確認
                print(translate("[DEBUG] セクション{0}の処理開始時 - 現在のsave_latent_frames型: {1}, 値: {2}").format(
                    i_section, type(save_latent_frames).__name__, save_latent_frames
                ))
                print(translate("[DEBUG] セクション{0}の処理開始時 - 現在のsave_last_section_frames型: {1}, 値: {2}").format(
                    i_section, type(save_last_section_frames).__name__, save_last_section_frames
                ))
                
                # 値のコピーではなく、明示的に新しい変数に適切な値を設定
                # BooleanかStringかの型変換ミスを防ぐ
                is_save_all_frames = bool(save_latent_frames)
                is_save_last_frame_only = bool(save_last_section_frames)
                
                # デバッグ情報を追加：変換後の値を確認
                print(translate("[DEBUG] セクション{0}の処理 - 変換後のis_save_all_frames型: {1}, 値: {2}").format(
                    i_section, type(is_save_all_frames).__name__, is_save_all_frames
                ))
                print(translate("[DEBUG] セクション{0}の処理 - 変換後のis_save_last_frame_only型: {1}, 値: {2}").format(
                    i_section, type(is_save_last_frame_only).__name__, is_save_last_frame_only
                ))
                
                # フレーム保存の判定ロジック
                if is_save_all_frames:
                    should_save_frames = True
                    print(translate("[DEBUG] 全フレーム画像保存が有効: 全セクションでフレーム保存します"))
                elif is_save_last_frame_only and is_last_section:
                    should_save_frames = True
                    print(translate("[DEBUG] 最終セクションのみ全フレーム画像保存が有効: 現在のセクション{0}が最終セクションと判定されたため保存します").format(i_section))
                else:
                    should_save_frames = False
                    if is_save_last_frame_only:
                        print(translate("[DEBUG] 最終セクションのみ全フレーム画像保存が有効: 現在のセクション{0}は最終セクションではないためスキップします").format(i_section))
                    else:
                        print(translate("[DEBUG] フレーム画像保存は無効です"))
                if should_save_frames and history_pixels is not None:
                    try:
                        # フレーム数
                        latent_frame_count = history_pixels.shape[2]
                        
                        # 保存モードに応じたメッセージを表示
                        # グローバル変数ではなく、ローカルのcopyを使用
                        if is_save_all_frames:
                            print(translate("\n[INFO] 全フレーム画像保存: 最初のセクション{0}の{1}フレームを保存します").format(i_section, latent_frame_count))
                        elif is_save_last_frame_only and is_last_section:
                            print(translate("\n[INFO] 最終セクションのみ全フレーム画像保存: セクション{0}/{1}の{2}フレームを保存します (最終セクション)").format(i_section, total_sections-1, latent_frame_count))
                        else:
                            print(translate("\n[INFO] フレーム画像保存: セクション{0}の{1}フレームを保存します").format(i_section, latent_frame_count))
                        
                        # セクションごとのフォルダを作成
                        frames_folder = os.path.join(outputs_folder, f'{job_id}_frames_section{i_section}')
                        os.makedirs(frames_folder, exist_ok=True)
                        
                        # 各フレームの保存
                        for frame_idx in range(latent_frame_count):
                            # フレームを取得
                            frame = history_pixels[0, :, frame_idx, :, :]
                            frame = einops.rearrange(frame, 'c h w -> h w c')
                            frame = frame.cpu().numpy()
                            frame = np.clip((frame * 127.5 + 127.5), 0, 255).astype(np.uint8)
                            frame = resize_and_center_crop(frame, target_width=width, target_height=height)
                            
                            # メタデータの準備
                            frame_metadata = {
                                PROMPT_KEY: prompt,  # メインプロンプト
                                SEED_KEY: seed,
                                SECTION_NUMBER_KEY: i_section,
                                "FRAME_NUMBER": frame_idx  # フレーム番号も追加
                            }
                            
                            # セクション固有のプロンプトがあれば追加
                            if section_map and i_section in section_map:
                                _, section_prompt = section_map[i_section]
                                if section_prompt and section_prompt.strip():
                                    frame_metadata[SECTION_PROMPT_KEY] = section_prompt
                            
                            # 画像の保存とメタデータの埋め込み
                            frame_path = os.path.join(frames_folder, f'frame_{frame_idx:03d}.png')
                            Image.fromarray(frame).save(frame_path)
                            embed_metadata_to_png(frame_path, frame_metadata)
                        
                        # 保存モードに応じたメッセージを表示
                        # グローバル変数ではなく、ローカルのcopyを使用
                        if is_save_all_frames:
                            print(translate("[INFO] 全フレーム画像保存: セクション{0}の{1}個のフレーム画像を保存しました: {2}").format(i_section, latent_frame_count, frames_folder))
                        elif is_save_last_frame_only and is_last_section:
                            print(translate("[INFO] 最終セクションのみ全フレーム画像保存: セクション{0}/{1}の{2}個のフレーム画像を保存しました (最終セクション): {3}").format(i_section, total_sections-1, latent_frame_count, frames_folder))
                        else:
                            print(translate("[INFO] セクション{0}の{1}個のフレーム画像を保存しました: {2}").format(i_section, latent_frame_count, frames_folder))
                    except Exception as e:
                        print(translate("[WARN] セクション{0}のフレーム画像保存中にエラー: {1}").format(i_section, e))
                        traceback.print_exc()
            else:
                # latent_window_sizeが4.5の場合は特別に5を使用
                if latent_window_size == 4.5:
                    section_latent_frames = 11 if is_last_section else 10  # 5 * 2 + 1 = 11, 5 * 2 = 10
                    overlapped_frames = 17  # 5 * 4 - 3 = 17
                else:
                    section_latent_frames = int(latent_window_size * 2 + 1) if is_last_section else int(latent_window_size * 2)
                    overlapped_frames = int(latent_window_size * 4 - 3)

                # VAEキャッシュ設定に応じてデコード関数を切り替え
                if use_vae_cache:
                    print(translate("[INFO] VAEキャッシュを使用: 現在のセクション"))
                    current_pixels = vae_decode_cache(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                else:
                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()

                if overlapped_frames > history_pixels.shape[2]:
                    overlapped_frames = history_pixels.shape[2]
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

                # 各セクションで生成された個々のフレームを静止画として保存
                # 「全フレーム画像保存」または「最終セクションのみ全フレーム画像保存かつ最終セクション」が有効な場合
                # 最終セクションかどうかを再判断
                is_last_section = i_section == total_sections - 1
                print(translate("\n[DEBUG] 現在のセクション: {0}, 総セクション数: {1}, 最終セクションと判定: {2}").format(i_section, total_sections, is_last_section))
                
                # save_latent_frames と save_last_section_frames の値をcopy
                # ループ内の変数を変更してもグローバルな値は変わらないため
                # 注意：既にここに来る前に万が一の文字列→ブール変換処理が済んでいるはず
                
                # デバッグ情報を追加：実際に使用されるフラグの値を確認
                print(translate("[DEBUG] セクション{0}の処理開始時 - 現在のsave_latent_frames型: {1}, 値: {2}").format(
                    i_section, type(save_latent_frames).__name__, save_latent_frames
                ))
                print(translate("[DEBUG] セクション{0}の処理開始時 - 現在のsave_last_section_frames型: {1}, 値: {2}").format(
                    i_section, type(save_last_section_frames).__name__, save_last_section_frames
                ))
                
                # 値のコピーではなく、明示的に新しい変数に適切な値を設定
                # BooleanかStringかの型変換ミスを防ぐ
                is_save_all_frames = bool(save_latent_frames)
                is_save_last_frame_only = bool(save_last_section_frames)
                
                # デバッグ情報を追加：変換後の値を確認
                print(translate("[DEBUG] セクション{0}の処理 - 変換後のis_save_all_frames型: {1}, 値: {2}").format(
                    i_section, type(is_save_all_frames).__name__, is_save_all_frames
                ))
                print(translate("[DEBUG] セクション{0}の処理 - 変換後のis_save_last_frame_only型: {1}, 値: {2}").format(
                    i_section, type(is_save_last_frame_only).__name__, is_save_last_frame_only
                ))
                
                # フレーム保存の判定ロジック
                if is_save_all_frames:
                    should_save_frames = True
                    print(translate("[DEBUG] 全フレーム画像保存が有効: 全セクションでフレーム保存します"))
                elif is_save_last_frame_only and is_last_section:
                    should_save_frames = True
                    print(translate("[DEBUG] 最終セクションのみ全フレーム画像保存が有効: 現在のセクション{0}が最終セクションと判定されたため保存します").format(i_section))
                else:
                    should_save_frames = False
                    if is_save_last_frame_only:
                        print(translate("[DEBUG] 最終セクションのみ全フレーム画像保存が有効: 現在のセクション{0}は最終セクションではないためスキップします").format(i_section))
                    else:
                        print(translate("[DEBUG] フレーム画像保存は無効です"))
                if should_save_frames:
                    try:
                        # source_pixelsは、このセクションで使用するピクセルデータ
                        source_pixels = None
                        
                        # どのソースを使用するかを決定
                        # i_section=0の場合、current_pixelsが定義される前に参照されるためエラーとなる可能性がある
                        # history_pixelsを優先して使用するよう処理順序を変更
                        if history_pixels is not None:
                            source_pixels = history_pixels
                            print(translate("\n[INFO] 全フレーム画像保存: history_pixelsを使用します"))
                        elif 'current_pixels' in locals() and current_pixels is not None:
                            source_pixels = current_pixels
                            print(translate("\n[INFO] 全フレーム画像保存: current_pixelsを使用します"))
                        else:
                            print(translate("\n[WARN] 全フレーム画像保存: 有効なピクセルデータがありません"))
                            return
                            
                        # フレーム数（1秒モードでは9フレーム、0.5秒モードでは5フレーム）
                        latent_frame_count = source_pixels.shape[2]
                        
                        # 保存モードに応じたメッセージを表示
                        # グローバル変数ではなく、ローカルのcopyを使用
                        if is_save_all_frames:
                            print(translate("[INFO] 全フレーム画像保存: セクション{0}の{1}フレームを保存します").format(i_section, latent_frame_count))
                        elif is_save_last_frame_only and is_last_section:
                            print(translate("[INFO] 最終セクションのみ全フレーム画像保存: セクション{0}/{1}の{2}フレームを保存します (最終セクション)").format(i_section, total_sections-1, latent_frame_count))
                        else:
                            print(translate("[INFO] フレーム画像保存: セクション{0}の{1}フレームを保存します").format(i_section, latent_frame_count))
                        
                        # セクションごとのフォルダを作成
                        frames_folder = os.path.join(outputs_folder, f'{job_id}_frames_section{i_section}')
                        os.makedirs(frames_folder, exist_ok=True)
                        
                        # 各フレームの保存
                        for frame_idx in range(latent_frame_count):
                            # フレームを取得
                            frame = source_pixels[0, :, frame_idx, :, :]
                            frame = einops.rearrange(frame, 'c h w -> h w c')
                            frame = frame.cpu().numpy()
                            frame = np.clip((frame * 127.5 + 127.5), 0, 255).astype(np.uint8)
                            frame = resize_and_center_crop(frame, target_width=width, target_height=height)
                            
                            # メタデータの準備
                            frame_metadata = {
                                PROMPT_KEY: prompt,  # メインプロンプト
                                SEED_KEY: seed,
                                SECTION_NUMBER_KEY: i_section,
                                "FRAME_NUMBER": frame_idx  # フレーム番号も追加
                            }
                            
                            # セクション固有のプロンプトがあれば追加
                            if section_map and i_section in section_map:
                                _, section_prompt = section_map[i_section]
                                if section_prompt and section_prompt.strip():
                                    frame_metadata[SECTION_PROMPT_KEY] = section_prompt
                            
                            # 画像の保存とメタデータの埋め込み
                            frame_path = os.path.join(frames_folder, f'frame_{frame_idx:03d}.png')
                            Image.fromarray(frame).save(frame_path)
                            embed_metadata_to_png(frame_path, frame_metadata)
                        
                        # 保存モードに応じたメッセージを表示
                        # グローバル変数ではなく、ローカルのcopyを使用
                        if is_save_all_frames:
                            print(translate("[INFO] 全フレーム画像保存: セクション{0}の{1}個のフレーム画像を保存しました: {2}").format(i_section, latent_frame_count, frames_folder))
                        elif is_save_last_frame_only and is_last_section:
                            print(translate("[INFO] 最終セクションのみ全フレーム画像保存: セクション{0}/{1}の{2}個のフレーム画像を保存しました (最終セクション): {3}").format(i_section, total_sections-1, latent_frame_count, frames_folder))
                        else:
                            print(translate("[INFO] セクション{0}の{1}個のフレーム画像を保存しました: {2}").format(i_section, latent_frame_count, frames_folder))
                    except Exception as e:
                        print(translate("[WARN] セクション{0}のフレーム画像保存中にエラー: {1}").format(i_section, e))
                        traceback.print_exc()

            # COMMENTED OUT: 明示的なCPU転送と不要テンソルの削除（処理速度向上のため）
            # if torch.cuda.is_available():
            #     # 必要なデコード後、明示的にキャッシュをクリア
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()

            # 各セクションの最終フレームを静止画として保存（セクション番号付き）
            if save_section_frames and history_pixels is not None:
                try:
                    if i_section == 0 or current_pixels is None:
                        # 最初のセクションは history_pixels の最後
                        last_frame = history_pixels[0, :, -1, :, :]
                    else:
                        # 2セクション目以降は current_pixels の最後
                        last_frame = current_pixels[0, :, -1, :, :]
                    last_frame = einops.rearrange(last_frame, 'c h w -> h w c')
                    last_frame = last_frame.cpu().numpy()
                    last_frame = np.clip((last_frame * 127.5 + 127.5), 0, 255).astype(np.uint8)
                    last_frame = resize_and_center_crop(last_frame, target_width=width, target_height=height)

                    # メタデータを埋め込むための情報を収集
                    print(translate("\n[DEBUG] セクション{0}のメタデータ埋め込み準備").format(i_section))
                    section_metadata = {
                        PROMPT_KEY: prompt,  # メインプロンプト
                        SEED_KEY: seed,
                        SECTION_NUMBER_KEY: i_section
                    }
                    print(translate("[DEBUG] 基本メタデータ: {0}").format(section_metadata))

                    # セクション固有のプロンプトがあれば取得
                    if section_map and i_section in section_map:
                        _, section_prompt = section_map[i_section]
                        if section_prompt and section_prompt.strip():
                            section_metadata[SECTION_PROMPT_KEY] = section_prompt
                            print(translate("[DEBUG] セクションプロンプトを追加: {0}").format(section_prompt))

                    # 画像の保存とメタデータの埋め込み
                    if is_first_section and end_frame is None:
                        frame_path = os.path.join(outputs_folder, f'{job_id}_{i_section}_end.png')
                        print(translate("[DEBUG] セクション画像パス: {0}").format(frame_path))
                        Image.fromarray(last_frame).save(frame_path)
                        print(translate("[DEBUG] メタデータ埋め込み実行: {0}").format(section_metadata))
                        embed_metadata_to_png(frame_path, section_metadata)
                    else:
                        frame_path = os.path.join(outputs_folder, f'{job_id}_{i_section}.png')
                        print(translate("[DEBUG] セクション画像パス: {0}").format(frame_path))
                        Image.fromarray(last_frame).save(frame_path)
                        print(translate("[DEBUG] メタデータ埋め込み実行: {0}").format(section_metadata))
                        embed_metadata_to_png(frame_path, section_metadata)

                    print(translate("\u2713 セクション{0}のフレーム画像をメタデータ付きで保存しました").format(i_section))
                except Exception as e:
                    print(translate("[WARN] セクション{0}最終フレーム画像保存時にエラー: {1}").format(i_section, e))

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(translate('Decoded. Current latent shape {0}; pixel shape {1}').format(real_history_latents.shape, history_pixels.shape))

            # COMMENTED OUT: セクション処理後の明示的なメモリ解放（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()
            #     import gc
            #     gc.collect()
            #     memory_allocated = torch.cuda.memory_allocated()/1024**3
            #     memory_reserved = torch.cuda.memory_reserved()/1024**3
            #     print(translate("セクション後メモリ状態: 割当={0:.2f}GB, 予約={1:.2f}GB").format(memory_allocated, memory_reserved))

            print(translate("■ セクション{0}の処理完了").format(i_section))
            print(translate("  - 現在の累計フレーム数: {0}フレーム").format(int(max(0, total_generated_latent_frames * 4 - 3))))
            print(translate("  - レンダリング時間: {0}秒").format(f"{max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f}"))
            print(translate("  - 出力ファイル: {0}").format(output_filename))

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                combined_output_filename = None
                # 全セクション処理完了後、テンソルデータを後方に結合
                if uploaded_tensor is not None:
                    try:
                        original_frames = real_history_latents.shape[2]  # 元のフレーム数を記録
                        uploaded_frames = uploaded_tensor.shape[2]  # アップロードされたフレーム数

                        print(translate("テンソルデータを後方に結合します: アップロードされたフレーム数 = {uploaded_frames}").format(uploaded_frames=uploaded_frames))
                        # UI上で進捗状況を更新
                        stream.output_queue.push(('progress', (None, translate("テンソルデータ({uploaded_frames}フレーム)の結合を開始します...").format(uploaded_frames=uploaded_frames), make_progress_bar_html(80, translate('テンソルデータ結合準備')))))

                        # テンソルデータを後方に結合する前に、互換性チェック
                        # デバッグログを追加して詳細を出力
                        print(translate("[DEBUG] テンソルデータの形状: {0}, 生成データの形状: {1}").format(uploaded_tensor.shape, real_history_latents.shape))
                        print(translate("[DEBUG] テンソルデータの型: {0}, 生成データの型: {1}").format(uploaded_tensor.dtype, real_history_latents.dtype))
                        print(translate("[DEBUG] テンソルデータのデバイス: {0}, 生成データのデバイス: {1}").format(uploaded_tensor.device, real_history_latents.device))

                        if uploaded_tensor.shape[3] != real_history_latents.shape[3] or uploaded_tensor.shape[4] != real_history_latents.shape[4]:
                            print(translate("警告: テンソルサイズが異なります: アップロード={0}, 現在の生成={1}").format(uploaded_tensor.shape, real_history_latents.shape))
                            print(translate("テンソルサイズの不一致のため、前方結合をスキップします"))
                            stream.output_queue.push(('progress', (None, translate("テンソルサイズの不一致のため、前方結合をスキップしました"), make_progress_bar_html(85, translate('互換性エラー')))))
                        else:
                            # 生成データの末尾のフレームとテンソルデータの先頭のフレームを補間するフレームを追加する
                            if use_interpolation_section:
                                interpolation_latent_size = interpolation_latents.value
                                if interpolation_latents.value > 0 and interpolation_latent_size > 0:
                                    # パディング
                                    interpolation_latent_padding_size = 0
                                    # 生成データの末尾のフレーム
                                    real_history_last_latent = real_history_latents[:, :, -1, :, :].clone()
                                    # テンソルデータの先頭のフレーム
                                    uploaded_tensor_first_latent = uploaded_tensor[:, :, 0, :, :].clone()

                                    # 開始、終了画像を出力
                                    # VAEキャッシュ設定に応じてデコード関数を切り替え
                                    if use_vae_cache:
                                        last_image = vae_decode_cache(real_history_last_latent.clone().unsqueeze(2), vae)
                                    else:
                                        last_image = vae_decode(real_history_last_latent.clone().unsqueeze(2), vae)
                                    last_image = (last_image[0, :, 0] * 127.5 + 127.5).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
                                    # デコードした画像を保存
                                    Image.fromarray(last_image).save(os.path.join(outputs_folder, f'{job_id}_interpolation_start.png'))
                                    # VAEキャッシュ設定に応じてデコード関数を切り替え
                                    if use_vae_cache:
                                        first_image = vae_decode_cache(uploaded_tensor_first_latent.unsqueeze(2), vae)
                                    else:
                                        first_image = vae_decode(uploaded_tensor_first_latent.unsqueeze(2), vae)
                                    first_image = (first_image[0, :, 0] * 127.5 + 127.5).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
                                    # デコードした画像を保存
                                    Image.fromarray(first_image).save(os.path.join(outputs_folder, f'{job_id}_interpolation_end.png'))

                                    def callback_interpolation(d):
                                        preview = d['denoised']
                                        preview = vae_decode_fake(preview)

                                        preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                                        preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                                        if stream.input_queue.top() == 'end':
                                            stream.output_queue.push(('end', None))
                                            raise KeyboardInterrupt('User ends the task.')

                                        current_step = d['i'] + 1
                                        percentage = int(100.0 * current_step / steps)
                                        hint = translate('Sampling {0}/{1}').format(current_step, steps)
                                        desc = "補間データを生成中です ..."
                                        stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                                        return

                                    uploaded_tensor_size = 1 + 2 + 16
                                    if uploaded_tensor.shape[2] < uploaded_tensor_size:
                                        # テンソルデータの足りない部分を補間
                                        interpolation_uploaded_tensor = torch.nn.functional.interpolate(uploaded_tensor, size=(uploaded_tensor_size, uploaded_tensor.shape[3], uploaded_tensor.shape[4]), mode='nearest')
                                    else:
                                        interpolation_uploaded_tensor = uploaded_tensor

                                    effective_window_size_2 = interpolation_latents.value

                                    # 補間frames
                                    interpolation_num_frames = int(effective_window_size_2 * 4 - 3)

                                    # indexとclean_latent
                                    indices_2 = torch.arange(0, sum([1, interpolation_latent_padding_size, effective_window_size_2, 1, 2, 16])).unsqueeze(0)
                                    clean_latent_indices_pre_2, _, latent_indices_2, clean_latent_indices_post_2, clean_latent_2x_indices_2, clean_latent_4x_indices_2 = indices_2.split([1, interpolation_latent_padding_size, effective_window_size_2, 1, 2, 16], dim=1)
                                    clean_latent_indices_2 = torch.cat([clean_latent_indices_pre_2, clean_latent_indices_post_2], dim=1)
                                    clean_latents_post_2, clean_latents_2x_2, clean_latents_4x_2 = interpolation_uploaded_tensor[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                                    clean_latents_2 = torch.cat([real_history_last_latent.unsqueeze(2), clean_latents_post_2[:, :, :1, :, :]], dim=2)

                                    # 補間フレームを生成
                                    generated_interpolation_latents = sample_hunyuan(
                                        transformer=transformer,
                                        sampler='unipc',
                                        width=width,
                                        height=height,
                                        frames=interpolation_num_frames,
                                        real_guidance_scale=cfg,
                                        distilled_guidance_scale=gs,
                                        guidance_rescale=rs,
                                        # shift=3.0,
                                        num_inference_steps=steps,
                                        generator=rnd,
                                        prompt_embeds=llama_vec,  # TODO:メインプロンプトを使用。補間用のプロンプトを入力できてもよいかも
                                        prompt_embeds_mask=llama_attention_mask,  # TODO:メインプロンプトのマスクを使用。補間用のプロンプトを入力できてもよいかも
                                        prompt_poolers=clip_l_pooler,  # TODO:メインプロンプトを使用。補間用のプロンプトを入力できてもよいかも
                                        negative_prompt_embeds=llama_vec_n,
                                        negative_prompt_embeds_mask=llama_attention_mask_n,
                                        negative_prompt_poolers=clip_l_pooler_n,
                                        device=gpu,
                                        dtype=torch.bfloat16,
                                        image_embeddings=image_encoder_last_hidden_state,
                                        latent_indices=latent_indices_2,
                                        clean_latents=clean_latents_2,
                                        clean_latent_indices=clean_latent_indices_2,
                                        clean_latents_2x=clean_latents_2x_2,
                                        clean_latent_2x_indices=clean_latent_2x_indices_2,
                                        clean_latents_4x=clean_latents_4x_2,
                                        clean_latent_4x_indices=clean_latent_4x_indices_2,
                                        callback=callback_interpolation,
                                    )

                                    # 補間された潜在変数を結合
                                    device = real_history_latents.device
                                    generated_interpolation_latents = generated_interpolation_latents.to(device)
                                    uploaded_tensor = uploaded_tensor.to(device)
                                    real_history_latents = torch.cat([real_history_latents, generated_interpolation_latents, uploaded_tensor], dim=2)

                                    # VAEキャッシュ設定に応じてデコード関数を切り替え
                                    if not high_vram:
                                        load_model_as_complete(vae, target_device=gpu)
                                    if use_vae_cache:
                                        interpolation_pixels = vae_decode_cache(generated_interpolation_latents[:, :, :], vae).cpu()
                                    else:
                                        interpolation_pixels = vae_decode(generated_interpolation_latents[:, :, :], vae).cpu()

                                    # overlapは小さめに固定
                                    interpolation_overlapped_frames = 2
                                    if interpolation_overlapped_frames > interpolation_pixels.shape[2]:
                                        interpolation_overlapped_frames = interpolation_pixels.shape[2]

                                    history_pixels = soft_append_bcthw(history_pixels, interpolation_pixels, interpolation_overlapped_frames)
                                    print(translate("新規生成データの末尾、補間データ、テンソルデータの先頭のフレームを結合しました。"))

                                    # 補間データをmp4にして出力
                                    interpolation_output_filename = os.path.join(outputs_folder, f'{job_id}_interpolation.mp4')
                                    save_bcthw_as_mp4(interpolation_pixels, interpolation_output_filename, fps=30, crf=mp4_crf)

                            # デバイスとデータ型を合わせる
                            processed_tensor = uploaded_tensor.clone()
                            if processed_tensor.device != real_history_latents.device:
                                processed_tensor = processed_tensor.to(real_history_latents.device)
                            if processed_tensor.dtype != real_history_latents.dtype:
                                processed_tensor = processed_tensor.to(dtype=real_history_latents.dtype)

                            # 元の動画を品質を保ちつつ保存
                            original_output_filename = os.path.join(outputs_folder, f'{job_id}_original.mp4')
                            save_bcthw_as_mp4(history_pixels, original_output_filename, fps=30, crf=mp4_crf)
                            print(translate("元の動画を保存しました: {original_output_filename}").format(original_output_filename=original_output_filename))

                            # 元データのコピーを取得
                            combined_history_latents = real_history_latents.clone()
                            combined_history_pixels = history_pixels.clone() if history_pixels is not None else None

                            # 各チャンクの処理前に明示的にメモリ解放
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                print(translate("[MEMORY] チャンク処理前のGPUメモリ確保状態: {memory:.2f}GB").format(memory=torch.cuda.memory_allocated()/1024**3))

                            # VAEをGPUに移動
                            if not high_vram and vae.device != torch.device('cuda'):
                                print(translate("[SETUP] VAEをGPUに移動: {0} → cuda").format(vae.device))
                                vae.to('cuda')

                            # 各チャンクを処理
                            # チャンクサイズを設定(各セクションと同等のサイズにする)
                            chunk_size = min(5, uploaded_frames)  # 最大チャンクサイズを5フレームに設定（メモリ使用量を減らすため）

                            # チャンク数を計算
                            num_chunks = (uploaded_frames + chunk_size - 1) // chunk_size

                            # テンソルデータの詳細を出力
                            print(translate("[DEBUG] テンソルデータの詳細分析:"))
                            print(translate("  - 形状: {0}").format(processed_tensor.shape))
                            print(translate("  - 型: {0}").format(processed_tensor.dtype))
                            print(translate("  - デバイス: {0}").format(processed_tensor.device))
                            print(translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(processed_tensor.min().item(), processed_tensor.max().item(), processed_tensor.mean().item()))
                            print(translate("  - チャンク数: {0}, チャンクサイズ: {1}").format(num_chunks, chunk_size))
                            tensor_size_mb = (processed_tensor.element_size() * processed_tensor.nelement()) / (1024 * 1024)
                            print(translate("  - テンソルデータ全体サイズ: {0:.2f} MB").format(tensor_size_mb))
                            print(translate("  - フレーム数: {0}フレーム（制限無し）").format(uploaded_frames))
                            # 各チャンクを処理
                            for chunk_idx in range(num_chunks):
                                chunk_start = chunk_idx * chunk_size
                                chunk_end = min(chunk_start + chunk_size, uploaded_frames)
                                chunk_frames = chunk_end - chunk_start

                                # 進捗状況を更新
                                chunk_progress = (chunk_idx + 1) / num_chunks * 100
                                progress_message = translate("テンソルデータ結合中: チャンク {0}/{1} (フレーム {2}-{3}/{4})").format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames)
                                stream.output_queue.push(('progress', (None, progress_message, make_progress_bar_html(int(80 + chunk_progress * 0.1), translate('テンソルデータ処理中')))))

                                # 現在のチャンクを取得
                                current_chunk = processed_tensor[:, :, chunk_start:chunk_end, :, :]
                                print(translate("チャンク{0}/{1}処理中: フレーム {2}-{3}/{4}").format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames))

                                # メモリ状態を出力
                                if torch.cuda.is_available():
                                    print(translate("[MEMORY] チャンク{0}処理前のGPUメモリ: {1:.2f}GB/{2:.2f}GB").format(chunk_idx+1, torch.cuda.memory_allocated()/1024**3, torch.cuda.get_device_properties(0).total_memory/1024**3))
                                    # メモリキャッシュをクリア
                                    torch.cuda.empty_cache()

                                try:
                                    # 各チャンク処理前にGPUメモリを解放
                                    if torch.cuda.is_available():
                                        torch.cuda.synchronize()
                                        torch.cuda.empty_cache()
                                        import gc
                                        gc.collect()
                                    # チャンクをデコード
                                    # VAEデコードは時間がかかるため、進行中であることを表示
                                    print(translate("チャンク{0}のVAEデコード開始...").format(chunk_idx+1))
                                    stream.output_queue.push(('progress', (None, translate("チャンク{0}/{1}のVAEデコード中...").format(chunk_idx+1, num_chunks), make_progress_bar_html(int(80 + chunk_progress * 0.1), translate('デコード処理')))))

                                    # VAEデコード前にテンソル情報を詳しく出力
                                    print(translate("[DEBUG] チャンク{0}のデコード前情報:").format(chunk_idx+1))
                                    print(translate("  - 形状: {0}").format(current_chunk.shape))
                                    print(translate("  - 型: {0}").format(current_chunk.dtype))
                                    print(translate("  - デバイス: {0}").format(current_chunk.device))
                                    print(translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(current_chunk.min().item(), current_chunk.max().item(), current_chunk.mean().item()))

                                    # 明示的にデバイスを合わせる
                                    if current_chunk.device != vae.device:
                                        print(translate("  - デバイスをVAEと同じに変更: {0} → {1}").format(current_chunk.device, vae.device))
                                        current_chunk = current_chunk.to(vae.device)

                                    # 型を明示的に合わせる
                                    if current_chunk.dtype != torch.float16:
                                        print(translate("  - データ型をfloat16に変更: {0} → torch.float16").format(current_chunk.dtype))
                                        current_chunk = current_chunk.to(dtype=torch.float16)

                                    # VAEデコード処理 - VAEキャッシュ設定に応じて関数を切り替え
                                    print(f"[DEBUG] チャンクデコード前のVAEキャッシュ設定: {use_vae_cache}, 型: {type(use_vae_cache)}")
                                    if use_vae_cache:
                                        print(translate("[INFO] VAEキャッシュを使用: チャンク{0}").format(chunk_idx+1))
                                        chunk_pixels = vae_decode_cache(current_chunk, vae).cpu()
                                    else:
                                        print(translate("[INFO] 通常デコード使用: チャンク{0}").format(chunk_idx+1))
                                        chunk_pixels = vae_decode(current_chunk, vae).cpu()
                                    print(translate("チャンク{0}のVAEデコード完了 (フレーム数: {1})").format(chunk_idx+1, chunk_frames))

                                    # デコード後のピクセルデータ情報を出力
                                    print(translate("[DEBUG] チャンク{0}のデコード結果:").format(chunk_idx+1))
                                    print(translate("  - 形状: {0}").format(chunk_pixels.shape))
                                    print(translate("  - 型: {0}").format(chunk_pixels.dtype))
                                    print(translate("  - デバイス: {0}").format(chunk_pixels.device))
                                    print(translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(chunk_pixels.min().item(), chunk_pixels.max().item(), chunk_pixels.mean().item()))

                                    # メモリ使用量を出力
                                    if torch.cuda.is_available():
                                        print(translate("[MEMORY] チャンク{0}デコード後のGPUメモリ: {1:.2f}GB").format(chunk_idx+1, torch.cuda.memory_allocated()/1024**3))

                                    # 結合する
                                    if combined_history_pixels is None:
                                        # 初回のチャンクの場合はそのまま設定
                                        combined_history_pixels = chunk_pixels
                                    else:
                                        # 2回目以降は結合
                                        print(translate("[DEBUG] 結合前の情報:"))
                                        print(translate("  - 既存: {0}, 型: {1}, デバイス: {2}").format(combined_history_pixels.shape, combined_history_pixels.dtype, combined_history_pixels.device))
                                        print(translate("  - 新規: {0}, 型: {1}, デバイス: {2}").format(chunk_pixels.shape, chunk_pixels.dtype, chunk_pixels.device))

                                        # 既存データと新規データで型とデバイスを揃える
                                        if combined_history_pixels.dtype != chunk_pixels.dtype:
                                            print(translate("  - データ型の不一致を修正: {0} → {1}").format(combined_history_pixels.dtype, chunk_pixels.dtype))
                                            combined_history_pixels = combined_history_pixels.to(dtype=chunk_pixels.dtype)

                                        # 両方とも必ずCPUに移動してから結合
                                        if combined_history_pixels.device != torch.device('cpu'):
                                            combined_history_pixels = combined_history_pixels.cpu()
                                        if chunk_pixels.device != torch.device('cpu'):
                                            chunk_pixels = chunk_pixels.cpu()

                                        # 結合処理
                                        combined_history_pixels = torch.cat([combined_history_pixels, chunk_pixels], dim=2)

                                    # 結合後のフレーム数を確認
                                    current_total_frames = combined_history_pixels.shape[2]
                                    print(translate("チャンク{0}の結合完了: 現在の組み込みフレーム数 = {1}").format(chunk_idx+1, current_total_frames))

                                    # 中間結果の保存（チャンクごとに保存すると効率が悪いので、最終チャンクのみ保存）
                                    if chunk_idx == num_chunks - 1 or (chunk_idx > 0 and (chunk_idx + 1) % 5 == 0):
                                        # 5チャンクごと、または最後のチャンクで保存
                                        interim_output_filename = os.path.join(outputs_folder, f'{job_id}_combined_interim_{chunk_idx+1}.mp4')
                                        print(translate("中間結果を保存中: チャンク{0}/{1}").format(chunk_idx+1, num_chunks))
                                        stream.output_queue.push(('progress', (None, translate("中間結果のMP4変換中... (チャンク{0}/{1})").format(chunk_idx+1, num_chunks), make_progress_bar_html(int(85 + chunk_progress * 0.1), translate('MP4保存中')))))

                                        # MP4として保存
                                        save_bcthw_as_mp4(combined_history_pixels, interim_output_filename, fps=30, crf=mp4_crf)
                                        print(translate("中間結果を保存しました: {0}").format(interim_output_filename))

                                        # 結合した動画をUIに反映するため、出力フラグを立てる
                                        stream.output_queue.push(('file', interim_output_filename))
                                except Exception as e:
                                    print(translate("チャンク{0}の処理中にエラーが発生しました: {1}").format(chunk_idx+1, e))
                                    traceback.print_exc()

                                    # エラー情報の詳細な出力
                                    print(translate("[ERROR] 詳細エラー情報:"))
                                    print(translate("  - チャンク情報: {0}/{1}, フレーム {2}-{3}/{4}").format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames))
                                    if 'current_chunk' in locals():
                                        print(translate("  - current_chunk: shape={0}, dtype={1}, device={2}").format(current_chunk.shape, current_chunk.dtype, current_chunk.device))
                                    if 'vae' in globals():
                                        print(translate("  - VAE情報: device={0}, dtype={1}").format(vae.device, next(vae.parameters()).dtype))

                                    # GPUメモリ情報
                                    if torch.cuda.is_available():
                                        print(translate("  - GPU使用量: {0:.2f}GB/{1:.2f}GB").format(torch.cuda.memory_allocated()/1024**3, torch.cuda.get_device_properties(0).total_memory/1024**3))

                                    stream.output_queue.push(('progress', (None, translate("エラー: チャンク{0}の処理に失敗しました - {1}").format(chunk_idx+1, str(e)), make_progress_bar_html(90, translate('エラー')))))
                                    break

                            # 処理完了後に明示的にメモリ解放
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                print(translate("[MEMORY] チャンク処理後のGPUメモリ確保状態: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))

                            # 全チャンクの処理が完了したら、最終的な結合動画を保存
                            if combined_history_pixels is not None:
                                # 結合された最終結果の情報を出力
                                print(translate("[DEBUG] 最終結合結果:"))
                                print(translate("  - 形状: {0}").format(combined_history_pixels.shape))
                                print(translate("  - 型: {0}").format(combined_history_pixels.dtype))
                                print(translate("  - デバイス: {0}").format(combined_history_pixels.device))
                                # 最終結果の保存
                                print(translate("最終結果を保存中: 全{0}チャンク完了").format(num_chunks))
                                stream.output_queue.push(('progress', (None, translate("結合した動画をMP4に変換中..."), make_progress_bar_html(95, translate('最終MP4変換処理')))))

                                # 最終的な結合ファイル名
                                combined_output_filename = os.path.join(outputs_folder, f'{job_id}_combined.mp4')

                                # MP4として保存
                                save_bcthw_as_mp4(combined_history_pixels, combined_output_filename, fps=30, crf=mp4_crf)
                                print(translate("最終結果を保存しました: {0}").format(combined_output_filename))
                                print(translate("結合動画の保存場所: {0}").format(os.path.abspath(combined_output_filename)))

                                # 中間ファイルの削除処理
                                print(translate("中間ファイルの削除を開始します..."))
                                deleted_files = []
                                try:
                                    # 現在のジョブIDに関連する中間ファイルを正規表現でフィルタリング
                                    import re
                                    interim_pattern = re.compile(f'{job_id}_combined_interim_\d+\.mp4')

                                    for filename in os.listdir(outputs_folder):
                                        if interim_pattern.match(filename):
                                            interim_path = os.path.join(outputs_folder, filename)
                                            try:
                                                os.remove(interim_path)
                                                deleted_files.append(filename)
                                                print(translate("  - 中間ファイルを削除しました: {0}").format(filename))
                                            except Exception as e:
                                                print(translate("  - ファイル削除エラー ({0}): {1}").format(filename, e))

                                    if deleted_files:
                                        print(translate("合計 {0} 個の中間ファイルを削除しました").format(len(deleted_files)))
                                        # 削除ファイル名をユーザーに表示
                                        files_str = ', '.join(deleted_files)
                                        stream.output_queue.push(('progress', (None, translate("中間ファイルを削除しました: {0}").format(files_str), make_progress_bar_html(97, translate('クリーンアップ完了')))))
                                    else:
                                        print(translate("削除対象の中間ファイルは見つかりませんでした"))
                                except Exception as e:
                                    print(translate("中間ファイル削除中にエラーが発生しました: {0}").format(e))
                                    traceback.print_exc()

                                # 結合した動画をUIに反映するため、出力フラグを立てる
                                stream.output_queue.push(('file', combined_output_filename))

                                # 結合後の全フレーム数を計算して表示
                                combined_frames = combined_history_pixels.shape[2]
                                combined_size_mb = (combined_history_pixels.element_size() * combined_history_pixels.nelement()) / (1024 * 1024)
                                print(translate("結合完了情報: テンソルデータ({0}フレーム) + 新規動画({1}フレーム) = 合計{2}フレーム").format(uploaded_frames, original_frames, combined_frames))
                                print(translate("結合動画の再生時間: {0:.2f}秒").format(combined_frames / 30))
                                print(translate("データサイズ: {0:.2f} MB（制限無し）").format(combined_size_mb))

                                # UI上で完了メッセージを表示
                                stream.output_queue.push(('progress', (None, translate("テンソルデータ({0}フレーム)と動画({1}フレーム)の結合が完了しました。\n合計フレーム数: {2}フレーム ({3:.2f}秒) - サイズ制限なし").format(uploaded_frames, original_frames, combined_frames, combined_frames / 30), make_progress_bar_html(100, translate('結合完了')))))
                            else:
                                print(translate("テンソルデータの結合に失敗しました。"))
                                stream.output_queue.push(('progress', (None, translate("テンソルデータの結合に失敗しました。"), make_progress_bar_html(100, translate('エラー')))))

                            # 正しく結合された動画はすでに生成済みなので、ここでの処理は不要

                            # この部分の処理はすでに上記のチャンク処理で完了しているため不要

                            # real_history_latentsとhistory_pixelsを結合済みのものに更新
                            real_history_latents = combined_history_latents
                            history_pixels = combined_history_pixels

                            # 結合した動画をUIに反映するため、出力フラグを立てる
                            stream.output_queue.push(('file', combined_output_filename))

                            # 出力ファイル名を更新
                            output_filename = combined_output_filename

                            # 結合後の全フレーム数を計算して表示
                            combined_frames = combined_history_pixels.shape[2]
                            combined_size_mb = (combined_history_pixels.element_size() * combined_history_pixels.nelement()) / (1024 * 1024)
                            print(translate("結合完了情報: テンソルデータ({0}フレーム) + 新規動画({1}フレーム) = 合計{2}フレーム").format(uploaded_frames, original_frames, combined_frames))
                            print(translate("結合動画の再生時間: {0:.2f}秒").format(combined_frames / 30))
                            print(translate("データサイズ: {0:.2f} MB（制限無し）").format(combined_size_mb))

                            # UI上で完了メッセージを表示
                            stream.output_queue.push(('progress', (None, translate("テンソルデータ({0}フレーム)と動画({1}フレーム)の結合が完了しました。\n合計フレーム数: {2}フレーム ({3:.2f}秒)").format(uploaded_frames, original_frames, combined_frames, combined_frames / 30), make_progress_bar_html(100, translate('結合完了')))))
                    except Exception as e:
                        print(translate("テンソルデータ結合中にエラーが発生しました: {0}").format(e))
                        traceback.print_exc()
                        stream.output_queue.push(('progress', (None, translate("エラー: テンソルデータ結合に失敗しました - {0}").format(str(e)), make_progress_bar_html(100, translate('エラー')))))

                # 処理終了時に通知
                if HAS_WINSOUND:
                    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                else:
                    print(translate("\n✓ 処理が完了しました！"))  # Linuxでの代替通知

                # メモリ解放を明示的に実行
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    print(translate("[MEMORY] 処理完了後のメモリクリア: {memory:.2f}GB/{total_memory:.2f}GB").format(memory=torch.cuda.memory_allocated()/1024**3, total_memory=torch.cuda.get_device_properties(0).total_memory/1024**3))

                # テンソルデータの保存処理
                print(translate("[DEBUG] テンソルデータ保存フラグの値: {0}").format(save_tensor_data))
                if save_tensor_data:
                    try:
                        # 結果のテンソルを保存するファイルパス
                        tensor_file_path = os.path.join(outputs_folder, f'{job_id}.safetensors')

                        # 保存するデータを準備
                        print(translate("=== テンソルデータ保存処理開始 ==="))
                        print(translate("保存対象フレーム数: {frames}").format(frames=real_history_latents.shape[2]))

                        # サイズ制限を完全に撤廃し、全フレームを保存
                        tensor_to_save = real_history_latents.clone().cpu()

                        # テンソルデータの保存サイズの概算
                        tensor_size_mb = (tensor_to_save.element_size() * tensor_to_save.nelement()) / (1024 * 1024)

                        print(translate("テンソルデータを保存中... shape: {shape}, フレーム数: {frames}, サイズ: {size:.2f} MB").format(shape=tensor_to_save.shape, frames=tensor_to_save.shape[2], size=tensor_size_mb))
                        stream.output_queue.push(('progress', (None, translate('テンソルデータを保存中... ({frames}フレーム)').format(frames=tensor_to_save.shape[2]), make_progress_bar_html(95, translate('テンソルデータの保存')))))

                        # メタデータの準備（フレーム数も含める）
                        metadata = torch.tensor([height, width, tensor_to_save.shape[2]], dtype=torch.int32)

                        # safetensors形式で保存
                        tensor_dict = {
                            "history_latents": tensor_to_save,
                            "metadata": metadata
                        }
                        sf.save_file(tensor_dict, tensor_file_path)

                        print(translate("テンソルデータを保存しました: {path}").format(path=tensor_file_path))
                        print(translate("保存済みテンソルデータ情報: {frames}フレーム, {size:.2f} MB").format(frames=tensor_to_save.shape[2], size=tensor_size_mb))
                        print(translate("=== テンソルデータ保存処理完了 ==="))
                        stream.output_queue.push(('progress', (None, translate("テンソルデータが保存されました: {path} ({frames}フレーム, {size:.2f} MB)").format(path=os.path.basename(tensor_file_path), frames=tensor_to_save.shape[2], size=tensor_size_mb), make_progress_bar_html(100, translate('処理完了')))))

                        # アップロードされたテンソルデータがあれば、それも結合したものを保存する
                        if tensor_data_input is not None and uploaded_tensor is not None:
                            try:
                                # アップロードされたテンソルデータのファイル名を取得
                                uploaded_tensor_filename = os.path.basename(tensor_data_input.name)
                                tensor_combined_path = os.path.join(outputs_folder, f'{job_id}_combined_tensors.safetensors')

                                print(translate("=== テンソルデータ結合処理開始 ==="))
                                print(translate("生成テンソルと入力テンソルを結合して保存します"))
                                print(translate("生成テンソル: {frames}フレーム").format(frames=tensor_to_save.shape[2]))
                                print(translate("入力テンソル: {frames}フレーム").format(frames=uploaded_tensor.shape[2]))

                                # データ型とデバイスを統一
                                if uploaded_tensor.dtype != tensor_to_save.dtype:
                                    uploaded_tensor = uploaded_tensor.to(dtype=tensor_to_save.dtype)
                                if uploaded_tensor.device != tensor_to_save.device:
                                    uploaded_tensor = uploaded_tensor.to(device=tensor_to_save.device)

                                # サイズチェック
                                if uploaded_tensor.shape[3] != tensor_to_save.shape[3] or uploaded_tensor.shape[4] != tensor_to_save.shape[4]:
                                    print(translate("警告: テンソルサイズが一致しないため結合できません: {uploaded_shape} vs {tensor_shape}").format(uploaded_shape=uploaded_tensor.shape, tensor_shape=tensor_to_save.shape))
                                else:
                                    # 結合（生成テンソルの後にアップロードされたテンソルを追加）
                                    combined_tensor = torch.cat([tensor_to_save, uploaded_tensor], dim=2)
                                    combined_frames = combined_tensor.shape[2]
                                    combined_size_mb = (combined_tensor.element_size() * combined_tensor.nelement()) / (1024 * 1024)

                                    # メタデータ更新
                                    combined_metadata = torch.tensor([height, width, combined_frames], dtype=torch.int32)

                                    # 結合したテンソルを保存
                                    combined_tensor_dict = {
                                        "history_latents": combined_tensor,
                                        "metadata": combined_metadata
                                    }
                                    sf.save_file(combined_tensor_dict, tensor_combined_path)

                                    print(translate("結合テンソルを保存しました: {path}").format(path=tensor_combined_path))
                                    print(translate("結合テンソル情報: 合計{0}フレーム ({1}+{2}), {3:.2f} MB").format(combined_frames, tensor_to_save.shape[2], uploaded_tensor.shape[2], combined_size_mb))
                                    print(translate("=== テンソルデータ結合処理完了 ==="))
                                    stream.output_queue.push(('progress', (None, translate("テンソルデータ結合が保存されました: 合計{frames}フレーム").format(frames=combined_frames), make_progress_bar_html(100, translate('結合テンソル保存完了')))))
                            except Exception as e:
                                print(translate("テンソルデータ結合保存エラー: {0}").format(e))
                                traceback.print_exc()
                    except Exception as e:
                        print(translate("テンソルデータ保存エラー: {0}").format(e))
                        traceback.print_exc()
                        stream.output_queue.push(('progress', (None, translate("テンソルデータの保存中にエラーが発生しました。"), make_progress_bar_html(100, translate('処理完了')))))

                # 全体の処理時間を計算
                process_end_time = time.time()
                total_process_time = process_end_time - process_start_time
                hours, remainder = divmod(total_process_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = ""
                if hours > 0:
                    time_str = translate("{0}時間 {1}分 {2}秒").format(int(hours), int(minutes), f"{seconds:.1f}")
                elif minutes > 0:
                    time_str = translate("{0}分 {1}秒").format(int(minutes), f"{seconds:.1f}")
                else:
                    time_str = translate("{0:.1f}秒").format(seconds)
                print(translate("\n全体の処理時間: {0}").format(time_str))

                # 完了メッセージの設定（結合有無によって変更）
                if combined_output_filename is not None:
                    # テンソル結合が成功した場合のメッセージ
                    combined_filename_only = os.path.basename(combined_output_filename)
                    completion_message = translate("すべてのセクション({sections}/{total_sections})が完了しました。テンソルデータとの後方結合も完了しました。結合ファイル名: {filename}\n全体の処理時間: {time}").format(sections=total_sections, total_sections=total_sections, filename=combined_filename_only, time=time_str)
                    # 最終的な出力ファイルを結合したものに変更
                    output_filename = combined_output_filename
                else:
                    # 通常の完了メッセージ
                    completion_message = translate("すべてのセクション({sections}/{total_sections})が完了しました。全体の処理時間: {time}").format(sections=total_sections, total_sections=total_sections, time=time_str)

                stream.output_queue.push(('progress', (None, completion_message, make_progress_bar_html(100, translate('処理完了')))))

                # 中間ファイルの削除処理
                if not keep_section_videos:
                    # 最終動画のフルパス
                    final_video_path = output_filename
                    final_video_name = os.path.basename(final_video_path)
                    # job_id部分を取得（タイムスタンプ部分）
                    job_id_part = job_id

                    # ディレクトリ内のすべてのファイルを取得
                    files = os.listdir(outputs_folder)
                    deleted_count = 0

                    for file in files:
                        # 同じjob_idを持つMP4ファイルかチェック
                        # 結合ファイル('combined'を含む)は消さないように保護
                        if file.startswith(job_id_part) and file.endswith('.mp4') \
                           and file != final_video_name \
                           and 'combined' not in file:  # combinedファイルは保護
                            file_path = os.path.join(outputs_folder, file)
                            try:
                                os.remove(file_path)
                                deleted_count += 1
                                print(translate("[削除] 中間ファイル: {0}").format(file))
                            except Exception as e:
                                print(translate("[エラー] ファイル削除時のエラー {0}: {1}").format(file, e))

                    if deleted_count > 0:
                        print(translate("[済] {0}個の中間ファイルを削除しました。最終ファイルは保存されています: {1}").format(deleted_count, final_video_name))
                        final_message = translate("中間ファイルを削除しました。最終動画と結合動画は保存されています。")
                        stream.output_queue.push(('progress', (None, final_message, make_progress_bar_html(100, translate('処理完了')))))

                break

            if not high_vram:
                unload_complete_models()
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return

# 画像のバリデーション関数
def validate_images(input_image, section_settings, length_radio=None, frame_size_radio=None):
    """入力画像または画面に表示されている最後のキーフレーム画像のいずれかが有効かを確認する"""
    # 入力画像をチェック
    if input_image is not None:
        return True, ""

    # 現在の設定から表示すべきセクション数を計算
    total_display_sections = None
    if length_radio is not None and frame_size_radio is not None:
        try:
            # 動画長を秒数で取得
            seconds = get_video_seconds(length_radio.value)

            # フレームサイズ設定からlatent_window_sizeを計算
            latent_window_size = 4.5 if frame_size_radio.value == translate("0.5秒 (17フレーム)") else 9
            frame_count = latent_window_size * 4 - 3

            # セクション数を計算
            total_frames = int(seconds * 30)
            total_display_sections = int(max(round(total_frames / frame_count), 1))
            print(translate("[DEBUG] 現在の設定によるセクション数: {0}").format(total_display_sections))
        except Exception as e:
            print(translate("[ERROR] セクション数計算エラー: {0}").format(e))

    # 入力画像がない場合、表示されているセクションの中で最後のキーフレーム画像をチェック
    last_visible_section_image = None
    last_visible_section_num = -1

    if section_settings is not None and not isinstance(section_settings, bool):
        # 有効なセクション番号を収集
        valid_sections = []
        try:
            for section in section_settings:
                if section and len(section) > 1 and section[0] is not None:
                    try:
                        section_num = int(section[0])
                        # 表示セクション数が計算されていれば、それ以下のセクションのみ追加
                        if total_display_sections is None or section_num < total_display_sections:
                            valid_sections.append((section_num, section[1]))
                    except (ValueError, TypeError):
                        continue
        except (TypeError, ValueError):
            # section_settingsがイテラブルでない場合（ブール値など）、空のリストとして扱う
            valid_sections = []
            print(f"[DEBUG] section_settings is not iterable: {type(section_settings)}")


        # 有効なセクションがあれば、最大の番号（最後のセクション）を探す
        if valid_sections:
            # 番号でソート
            valid_sections.sort(key=lambda x: x[0])
            # 最後のセクションを取得
            last_visible_section_num, last_visible_section_image = valid_sections[-1]

            print(translate("[DEBUG] 最後のキーフレーム確認: セクション{0} (画像あり: {1})").format(last_visible_section_num, last_visible_section_image is not None))

    # 最後のキーフレーム画像があればOK
    if last_visible_section_image is not None:
        return True, ""

    # どちらの画像もない場合はエラー
    error_html = f"""
    <div style="padding: 15px; border-radius: 10px; background-color: #ffebee; border: 1px solid #f44336; margin: 10px 0;">
        <h3 style="color: #d32f2f; margin: 0 0 10px 0;">{translate('❗️ 画像が選択されていません')}</h3>
        <p>{translate('生成を開始する前に「Image」欄または表示されている最後のキーフレーム画像に画像をアップロードしてください。これはあまねく叡智の始発点となる重要な画像です。')}</p>
    </div>
    """
    error_bar = make_progress_bar_html(100, translate('画像がありません'))
    return False, error_html + error_bar

def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_random_seed, mp4_crf=16, all_padding_value=1.0, end_frame=None, end_frame_strength=1.0, frame_size_setting="1秒 (33フレーム)", keep_section_videos=False, lora_files=None, lora_files2=None, lora_files3=None, lora_scales_text="0.8,0.8,0.8", output_dir=None, save_section_frames=False, section_settings=None, use_all_padding=False, use_lora=False, lora_mode=None, lora_dropdown1=None, lora_dropdown2=None, lora_dropdown3=None, save_tensor_data=False, tensor_data_input=None, fp8_optimization=False, resolution=640, batch_count=1, save_latent_frames=False, save_last_section_frames=False, use_vae_cache=False, use_queue=False, prompt_queue_file=None):
    # プロセス関数の最初でVAEキャッシュ設定を確認
    print(f"process関数開始時のVAEキャッシュ設定: {use_vae_cache}, 型: {type(use_vae_cache)}")
    global stream
    global batch_stopped
    global queue_enabled
    global queue_type
    global prompt_queue_file_path
    global vae_cache_enabled
    global image_queue_files

    # バッチ処理開始時に停止フラグをリセット
    batch_stopped = False

    # バリデーション関数で既にチェック済みなので、ここでの再チェックは不要

    # フレームサイズ設定に応じてlatent_window_sizeを先に調整
    if frame_size_setting == "0.5秒 (17フレーム)":
        # 0.5秒の場合はlatent_window_size=4.5に設定（実際には4.5*4-3=17フレーム≒0.5秒@30fps）
        latent_window_size = 4.5
        print(translate('フレームサイズを0.5秒モードに設定: latent_window_size = {0}').format(latent_window_size))
    else:
        # デフォルトの1秒モードではlatent_window_size=9を使用（9*4-3=33フレーム≒1秒@30fps）
        latent_window_size = 9
        print(translate('フレームサイズを1秒モードに設定: latent_window_size = {0}').format(latent_window_size))

    # バッチ処理回数を確認し、詳細を出力
    # 型チェックしてから変換（数値でない場合はデフォルト値の1を使用）
    try:
        batch_count_val = int(batch_count)
        batch_count = max(1, min(batch_count_val, 100))  # 1〜100の間に制限
    except (ValueError, TypeError):
        print(translate("[WARN] バッチ処理回数が無効です。デフォルト値の1を使用します: {0}").format(batch_count))
        batch_count = 1  # デフォルト値
    # プロンプトキューが有効な場合は行数も表示
    if bool(use_queue) and queue_type == "prompt" and prompt_queue_file_path is not None and os.path.exists(prompt_queue_file_path):
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
        except Exception:
            print(translate("\u25c6 バッチ処理回数: {0}回").format(batch_count))
    else:
        print(translate("\u25c6 バッチ処理回数: {0}回").format(batch_count))

    # 解像度を安全な値に丸めてログ表示
    from diffusers_helper.bucket_tools import SAFE_RESOLUTIONS

    # 解像度値を表示
    print(translate("\u25c6 UIから受け取った解像度値: {0}（型: {1}）").format(resolution, type(resolution).__name__))

    # 安全な値に丸める
    if resolution not in SAFE_RESOLUTIONS:
        closest_resolution = min(SAFE_RESOLUTIONS, key=lambda x: abs(x - resolution))
        print(translate('安全な解像度値ではないため、{0}から{1}に自動調整しました').format(resolution, closest_resolution))
        resolution = closest_resolution

    # 解像度設定を出力
    print(translate('解像度を設定: {0}').format(resolution))

    # 動画生成の設定情報をログに出力
    # 4.5の場合は5として計算するための特別処理
    if latent_window_size == 4.5:
        frame_count = 17  # 5 * 4 - 3 = 17
    else:
        frame_count = int(latent_window_size * 4 - 3)
    total_latent_sections = int(max(round((total_second_length * 30) / frame_count), 1))

    mode_name = translate("通常モード") if mode_radio.value == MODE_TYPE_NORMAL else translate("ループモード")

    print(translate("\n==== 動画生成開始 ====="))
    print(translate("\u25c6 生成モード: {0}").format(mode_name))
    print(translate("\u25c6 動画長: {0}秒").format(total_second_length))
    print(translate("\u25c6 フレームサイズ: {0}").format(frame_size_setting))
    print(translate("\u25c6 生成セクション数: {0}回").format(total_latent_sections))
    print(translate("\u25c6 サンプリングステップ数: {0}").format(steps))
    print(translate("\u25c6 TeaCache使用: {0}").format(use_teacache))
    # TeaCache使用の直後にSEED値の情報を表示 - バッチ処理の初期値として表示
    # 実際のバッチ処理では各バッチでSEED値が変わる可能性があるため、「初期SEED値」として表示
    print(translate("\u25c6 初期SEED値: {0}").format(seed))
    print(translate("\u25c6 LoRA使用: {0}").format(use_lora))

    # FP8最適化設定のログ出力
    print(translate("\u25c6 FP8最適化: {0}").format(fp8_optimization))
    
    # VAEキャッシュ設定のログ出力
    print(translate("\u25c6 VAEキャッシュ: {0}").format(use_vae_cache))
    print(f"VAEキャッシュ詳細状態: use_vae_cache={use_vae_cache}, type={type(use_vae_cache)}")

    # オールパディング設定のログ出力
    if use_all_padding:
        print(translate("\u25c6 オールパディング: 有効 (値: {0})").format(round(all_padding_value, 1)))
    else:
        print(translate("\u25c6 オールパディング: 無効"))

    # LoRA情報のログ出力とLoRAモード判定
    # LoRAモードがディレクトリ選択で、ドロップダウンに値が選択されている場合は使用フラグを上書き
    if lora_mode == translate("ディレクトリから選択") and has_lora_support:
        # ディレクトリからドロップダウンで選択されたLoRAが1つでもあるか確認
        has_selected_lora = False
        for dropdown in [lora_dropdown1, lora_dropdown2, lora_dropdown3]:
            dropdown_value = dropdown.value if hasattr(dropdown, 'value') else dropdown
            if dropdown_value and dropdown_value != translate("なし"):
                has_selected_lora = True
                break
        
        # LoRA選択があれば強制的に有効にする
        if has_selected_lora:
            use_lora = True
            print(translate("[INFO] ディレクトリでLoRAが選択されているため、LoRA使用を有効にしました"))
    
    if use_lora and has_lora_support:
        all_lora_files = []
        lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
        
        # UI状態からモードを検出 (lora_modeがgr.Radioオブジェクトの場合、valueプロパティを使用)
        lora_mode_value = lora_mode.value if hasattr(lora_mode, 'value') else lora_mode
        if lora_mode_value and lora_mode_value == translate("ディレクトリから選択"):
            # ディレクトリから選択モード
            print(translate("[INFO] LoRA読み込み方式: ディレクトリから選択"))
            
            # ドロップダウンから選択されたファイルを処理 (ログ出力用)
            selected_lora_names = []
            
            # 各ドロップダウンを確認
            for dropdown, dropdown_name in [(lora_dropdown1, "LoRA1"), (lora_dropdown2, "LoRA2"), (lora_dropdown3, "LoRA3")]:
                # ドロップダウンの値を取得（gr.Dropdownオブジェクトの場合はvalueプロパティを使用）
                dropdown_value = dropdown.value if hasattr(dropdown, 'value') else dropdown
                
                # 通常の値が0や0.0などの数値の場合の特別処理（GradioのUIの問題によるもの）
                if dropdown_value == 0 or dropdown_value == "0" or dropdown_value == 0.0:
                    # 数値の0を"なし"として扱う
                    print(translate("[DEBUG] 情報表示: {name}の値が数値0として検出されました。'なし'として扱います").format(name=dropdown_name))
                    dropdown_value = translate("なし")
                
                # 型チェックと文字列変換を追加
                if not isinstance(dropdown_value, str) and dropdown_value is not None:
                    print(translate("[DEBUG] 情報表示: {name}の値のタイプ変換が必要: {type}").format(name=dropdown_name, type=type(dropdown_value).__name__))
                    dropdown_value = str(dropdown_value)
                
                if dropdown_value and dropdown_value != translate("なし"):
                    lora_path = os.path.join(lora_dir, dropdown_value)
                    # よりわかりやすい表記に
                    model_name = f"LoRA{dropdown_name[-1]}: {dropdown_value}"
                    selected_lora_names.append(model_name)
            
            # 選択されたLoRAモデルの情報出力を明確に
            if selected_lora_names:
                print(translate("[INFO] 選択されたLoRAモデル: {0}").format(", ".join(selected_lora_names)))
            else:
                print(translate("[INFO] 有効なLoRAモデルが選択されていません"))
        else:
            # ファイルアップロードモード
            # 1つ目のLoRAファイルを処理
            if lora_files is not None:
                # デバッグ情報を出力
                print(f"[DEBUG] 情報表示: lora_filesの型: {type(lora_files)}")
                
                if isinstance(lora_files, list):
                    # 各ファイルが有効かチェック
                    for file in lora_files:
                        if file is not None:
                            all_lora_files.append(file)
                        else:
                            print(translate("[WARN] LoRAファイル1のリスト内に無効なファイルがあります"))
                elif lora_files is not None:
                    all_lora_files.append(lora_files)
                    
            # 2つ目のLoRAファイルを処理
            if lora_files2 is not None:
                # デバッグ情報を出力
                print(f"[DEBUG] 情報表示: lora_files2の型: {type(lora_files2)}")
                
                if isinstance(lora_files2, list):
                    # 各ファイルが有効かチェック
                    for file in lora_files2:
                        if file is not None:
                            all_lora_files.append(file)
                        else:
                            print(translate("[WARN] LoRAファイル2のリスト内に無効なファイルがあります"))
                elif lora_files2 is not None:
                    all_lora_files.append(lora_files2)
            
            # 3つ目のLoRAファイルを処理
            if lora_files3 is not None:
                # デバッグ情報を出力
                print(f"[DEBUG] 情報表示: lora_files3の型: {type(lora_files3)}")
                
                if isinstance(lora_files3, list):
                    # 各ファイルが有効かチェック
                    for file in lora_files3:
                        if file is not None:
                            all_lora_files.append(file)
                        else:
                            print(translate("[WARN] LoRAファイル3のリスト内に無効なファイルがあります")) 
                elif lora_files3 is not None:
                    all_lora_files.append(lora_files3)
        
        # スケール値を解析
        try:
            scales = [float(s.strip()) for s in lora_scales_text.split(',')]
        except:
            # 解析エラーの場合はデフォルト値を使用
            scales = [0.8] * len(all_lora_files)
            
        # スケール値の数を調整
        if len(scales) < len(all_lora_files):
            scales.extend([0.8] * (len(all_lora_files) - len(scales)))
        elif len(scales) > len(all_lora_files):
            scales = scales[:len(all_lora_files)]
            
        # LoRAファイル情報を出力
        if len(all_lora_files) == 1:
            # 単一ファイル
            print(translate("\u25c6 LoRAファイル: {0}").format(os.path.basename(all_lora_files[0].name)))
            print(translate("\u25c6 LoRA適用強度: {0}").format(scales[0]))
        elif len(all_lora_files) > 1:
            # 複数ファイル
            print(translate("\u25c6 LoRAファイル (複数):"))
            for i, file in enumerate(all_lora_files):
                print(f"   - {os.path.basename(file.name)} (スケール: {scales[i]})")
        else:
            # LoRAファイルなし
            print(translate("\u25c6 LoRA: 使用しない"))

    # セクションごとのキーフレーム画像の使用状況をログに出力
    valid_sections = []
    if section_settings is not None:
        for i, sec_data in enumerate(section_settings):
            if sec_data and sec_data[1] is not None:  # 画像が設定されている場合
                valid_sections.append(sec_data[0])

    if valid_sections:
        print(translate("\u25c6 使用するキーフレーム画像: セクション{0}").format(', '.join(map(str, valid_sections))))
    else:
        print(translate("◆ キーフレーム画像: デフォルト設定のみ使用"))

    print("=============================\n")

    # バッチ処理の全体停止用フラグ
    batch_stopped = False

    # 元のシード値を保存（バッチ処理用）
    original_seed = seed

    # ランダムシード状態をデバッグ表示
    print(f"[DEBUG] use_random_seed: {use_random_seed}, タイプ: {type(use_random_seed)}")
    
    # ランダムシード生成を文字列型も含めて判定
    use_random = False
    if isinstance(use_random_seed, bool):
        use_random = use_random_seed
    elif isinstance(use_random_seed, str):
        use_random = use_random_seed.lower() in ["true", "yes", "1", "on"]
        
    print(f"[DEBUG] 実際のランダムシード使用状態: {use_random}")
    
    if use_random:
        # ランダムシード設定前の値を保存
        previous_seed = seed
        # 特定の範囲内で新しいシード値を生成
        seed = random.randint(0, 2**32 - 1)
        # ユーザーにわかりやすいメッセージを表示
        print(translate("\n[INFO] ランダムシード機能が有効なため、指定されたSEED値 {0} の代わりに新しいSEED値 {1} を使用します。").format(previous_seed, seed))
        # UIのseed欄もランダム値で更新
        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update(value=seed)
        # ランダムシードの場合は最初の値を更新
        original_seed = seed
    else:
        print(translate("[INFO] 指定されたSEED値 {0} を使用します。").format(seed))
        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update()

    stream = AsyncStream()

    # stream作成後、バッチ処理前もう一度フラグを確認
    if batch_stopped:
        print(translate("\nバッチ処理が中断されました（バッチ開始前）"))
        yield (
            None,
            gr.update(visible=False),
            translate("バッチ処理が中断されました"),
            '',
            gr.update(interactive=True),
            gr.update(interactive=False, value=translate("End Generation")),
            gr.update()
        )
        return

    # バッチ処理ループの開始
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
                gr.update(interactive=False, value=translate("End Generation")),
                gr.update()
            )
            break

        # 現在のバッチ番号を表示
        if batch_count > 1:
            # キューが有効な場合はその情報も表示
            if bool(use_queue):
                # プロンプトキューの場合
                if queue_type == "prompt" and prompt_queue_file_path is not None and os.path.exists(prompt_queue_file_path):
                    try:
                        with open(prompt_queue_file_path, 'r', encoding='utf-8') as f:
                            prompt_lines = [line.strip() for line in f.readlines() if line.strip()]
                            prompt_count = len(prompt_lines)
                            if prompt_count > 0:
                                if batch_index + 1 <= prompt_count:
                                    batch_info = translate("バッチ処理: {0}/{1}（プロンプトキュー行を優先: {2}/{3}）").format(batch_index + 1, batch_count, batch_index + 1, prompt_count)
                                else:
                                    batch_info = translate("バッチ処理: {0}/{1}（共通プロンプト使用）").format(batch_index + 1, batch_count)
                            else:
                                batch_info = translate("バッチ処理: {0}/{1}").format(batch_index + 1, batch_count)
                    except Exception:
                        batch_info = translate("バッチ処理: {0}/{1}").format(batch_index + 1, batch_count)

                # イメージキューの場合
                elif queue_type == "image":
                    # グローバル変数からイメージキュー情報を取得
                    global image_queue_files

                    # イメージキューのファイルリストを更新（まだ取得していない場合）
                    if not image_queue_files:
                        get_image_queue_files()

                    if batch_index == 0:
                        batch_info = translate("バッチ処理: {0}/{1}（イメージキュー: 入力画像使用）").format(batch_index + 1, batch_count)
                    elif batch_index - 1 < len(image_queue_files):
                        img_file = os.path.basename(image_queue_files[batch_index - 1])
                        batch_info = translate("バッチ処理: {0}/{1}（イメージキュー: 画像 {2} 使用）").format(
                            batch_index + 1, batch_count, img_file)
                    else:
                        batch_info = translate("バッチ処理: {0}/{1}（イメージキュー: 画像不足のため入力画像再使用）").format(batch_index + 1, batch_count)

                # その他の場合
                else:
                    batch_info = translate("バッチ処理: {0}/{1}").format(batch_index + 1, batch_count)
            else:
                batch_info = translate("バッチ処理: {0}/{1}").format(batch_index + 1, batch_count)

            print(f"\n{batch_info}")
            # UIにもバッチ情報を表示
            yield None, gr.update(visible=False), batch_info, "", gr.update(interactive=False), gr.update(interactive=True), gr.update()

        # バッチインデックスに応じてSEED値を設定
        # ランダムシード使用判定を再度実施
        use_random = False
        if isinstance(use_random_seed, bool):
            use_random = use_random_seed
        elif isinstance(use_random_seed, str):
            use_random = use_random_seed.lower() in ["true", "yes", "1", "on"]
        
        # 複数バッチがある場合の表示
        if batch_count > 1:
            # ランダムシードを使用しない場合のみ、バッチインデックスでシードを調整
            if not use_random:
                prev_seed = seed
                current_seed = original_seed + batch_index
                # 現在のバッチ用のシードを設定
                seed = current_seed
                if batch_index > 0:  # 最初のバッチ以外でメッセージ表示
                    print(translate("[INFO] バッチ {0}/{1} の処理を開始: SEED値を {2} に設定しました。").format(
                        batch_index + 1, batch_count, seed
                    ))
            else:
                # ランダムシード使用時は各バッチで新しい値を生成
                if batch_index > 0:  # 最初のバッチ以外は新しい値を生成
                    prev_seed = seed
                    seed = random.randint(0, 2**32 - 1)
                    print(translate("[INFO] バッチ {0}/{1} の処理を開始: 新しいランダムSEED値 {2} を生成しました。").format(
                        batch_index + 1, batch_count, seed
                    ))
        
        # 常に現在のシード値を表示（バッチ数に関わらず）
        print(translate("現在のSEED値: {0}").format(seed))

        # もう一度停止フラグを確認 - worker処理実行前
        if batch_stopped:
            print(translate("バッチ処理が中断されました。worker関数の実行をキャンセルします。"))
            # 中断メッセージをUIに表示
            yield (None,
                   gr.update(visible=False),
                   translate("バッチ処理が中断されました（{0}/{1}）").format(batch_index, batch_count),
                   '',
                   gr.update(interactive=True),
                   gr.update(interactive=False, value=translate("End Generation")),
                   gr.update())
            break

        # GPUメモリの設定値をデバッグ出力し、正しい型に変換
        gpu_memory_value = float(gpu_memory_preservation) if gpu_memory_preservation is not None else 6.0
        print(translate('Using GPU memory preservation setting: {0} GB').format(gpu_memory_value))

        # 出力フォルダが空の場合はデフォルト値を使用
        if not output_dir or not output_dir.strip():
            output_dir = "outputs"
        print(translate('Output directory: {0}').format(output_dir))

        # 先に入力データの状態をログ出力（デバッグ用）
        if input_image is not None:
            if isinstance(input_image, str):
                print(translate("[DEBUG] input_image path: {0}, type: {1}").format(input_image, type(input_image)))
            else:
                print(translate("[DEBUG] input_image shape: {0}, type: {1}").format(input_image.shape, type(input_image)))
        if end_frame is not None:
            if isinstance(end_frame, str):
                print(translate("[DEBUG] end_frame path: {0}, type: {1}").format(end_frame, type(end_frame)))
            else:
                print(translate("[DEBUG] end_frame shape: {0}, type: {1}").format(end_frame.shape, type(end_frame)))
        if section_settings is not None:
            print(translate("[DEBUG] section_settings count: {0}").format(len(section_settings)))
            valid_images = sum(1 for s in section_settings if s and s[1] is not None)
            print(translate("[DEBUG] Valid section images: {0}").format(valid_images))

        # バッチ処理の各回で実行
        # worker関数の定義と引数の順序を完全に一致させる
        print(translate("[DEBUG] async_run直前のsave_tensor_data: {0}").format(save_tensor_data))
        print(translate("[DEBUG] async_run直前のLoRA関連パラメータ:"))
        print(translate("  - lora_mode: {0}, 型: {1}").format(lora_mode, type(lora_mode).__name__))
        print(translate("  - lora_dropdown1: {0!r}, 型: {1}").format(lora_dropdown1, type(lora_dropdown1).__name__))
        print(translate("  - lora_dropdown2: {0!r}, 型: {1}").format(lora_dropdown2, type(lora_dropdown2).__name__))
        print(translate("  - lora_dropdown3: {0!r}, 型: {1}").format(lora_dropdown3, type(lora_dropdown3).__name__))
        print(translate("  - use_lora: {0}, 型: {1}").format(use_lora, type(use_lora).__name__))
        
        # 特に2番目のドロップダウン値が正しく扱われていない場合のデバッグ情報を出力
        # この段階では値を変更せず、情報収集のみ行う
        if lora_mode == translate("ディレクトリから選択") and lora_dropdown2 == 0:
            print(translate("[DEBUG] lora_dropdown2が数値0になっています"))
            
            # ディレクトリ情報を出力（デバッグ用）
            lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
            if os.path.exists(lora_dir):
                print(translate("[DEBUG] LoRAディレクトリ内ファイル:"))
                for filename in os.listdir(lora_dir):
                    if filename.endswith(('.safetensors', '.pt', '.bin')):
                        print(f"  - {filename}")

        # イメージキューが有効な場合、バッチに応じた入力画像を設定
        current_input_image = input_image  # デフォルトでは元の入力画像

        if bool(use_queue) and queue_type == "image":
            # 最初のバッチ以外は処理
            if batch_index > 0:

                # イメージキューのファイルリストを更新（まだ取得していない場合）
                if not image_queue_files:
                    get_image_queue_files()

                # イメージキューのリストを詳細に出力（デバッグ用）
                print(f"イメージキュー詳細: 画像数={len(image_queue_files)}, batch_index={batch_index}")
                for i, img in enumerate(image_queue_files):
                    print(f"    画像{i+1}: {os.path.basename(img)}")

                # 画像キューから適切な画像を選択
                if image_queue_files and batch_index - 1 < len(image_queue_files):
                    queue_index = batch_index - 1
                    img_path = image_queue_files[queue_index]
                    img_name = os.path.basename(img_path)
                    print(translate("イメージキュー: バッチ{0}に画像「{1}」を設定 (インデックス: {2})").format(batch_index + 1, img_name, queue_index))
                    current_input_image = img_path
                else:
                    print(translate("イメージキュー: 画像が足りないため入力画像を再使用"))
            else:
                print(translate("イメージキュー: 最初のバッチには入力画像を使用"))

        # 入力画像の情報をログに出力
        if isinstance(current_input_image, str):
            print(translate("処理用入力画像: {0}").format(current_input_image))

        async_run(
            worker,
            current_input_image,  # イメージキューで変更された可能性がある入力画像
            prompt,
            n_prompt,
            seed,
            total_second_length,
            latent_window_size,
            steps,
            cfg,
            gs,
            rs,
            gpu_memory_value,  # gpu_memory_preservation
            use_teacache,
            mp4_crf,
            all_padding_value,
            end_frame,
            end_frame_strength,
            keep_section_videos,
            lora_files,
            lora_files2,
            lora_files3,  # 追加：3つ目のLoRAファイル
            lora_scales_text,
            output_dir,
            save_section_frames,
            section_settings,
            use_all_padding,
            use_lora,
            lora_mode,  # 追加：LoRAモード設定
            lora_dropdown1,  # 追加：LoRAドロップダウン1
            lora_dropdown2,  # 追加：LoRAドロップダウン2
            lora_dropdown3,  # 追加：LoRAドロップダウン3
            save_tensor_data,  # テンソルデータ保存フラグ - 確実に正しい位置に配置
            tensor_data_input,
            fp8_optimization,
            resolution,
            batch_index,
            save_latent_frames,  # 全フレーム画像保存フラグ（ラジオボタンから設定）
            save_last_section_frames,  # 最終セクションのみ全フレーム画像保存フラグ（ラジオボタンから設定）
            use_vae_cache,  # VAEキャッシュ設定
            bool(use_queue),  # キュー使用フラグ - 確実にブール値として渡す
            prompt_queue_file  # プロンプトキューファイル
        )

        # 現在のバッチの出力ファイル名
        batch_output_filename = None

        # 現在のバッチの処理結果を取得
        while True:
            flag, data = stream.output_queue.next()

            if flag == 'file':
                batch_output_filename = data
                # より明確な更新方法を使用し、preview_imageを明示的にクリア
                yield batch_output_filename, gr.update(value=None, visible=False), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update()

            if flag == 'progress':
                preview, desc, html = data
                # バッチ処理中は現在のバッチ情報を追加
                # 単一バッチでも常にSEED値情報を表示する
                batch_info = ""
                if batch_count > 1:
                    # プロンプトキューが有効な場合はその情報も表示
                    if bool(use_queue) and queue_type == "prompt" and prompt_queue_file_path is not None and os.path.exists(prompt_queue_file_path):
                        try:
                            with open(prompt_queue_file_path, 'r', encoding='utf-8') as f:
                                prompt_lines = [line.strip() for line in f.readlines() if line.strip()]
                                prompt_count = len(prompt_lines)
                                if prompt_count > 0:
                                    if batch_index + 1 <= prompt_count:
                                        batch_info = translate("バッチ処理: {0}/{1}（プロンプトキュー行を優先: {2}/{3}） - ").format(batch_index + 1, batch_count, batch_index + 1, prompt_count)
                                    else:
                                        batch_info = translate("バッチ処理: {0}/{1}（共通プロンプト使用） - ").format(batch_index + 1, batch_count)
                                else:
                                    batch_info = translate("バッチ処理: {0}/{1} - ").format(batch_index + 1, batch_count)
                        except Exception:
                            batch_info = translate("バッチ処理: {0}/{1} - ").format(batch_index + 1, batch_count)
                    else:
                        batch_info = translate("バッチ処理: {0}/{1} - ").format(batch_index + 1, batch_count)

                # 現在のSEED値を常に表示
                current_seed_info = translate("現在のSEED値: {0}").format(seed)
                if batch_info:
                    desc = batch_info + desc
                
                # プロンプトの末尾に現在のシード値の情報を追加（バッチ処理数に関わらず）
                if current_seed_info not in desc:
                    desc = desc + "\n\n" + current_seed_info
                # preview_imageを明示的に設定
                yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), gr.update()

            if flag == 'end':
                # このバッチの処理が終了
                if batch_index == batch_count - 1 or batch_stopped:
                    # 最終バッチの場合は処理完了を通知
                    completion_message = ""
                    if batch_stopped:
                        completion_message = translate("バッチ処理が中止されました（{0}/{1}）").format(batch_index + 1, batch_count)
                    else:
                        completion_message = translate("バッチ処理が完了しました（{0}/{1}）").format(batch_count, batch_count)
                    yield (
                        batch_output_filename,
                        gr.update(value=None, visible=False),
                        completion_message,
                        '',
                        gr.update(interactive=True),
                        gr.update(interactive=False, value=translate("End Generation")),
                        gr.update()
                    )
                else:
                    # 次のバッチに進むメッセージを表示
                    next_batch_message = translate("バッチ処理: {0}/{1} 完了、次のバッチに進みます...").format(batch_index + 1, batch_count)
                    yield (
                        batch_output_filename,
                        gr.update(value=None, visible=False),
                        next_batch_message,
                        '',
                        gr.update(interactive=False),
                        gr.update(interactive=True),
                        gr.update()
                    )
                break

        # 最終的な出力ファイル名を更新
        output_filename = batch_output_filename

        # バッチ処理が停止されている場合はループを抜ける
        if batch_stopped:
            print(translate("バッチ処理ループを中断します"))
            break

def end_process():
    global stream
    global batch_stopped

    # 現在のバッチと次のバッチ処理を全て停止するフラグを設定
    batch_stopped = True
    print(translate("\n停止ボタンが押されました。バッチ処理を停止します..."))
    # 現在実行中のバッチを停止
    stream.input_queue.push('end')

    # ボタンの名前を一時的に変更することでユーザーに停止処理が進行中であることを表示
    return gr.update(value=translate("停止処理中..."))

# 既存のQuick Prompts（初期化時にプリセットに変換されるので、互換性のために残す）
quick_prompts = [
    'A character doing some simple body movements.',
    'A character uses expressive hand gestures and body language.',
    'A character walks leisurely with relaxed movements.',
    'A character performs dynamic movements with energy and flowing motion.',
    'A character moves in unexpected ways, with surprising transitions poses.',
]
quick_prompts = [[x] for x in quick_prompts]

css = get_app_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.HTML('<h1>FramePack<span class="title-suffix">-eichi</span></h1>')

    # デバッグ情報の表示
    # print_keyframe_debug_info()

    # 一番上の行に「生成モード、セクションフレームサイズ、オールパディング、動画長」を配置
    with gr.Row():
        with gr.Column(scale=1):
            mode_radio = gr.Radio(choices=[MODE_TYPE_NORMAL, MODE_TYPE_LOOP], value=MODE_TYPE_NORMAL, label=translate("生成モード"), info=translate("通常：一般的な生成 / ループ：ループ動画用"))
        with gr.Column(scale=1):
            # フレームサイズ切替用のUIコントロール（名前を「セクションフレームサイズ」に変更）
            frame_size_radio = gr.Radio(
                choices=[translate("1秒 (33フレーム)"), translate("0.5秒 (17フレーム)")],
                value=translate("1秒 (33フレーム)"),
                label=translate("セクションフレームサイズ"),
                info=translate("1秒 = 高品質・通常速度 / 0.5秒 = よりなめらかな動き（実験的機能）")
            )
        with gr.Column(scale=1):
            # オールパディング設定
            use_all_padding = gr.Checkbox(label=translate("オールパディング"), value=False, info=translate("数値が小さいほど直前の絵への影響度が下がり動きが増える"), elem_id="all_padding_checkbox")
            all_padding_value = gr.Slider(label=translate("パディング値"), minimum=0.2, maximum=3, value=1, step=0.1, info=translate("すべてのセクションに適用するパディング値（0.2〜3の整数）"), visible=False)
            
            # オールパディングのチェックボックス状態に応じてスライダーの表示/非表示を切り替える
            def toggle_all_padding_visibility(use_all_padding):
                return gr.update(visible=use_all_padding)
                
            # オールパディングのチェックボックス状態変更のみ
            use_all_padding.change(
                fn=toggle_all_padding_visibility,
                inputs=[use_all_padding],
                outputs=[all_padding_value]
            )
        with gr.Column(scale=1):
            # 設定から動的に選択肢を生成
            length_radio = gr.Radio(choices=get_video_modes(), value=translate("1秒"), label=translate("動画長"), info=translate("キーフレーム画像のコピー範囲と動画の長さを設定"))

    with gr.Row():
        with gr.Column():
            # Final Frameの上に説明を追加
            gr.Markdown(translate("**Finalは最後の画像、Imageは最初の画像(最終キーフレーム画像といずれか必須)となります。**"))
            end_frame = gr.Image(sources=['upload', 'clipboard'], type="filepath", label=translate("Final Frame (Optional)"), height=320)

            # End Frame画像のアップロード時のメタデータ抽出機能は一旦コメント化
            # def update_from_end_frame_metadata(image):
            #     """End Frame画像からメタデータを抽出してUIに反映する"""
            #     if image is None:
            #         return [gr.update()] * 2
            #
            #     try:
            #         # NumPy配列からメタデータを抽出
            #         metadata = extract_metadata_from_numpy_array(image)
            #
            #         if not metadata:
            #             print(translate("End Frame画像にメタデータが含まれていません"))
            #             return [gr.update()] * 2
            #
            #         print(translate("End Frame画像からメタデータを抽出しました: {0}").format(metadata))
            #
            #         # プロンプトとSEEDをUIに反映
            #         prompt_update = gr.update()
            #         seed_update = gr.update()
            #
            #         if PROMPT_KEY in metadata and metadata[PROMPT_KEY]:
            #             prompt_update = gr.update(value=metadata[PROMPT_KEY])
            #             print(translate("プロンプトをEnd Frame画像から取得: {0}").format(metadata[PROMPT_KEY]))
            #
            #         if SEED_KEY in metadata and metadata[SEED_KEY]:
            #             # SEED値を整数に変換
            #             try:
            #                 seed_value = int(metadata[SEED_KEY])
            #                 seed_update = gr.update(value=seed_value)
            #                 print(translate("SEED値をEnd Frame画像から取得: {0}").format(seed_value))
            #             except (ValueError, TypeError):
            #                 print(translate("SEED値の変換エラー: {0}").format(metadata[SEED_KEY]))
            #
            #         return [prompt_update, seed_update]
            #     except Exception as e:
            #         print(translate("End Frameメタデータ抽出エラー: {0}").format(e))
            #         return [gr.update()] * 2
            #
            # # End Frame画像アップロード時のメタデータ取得処理を登録
            # end_frame.change(
            #     fn=update_from_end_frame_metadata,
            #     inputs=[end_frame],
            #     outputs=[prompt, seed]
            # )

            # テンソルデータ設定をグループ化して灰色のタイトルバーに変更
            with gr.Group():
                gr.Markdown(f"### " + translate("テンソルデータ設定"))

                # テンソルデータ使用有無のチェックボックス
                use_tensor_data = gr.Checkbox(label=translate("テンソルデータを使用する"), value=False, info=translate("チェックをオンにするとテンソルデータをアップロードできます"))

                # テンソルデータ設定コンポーネント（初期状態では非表示）
                with gr.Group(visible=False) as tensor_data_group:
                    # テンソルデータファイルのアップロード
                    tensor_data_input = gr.File(
                        label=translate("テンソルデータアップロード (.safetensors) - 生成動画の後方(末尾)に結合されます"),
                        file_types=[".safetensors"]
                    )

                    gr.Markdown(translate("※ テンソルデータをアップロードすると通常の動画生成後に、その動画の後方（末尾）に結合されます。\n結合した動画は「元のファイル名_combined.mp4」として保存されます。\n※ テンソルデータの保存機能を有効にすると、生成とアップロードのテンソルを結合したデータも保存されます。\n※ テンソルデータの結合は別ツール `python eichi_utils/tensor_combiner.py --ui` でもできます。"))

                    # チェックボックスで表示/非表示を切り替え
                    use_interpolation_section = gr.Checkbox(
                        label=translate("テンソルデータと生成動画のスムーズ結合機能を表示"),
                        value=False,
                        info=translate("チェックをオンにするとテンソルデータと新規に生成する動画をスムーズに結合する機能を表示します")
                    )

                    with gr.Group(visible=False) as interpolation_group:
                        # テンソルデータと新規生成動画のスムージング結合のチェックボックス
                        gr.Markdown(f"### " + translate("テンソルデータと生成動画のスムーズ結合機能"))

                        # テンソルデータの先頭の削除フレーム数
                        tensor_trim_start_latents = gr.Slider(
                            label=translate("テンソルデータの先頭の削除フレーム数"),
                            minimum=0,
                            maximum=5,
                            value=0,
                            step=1,
                            interactive=True,
                            info=translate("テンソルデータの先頭から削除するフレーム数。テンソルデータの先頭部分にノイズがある場合に、設定してください。出力結果の品質を確認して調整してください。")
                        )

                        # 新規生成側の末尾の削除フレーム数
                        current_trim_end_latents = gr.Slider(
                            label=translate("新規生成側の末尾の削除フレーム数"),
                            minimum=0,
                            maximum=3,
                            value=0,
                            step=1,
                            interactive=True,
                            info=translate("新規で生成する動画の末尾から削除するフレーム数。0で大丈夫です。※新規生成部分の末尾はコントロールしやすいので、今のところ利用想定ケースがないです。")
                        )

                        # 補間フレーム数
                        interpolation_latents = gr.Slider(
                            label=translate("補間フレーム数"),
                            minimum=0,
                            maximum=12,
                            value=0,
                            step=1,
                            interactive=True,
                            info=translate("新規で生成する動画の末尾とテンソルデータの先頭をつなげるため、追加する補間フレーム数。6-9推奨。補間された動画は、動画長とは別に加算されます。※現状専用プロンプトはなく、メインプロンプトが使用されます。\n設定値と時間の目安：3（0.3秒）、6（0.7秒）、9（1.1秒）、12（1.5秒）")
                        )

                        def change_trim_start_latents(value):
                            return value

                        tensor_trim_start_latents.change(
                            fn=change_trim_start_latents,
                            inputs=[tensor_trim_start_latents],
                            outputs=[]
                        )

                        def change_current_trim_end_latents(value):
                            return value

                        current_trim_end_latents.change(
                            fn=change_current_trim_end_latents,
                            inputs=[current_trim_end_latents],
                            outputs=[]
                        )

                        def change_interpolation_latents(value):
                            return value

                        interpolation_latents.change(
                            fn=change_interpolation_latents,
                            inputs=[interpolation_latents],
                            outputs=[]
                        )

                    # チェックボックスの状態によってテンソルデータと生成動画のスムーズ機能の表示/非表示を切り替える関数
                    def toggle_interpolation_settings(use_interpolation):
                        return gr.update(visible=use_interpolation)

                    # チェックボックスの変更イベントに関数を紐づけ
                    use_interpolation_section.change(
                        fn=toggle_interpolation_settings,
                        inputs=[use_interpolation_section],
                        outputs=[interpolation_group]
                    )

                # チェックボックスの状態によってテンソルデータ設定の表示/非表示を切り替える関数
                def toggle_tensor_data_settings(use_tensor):
                    return gr.update(visible=use_tensor)

                # チェックボックスの変更イベントに関数を紐づけ
                use_tensor_data.change(
                    fn=toggle_tensor_data_settings,
                    inputs=[use_tensor_data],
                    outputs=[tensor_data_group]
                )

            # テンソルデータ設定の下に解像度スライダーとバッチ処理回数を追加
            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=2):
                        resolution = gr.Dropdown(
                            label=translate("解像度"),
                            choices=[512, 640, 768, 960, 1080],
                            value=640,
                            info=translate("出力動画の基準解像度。640推奨。960/1080は高負荷・高メモリ消費")
                        )
                    with gr.Column(scale=1):
                        batch_count = gr.Slider(
                            label=translate("バッチ処理回数"),
                            minimum=1,
                            maximum=100,
                            value=1,
                            step=1,
                            info=translate("同じ設定で連続生成する回数。SEEDは各回で+1されます")
                        )

                # キュー機能のチェックボックス
                use_queue = gr.Checkbox(
                    label=translate("キューを使用"),
                    value=False,
                    info=translate("チェックをオンにするとプロンプトまたは画像の連続処理ができます。")
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
                    prompt_queue_file = gr.File(
                        label=translate("プロンプトキューファイル (.txt) - 1行に1つのプロンプトが記載されたテキストファイル"),
                        file_types=[".txt"]
                    )
                    gr.Markdown(translate("※ ファイル内の各行が別々のプロンプトとして処理されます。\n※ チェックボックスがオフの場合は無効。\n※ バッチ処理回数より行数が多い場合は行数分処理されます。\n※ バッチ処理回数が1でもキュー回数が優先されます。"))

                # イメージキュー用グループ
                with gr.Group(visible=False) as image_queue_group:
                    gr.Markdown(translate("※ 1回目はImage画像を使用し、2回目以降は入力フォルダの画像ファイルを名前順に使用します。\n※ バッチ回数が全画像数を超える場合、残りはImage画像で処理されます。\n※ バッチ処理回数が1でもキュー回数が優先されます。"))

                    # 入力フォルダ設定
                    with gr.Row():
                        input_folder_name = gr.Textbox(
                            label=translate("入力フォルダ名"),
                            value=input_folder_name_value,  # グローバル変数から値を取得
                            info=translate("画像ファイルを格納するフォルダ名")
                        )
                        open_input_folder_btn = gr.Button(value="📂 " + translate("保存及び入力フォルダを開く"), size="md")

                    # 入力フォルダ関連のイベントハンドラ
                    def handle_input_folder_change(folder_name):
                        """入力フォルダ名が変更されたときの処理（グローバル変数に保存するだけ）"""
                        global input_folder_name_value

                        # 入力値をトリミング
                        folder_name = folder_name.strip()

                        # 空の場合はデフォルト値に戻す
                        if not folder_name:
                            folder_name = "inputs"

                        # 無効な文字を削除（パス区切り文字やファイル名に使えない文字）
                        folder_name = ''.join(c for c in folder_name if c.isalnum() or c in ('_', '-'))

                        # グローバル変数を更新（設定の保存は行わない）
                        input_folder_name_value = folder_name
                        print(translate("入力フォルダ名をメモリに保存: {0}（保存及び入力フォルダを開くボタンを押すと保存されます）").format(folder_name))

                        # UIの表示を更新
                        return gr.update(value=folder_name)

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

                        # プラットフォームに応じてフォルダを開く
                        try:
                            if os.name == 'nt':  # Windows
                                os.startfile(input_dir)
                            elif os.name == 'posix':  # macOS, Linux
                                if sys.platform == 'darwin':  # macOS
                                    subprocess.Popen(['open', input_dir])
                                else:  # Linux
                                    subprocess.Popen(['xdg-open', input_dir])
                            print(translate("入力フォルダを開きました: {0}").format(input_dir))
                            return translate("設定を保存し、入力フォルダを開きました")
                        except Exception as e:
                            error_msg = translate("フォルダを開けませんでした: {0}").format(str(e))
                            print(error_msg)
                            return error_msg

                # ファイルアップロード時のイベントハンドラ
                def handle_file_upload(file_obj):
                    global prompt_queue_file_path

                    if file_obj is not None:
                        print(f"ファイルアップロード検出: 型={type(file_obj).__name__}")

                        if hasattr(file_obj, 'name'):
                            prompt_queue_file_path = file_obj.name
                            print(f"アップロードファイルパス保存: {prompt_queue_file_path}")
                        else:
                            prompt_queue_file_path = file_obj
                            print(f"アップロードファイルデータ保存: {file_obj}")
                    else:
                        prompt_queue_file_path = None
                        print("ファイルアップロード解除")

                    return file_obj

                # イメージキューのための画像ファイルリスト取得関数はグローバル関数を使用

                # ファイルアップロードイベントをハンドラに接続
                prompt_queue_file.change(
                    fn=handle_file_upload,
                    inputs=[prompt_queue_file],
                    outputs=[prompt_queue_file]
                )

                # 入力フォルダ名変更イベントをハンドラに接続
                input_folder_name.change(
                    fn=handle_input_folder_change,
                    inputs=[input_folder_name],
                    outputs=[input_folder_name]
                )

                # 入力フォルダを開くボタンにイベントを接続
                open_input_folder_btn.click(
                    fn=open_input_folder,
                    inputs=[],
                    outputs=[gr.Textbox(visible=False)]  # 一時的なフィードバック表示用（非表示）
                )

                # チェックボックスの状態によってキュー設定の表示/非表示を切り替える関数
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
                    # ブール値の場合
                    elif isinstance(use_queue_val, bool):
                        is_enabled = use_queue_val
                    # 文字列の場合 (True/Falseを表す文字列かチェック)
                    elif isinstance(use_queue_val, str) and use_queue_val.lower() in ('true', 'false', 't', 'f', 'yes', 'no', 'y', 'n', '1', '0'):
                        is_enabled = use_queue_val.lower() in ('true', 't', 'yes', 'y', '1')

                    # グローバル変数に保存
                    queue_enabled = is_enabled

                    print(f"キュー設定の表示状態を変更: {is_enabled} (グローバル変数に保存: queue_enabled={queue_enabled})")

                    # キュータイプの選択肢も表示/非表示を切り替える
                    # チェックがオンならプロンプトキュー（デフォルト）のグループを表示、オフなら両方非表示
                    if is_enabled:
                        # キュータイプの現在の選択に基づいて適切なグループを表示
                        if queue_type == "prompt":
                            return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]
                        else:  # image
                            return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)]
                    else:
                        # チェックがオフなら全て非表示
                        return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]

                # キュータイプの選択による表示切り替え
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

                # チェックボックスの変更イベントに関数を紐づけ
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

            # 開始・終了ボタン
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

            # セクション入力用のリストを初期化
            section_number_inputs = []
            section_image_inputs = []
            section_prompt_inputs = []  # プロンプト入力欄用のリスト
            section_row_groups = []  # 各セクションのUI行を管理するリスト
            section_image_states = []   # 画像のパスを保持するState配列
            section_prompt_states = []  # プロンプトの値を保持するState配列
            
            # エンドフレームと初期フレーム用のState変数
            end_frame_state = gr.State(None)  # エンドフレーム画像のパスを保持
            input_image_state = gr.State(None)  # 開始フレーム画像のパスを保持
            
            # 設定から最大キーフレーム数を取得
            max_keyframes = get_max_keyframes_count()
            
            # 各セクションの画像パスとプロンプトを保持するためのStateを初期化
            for i in range(max_keyframes):
                section_image_states.append(gr.State(None))
                section_prompt_states.append(gr.State(""))
            
            # セクションUIデバッグ用フラグ
            debug_section_inputs = True  # デバッグ有効

            # 現在の動画モードで必要なセクション数を取得する関数
            def get_current_sections_count():
                mode_value = length_radio.value
                if mode_value in VIDEO_MODE_SETTINGS:
                    # sections値をそのまま使用 - 注：これは0から始めた場合の最大値となる
                    return VIDEO_MODE_SETTINGS[mode_value]["sections"]
                return max_keyframes  # デフォルト値

            # 現在の必要セクション数を取得
            initial_sections_count = get_current_sections_count()
            # セクション設定タイトルの定義と動的な更新用の関数
            # 現在のセクション数に応じたMarkdownを返す関数
            def generate_section_title(total_sections):
                last_section = total_sections - 1
                return translate('### セクション設定（逆順表示）\n\nセクションは逆時系列で表示されています。Image(始点)は必須でFinal(終点)から遡って画像を設定してください。**最終キーフレームの画像は、Image(始点)より優先されます。総数{0}**').format(total_sections)

            # 動画のモードとフレームサイズに基づいてセクション数を計算し、タイトルを更新する関数
            def update_section_title(frame_size, mode, length):
                seconds = get_video_seconds(length)
                latent_window_size = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                frame_count = latent_window_size * 4 - 3
                total_frames = int(seconds * 30)
                total_sections = int(max(round(total_frames / frame_count), 1))
                # 表示セクション数の設定
                # 例: 総セクション数が5の場合、4～0の5個のセクションが表示される
                display_sections = total_sections
                return generate_section_title(display_sections)

            # 初期タイトルを計算
            initial_title = update_section_title(translate("1秒 (33フレーム)"), MODE_TYPE_NORMAL, translate("1秒"))

            # 埋め込みプロンプトおよびシードを複写するチェックボックスの定義
            # 参照先として必要だが、表示はLoRA設定の下で行うため非表示に設定
            global copy_metadata
            copy_metadata = gr.Checkbox(
                label=translate("埋め込みプロンプトおよびシードを複写する"),
                value=False,
                info=translate("チェックをオンにすると、画像のメタデータからプロンプトとシードを自動的に取得します"),
                visible=False  # 元の位置では非表示
            )

            with gr.Accordion(translate("セクション設定"), open=False, elem_classes="section-accordion"):
                # セクション情報zipファイル処理（一括ダウンロード&アップロード）を追加
                with gr.Group():
                    gr.Markdown(f"### " + translate("セクション情報一括ダウンロード"))
                    # 一括ダウンロードボタンを追加（primary=オレンジ色）
                    download_sections_button = gr.Button(translate("セクション情報をZIPでダウンロード"), variant="primary")
                    # ダウンロードコンポーネント（非表示）
                    download_file = gr.File(label=translate("ダウンロードファイル"), visible=False, interactive=False)
                
                with gr.Group():
                    gr.Markdown(f"### " + translate("セクション情報一括アップロード"))
                    # チェックボックスで表示/非表示を切り替え
                    show_upload_section = gr.Checkbox(
                        label=translate("一括アップロード機能を表示"),
                        value=False,
                        info=translate("チェックをオンにするとセクション情報の一括アップロード機能を表示します")
                    )
                    # 初期状態では非表示
                    with gr.Group(visible=False) as upload_section_group:
                        upload_zipfile = gr.File(label=translate("セクション情報アップロードファイル"), file_types=[".zip"], interactive=True)

                    # チェックボックスの状態変更時に表示/非表示を切り替える
                    show_upload_section.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=[show_upload_section],
                        outputs=[upload_section_group]
                    )

                with gr.Group(elem_classes="section-container"):
                    section_title = gr.Markdown(initial_title)

                    # セクション番号0の上にコピー機能チェックボックスを追加（ループモード時のみ表示）
                    with gr.Row(visible=(mode_radio.value == MODE_TYPE_LOOP)) as copy_button_row:
                        keyframe_copy_checkbox = gr.Checkbox(label=translate("キーフレーム自動コピー機能を有効にする"), value=True, info=translate("オンにするとキーフレーム間の自動コピーが行われます"))

                    for i in range(max_keyframes):
                        with gr.Row(visible=(i < initial_sections_count), elem_classes="section-row") as row_group:
                            # 左側にセクション番号とプロンプトを配置
                            with gr.Column(scale=1):
                                section_number = gr.Number(label=translate("セクション番号 {0}").format(i), value=i, precision=0)
                                # デフォルト値は空文字列
                                section_prompt = gr.Textbox(
                                    label=translate("セクションプロンプト {0}").format(i), 
                                    placeholder=translate("セクション固有のプロンプト（空白の場合は共通プロンプトを使用）"), 
                                    lines=2,
                                    # 値が変更されるたびに即時に保存するためのイベントを追加
                                    every=1  # 入力のたびに値を更新
                                )

                            # 右側にキーフレーム画像のみ配置
                            with gr.Column(scale=2):
                                section_image = gr.Image(label=translate("キーフレーム画像 {0}").format(i), sources="upload", type="filepath", height=200)

                                # プロンプト変更時にStateを更新するハンドラー
                                def update_prompt_state(prompt_value, section_idx=i):
                                    """プロンプト入力欄が変更されたときにStateに値を保存するハンドラー"""
                                    # print(f"[DEBUG] セクション{section_idx}のプロンプト変更を検知: '{prompt_value}'")
                                    # Stateに直接値を設定して確実に保存
                                    if prompt_value is not None:
                                        section_prompt_states[section_idx].value = prompt_value
                                        # print(f"[IMPORTANT] セクション{section_idx}のプロンプトをStateに保存: '{prompt_value}'")
                                    return prompt_value
                                
                                # プロンプト入力欄の変更を監視してStateに保存する - change イベント
                                section_prompt.change(
                                    fn=update_prompt_state,
                                    inputs=[section_prompt],
                                    outputs=[section_prompt_states[i]]
                                )
                                
                                # プロンプトに対するすべての入力も監視（テキスト入力中も検知）
                                if hasattr(section_prompt, 'input'):
                                    section_prompt.input(
                                        fn=update_prompt_state,
                                        inputs=[section_prompt],
                                        outputs=[section_prompt_states[i]]
                                    )
                                # submit イベントも監視
                                if hasattr(section_prompt, 'submit'):
                                    section_prompt.submit(
                                        fn=update_prompt_state,
                                        inputs=[section_prompt],
                                        outputs=[section_prompt_states[i]]
                                    )
                                
                                # 各キーフレーム画像のアップロード時のメタデータ抽出処理
                                # クロージャーで現在のセクション番号を捕捉
                                def create_section_metadata_handler(section_idx, section_prompt_input):
                                    def update_from_section_image_metadata(image_path, copy_enabled=False):
                                        # print(translate("\n[DEBUG] セクション{0}の画像メタデータ抽出処理が開始されました").format(section_idx))
                                        # print(translate("[DEBUG] メタデータ複写機能: {0}").format(copy_enabled))

                                        # 画像パスをState変数に保存する - 必ず保存する（メタデータ複写機能の影響を受けない）
                                        if image_path is not None:
                                            # print(translate("[DEBUG] セクション{0}の画像パスをStateに保存: {1}").format(section_idx, image_path))
                                            # いずれの場合も画像パスを確実に保存
                                            # この時点で画像パスのみを保存するため、グローバル変数に直接アクセス
                                            section_image_states[section_idx].value = image_path
                                            # print(translate("[IMPORTANT] セクション{0}のState値を直接更新: {1}").format(section_idx, image_path))
                                        else:
                                            # print(translate("[DEBUG] セクション{0}の画像パスがNoneです").format(section_idx))
                                            pass

                                        # 複写機能が無効の場合は無視
                                        if not copy_enabled:
                                            # print(translate("[DEBUG] セクション{0}: メタデータ複写機能が無効化されているため、処理をスキップします").format(section_idx))
                                            pass
                                            # 画像パスは保存するがプロンプト更新はしない
                                            # gr.updateを返す（valueを指定しないとUI値が維持される）
                                            return gr.update(), image_path

                                        if image_path is None:
                                            # print(translate("[DEBUG] セクション{0}の画像パスがNoneです").format(section_idx))
                                            # 画像がなければ空の文字列を返す
                                            return "", None

                                        # print(translate("[DEBUG] セクション{0}の画像パス: {1}").format(section_idx, image_path))

                                        try:
                                            # ファイルパスから直接メタデータを抽出
                                            # print(translate("[DEBUG] セクション{0}からextract_metadata_from_pngを直接呼び出し").format(section_idx))
                                            metadata = extract_metadata_from_png(image_path)

                                            if not metadata:
                                                # print(translate("[DEBUG] セクション{0}の画像からメタデータが抽出されませんでした").format(section_idx))
                                                # メタデータが存在しない場合も空文字列を返す
                                                current_value = ""
                                                if section_prompt_input is not None:
                                                    if hasattr(section_prompt_input, 'value'):
                                                        current_value = section_prompt_input.value
                                                    else:
                                                        current_value = str(section_prompt_input)
                                                return current_value, image_path

                                            # print(translate("[DEBUG] セクション{0}の抽出されたメタデータ: {1}").format(section_idx, metadata))

                                            # セクションプロンプトを取得
                                            if SECTION_PROMPT_KEY in metadata and metadata[SECTION_PROMPT_KEY]:
                                                section_prompt_value = metadata[SECTION_PROMPT_KEY]
                                                # print(translate("[DEBUG] セクション{0}のプロンプトを画像から取得: {1}").format(section_idx, section_prompt_value))
                                                print(translate("セクション{0}のプロンプトを画像から取得: {1}").format(section_idx, section_prompt_value))
                                                
                                                # Stateに直接値を設定して確実に保存
                                                if section_prompt_states[section_idx] is not None:
                                                    # 複写機能がオンの時だけプロンプトを上書き - 冗長チェックだが安全のため
                                                    if copy_enabled:
                                                        section_prompt_states[section_idx].value = section_prompt_value
                                                        # print(translate("[IMPORTANT] セクション{0}のプロンプトをStateに直接保存: {1}").format(section_idx, section_prompt_value))
                                                    else:
                                                        pass
                                                        # print(translate("[INFO] 複写機能がオフのため、セクション{0}のプロンプト({1})は既存の値を維持します").format(section_idx, section_prompt_value))
                                                
                                                # copy_enabledがtrueの場合のみプロンプト値を返す
                                                # falseの場合はgr.update()を返して現在の値を維持
                                                if copy_enabled:
                                                    return section_prompt_value, image_path
                                                else:
                                                    return gr.update(), image_path

                                            # 通常のプロンプトがあればそれをセクションプロンプトに設定
                                            elif PROMPT_KEY in metadata and metadata[PROMPT_KEY]:
                                                prompt_value = metadata[PROMPT_KEY]
                                                # print(translate("[DEBUG] セクション{0}のプロンプトを画像の一般プロンプトから取得: {1}").format(section_idx, prompt_value))
                                                print(translate("セクション{0}のプロンプトを画像の一般プロンプトから取得: {1}").format(section_idx, prompt_value))
                                                
                                                # Stateに直接値を設定して確実に保存
                                                if section_prompt_states[section_idx] is not None:
                                                    # 複写機能がオンの時だけプロンプトを上書き - 冗長チェックだが安全のため
                                                    if copy_enabled:
                                                        section_prompt_states[section_idx].value = prompt_value
                                                        # print(translate("[IMPORTANT] セクション{0}のプロンプトをStateに直接保存: {1}").format(section_idx, prompt_value))
                                                    else:
                                                        pass
                                                        # print(translate("[INFO] 複写機能がオフのため、セクション{0}のプロンプト({1})は既存の値を維持します").format(section_idx, prompt_value))
                                                
                                                # copy_enabledがtrueの場合のみプロンプト値を返す
                                                # falseの場合はgr.update()を返して現在の値を維持
                                                if copy_enabled:
                                                    return prompt_value, image_path
                                                else:
                                                    return gr.update(), image_path
                                        except Exception as e:
                                            # print(translate("[ERROR] セクション{0}のメタデータ抽出エラー: {1}").format(section_idx, e))
                                            # traceback.print_exc()
                                            print(translate("セクション{0}のメタデータ抽出エラー: {1}").format(section_idx, e))

                                        # エラー時や他の条件に該当しない場合はgr.update()を返して現在の値を維持
                                        return gr.update(), image_path
                                    return update_from_section_image_metadata

                                # キーフレーム画像アップロード時のメタデータ取得処理を登録
                                # 画像変更イベントを処理するハンドラ - State直接更新実装のため、outputs側にはStateを含めない
                                # 問題: outputs内にsection_prompt_states[i]を含めると、埋め込みプロンプトの形式が変化する
                                # 修正: section_prompt_states[i]をoutputsから除外し、関数内で直接更新
                                section_image.change(
                                    fn=create_section_metadata_handler(i, section_prompt),
                                    inputs=[section_image, copy_metadata],
                                    outputs=[section_prompt, section_image_states[i]]  # section_prompt_states[i]は除外
                                )
                                
                                # 同じキーフレーム画像を再度アップロードしたときの安全策
                                section_image.upload(
                                    fn=lambda img, idx=i: (img, img),
                                    inputs=[section_image],
                                    outputs=[section_image, section_image_states[i]]
                                )
                            section_number_inputs.append(section_number)
                            section_image_inputs.append(section_image)
                            section_prompt_inputs.append(section_prompt)
                            section_row_groups.append(row_group)  # 行全体をリストに保存

                    # ※ enable_keyframe_copy変数は後で使用するため、ここで定義（モードに応じた初期値設定）
                    enable_keyframe_copy = gr.State(mode_radio.value == MODE_TYPE_LOOP) # ループモードの場合はTrue、通常モードの場合はFalse

                    # キーフレーム自動コピーチェックボックスの変更をenable_keyframe_copyに反映させる関数
                    def update_keyframe_copy_state(value):
                        return value

                    # チェックボックスの変更がenable_keyframe_copyに反映されるようにイベントを設定
                    keyframe_copy_checkbox.change(
                        fn=update_keyframe_copy_state,
                        inputs=[keyframe_copy_checkbox],
                        outputs=[enable_keyframe_copy]
                    )

                    # チェックボックス変更時に赤枠/青枠の表示を切り替える
                    def update_frame_visibility_from_checkbox(value, mode):
                    #   print(translate("チェックボックス変更: 値={0}, モード={1}").format(value, mode))
                        # モードとチェックボックスの両方に基づいて枠表示を決定
                        is_loop = (mode == MODE_TYPE_LOOP)

                        # 通常モードでは常に赤枠/青枠を非表示 (最優先で確認)
                        if not is_loop:
                        #   print(translate("通常モード (チェックボックス値={0}): 赤枠/青枠を強制的に非表示にします").format(value))
                            # 通常モードでは常にelm_classesを空にして赤枠/青枠を非表示に確定する
                            return gr.update(elem_classes=""), gr.update(elem_classes="")

                        # ループモードでチェックボックスがオンの場合のみ枠を表示
                        if value:
                        #   print(translate("ループモード + チェックボックスオン: 赤枠/青枠を表示します"))
                            return gr.update(elem_classes="highlighted-keyframe-red"), gr.update(elem_classes="highlighted-keyframe-blue")
                        else:
                        #   print(translate("ループモード + チェックボックスオフ: 赤枠/青枠を非表示にします"))
                            # ループモードでもチェックがオフなら必ずelem_classesを空にして赤枠/青枠を非表示にする
                            return gr.update(elem_classes=""), gr.update(elem_classes="")

                    keyframe_copy_checkbox.change(
                        fn=update_frame_visibility_from_checkbox,
                        inputs=[keyframe_copy_checkbox, mode_radio],
                        outputs=[section_image_inputs[0], section_image_inputs[1]]
                    )

                    # モード切り替え時にチェックボックスの値と表示状態を制御する
                    def toggle_copy_checkbox_visibility(mode):
                        """モード切り替え時にチェックボックスの表示/非表示を切り替える"""
                        is_loop = (mode == MODE_TYPE_LOOP)
                        # 通常モードの場合はチェックボックスを非表示に設定、コピー機能を必ずFalseにする
                        if not is_loop:
                        #   print(translate("モード切替: {0} -> チェックボックス非表示、コピー機能を無効化").format(mode))
                            return gr.update(visible=False, value=False), gr.update(visible=False), False
                        # ループモードの場合は表示し、デフォルトでオンにする
                    #   print(translate("モード切替: {0} -> チェックボックス表示かつオンに設定").format(mode))
                        return gr.update(visible=True, value=True), gr.update(visible=True), True

                    # モード切り替え時にチェックボックスの表示/非表示と値を制御するイベントを設定
                    mode_radio.change(
                        fn=toggle_copy_checkbox_visibility,
                        inputs=[mode_radio],
                        outputs=[keyframe_copy_checkbox, copy_button_row, enable_keyframe_copy]
                    ) # ループモードに切替時は常にチェックボックスをオンにし、通常モード時は常にオフにする

                    # モード切り替え時に赤枠/青枠の表示を更新
                    def update_frame_visibility_from_mode(mode):
                        # モードに基づいて枠表示を決定
                        is_loop = (mode == MODE_TYPE_LOOP)

                        # 通常モードでは無条件で赤枠/青枠を非表示 (最優先で確定)
                        if not is_loop:
                        #   print(translate("モード切替: 通常モード -> 枠を強制的に非表示"))
                            return gr.update(elem_classes=""), gr.update(elem_classes="")
                        else:
                            # ループモードではチェックボックスが常にオンになるので枠を表示
                        #   print(translate("モード切替: ループモード -> チェックボックスオンなので枠を表示"))
                            return gr.update(elem_classes="highlighted-keyframe-red"), gr.update(elem_classes="highlighted-keyframe-blue")

                    mode_radio.change(
                        fn=update_frame_visibility_from_mode,
                        inputs=[mode_radio],
                        outputs=[section_image_inputs[0], section_image_inputs[1]]
                    )

            input_image = gr.Image(sources=['upload', 'clipboard'], type="filepath", label="Image", height=320)

            # メタデータ抽出関数を定義（後で登録する）
            def update_from_image_metadata(image_path, copy_enabled=False):
                """Imageアップロード時にメタデータを抽出してUIに反映する
                copy_enabled: メタデータの複写が有効化されているかどうか
                """
                # print("\n[DEBUG] update_from_image_metadata関数が実行されました")
                # print(translate("[DEBUG] メタデータ複写機能: {0}").format(copy_enabled))
                # print(f"[DEBUG] 受け取ったimage_path: {image_path}, 型: {type(image_path)}")
                # if isinstance(image_path, dict):
                #     print(f"[DEBUG] image_path keys: {image_path.keys()}")
                #     if 'path' in image_path:
                #         print(f"[DEBUG] image_path['path']: {image_path['path']}")

                # 複写機能が無効の場合は何もしない
                if not copy_enabled:
                    # print("[DEBUG] メタデータ複写機能が無効化されているため、処理をスキップします")
                    return [gr.update()] * 2

                if image_path is None:
                    # print("[DEBUG] image_pathはNoneです")
                    return [gr.update()] * 2

                # print(translate("[DEBUG] 画像パス: {0}").format(image_path))

                try:
                    # ファイルパスから直接メタデータを抽出
                    # print("[DEBUG] extract_metadata_from_pngをファイルパスから直接呼び出します")
                    metadata = extract_metadata_from_png(image_path)

                    if not metadata:
                        # print("[DEBUG] メタデータが抽出されませんでした")
                        print(translate("アップロードされた画像にメタデータが含まれていません"))
                        return [gr.update()] * 2

                    # print(translate("[DEBUG] メタデータサイズ: {0}, 内容: {1}").format(len(metadata), metadata))
                    print(translate("画像からメタデータを抽出しました: {0}").format(metadata))

                    # プロンプトとSEEDをUIに反映
                    prompt_update = gr.update()
                    seed_update = gr.update()

                    if PROMPT_KEY in metadata and metadata[PROMPT_KEY]:
                        prompt_update = gr.update(value=metadata[PROMPT_KEY])
                        # print(translate("[DEBUG] プロンプトを更新: {0}").format(metadata[PROMPT_KEY]))
                        print(translate("プロンプトを画像から取得: {0}").format(metadata[PROMPT_KEY]))

                    if SEED_KEY in metadata and metadata[SEED_KEY]:
                        # SEED値を整数に変換
                        try:
                            seed_value = int(metadata[SEED_KEY])
                            seed_update = gr.update(value=seed_value)
                            # print(translate("[DEBUG] SEED値を更新: {0}").format(seed_value))
                            print(translate("SEED値を画像から取得: {0}").format(seed_value))
                        except (ValueError, TypeError):
                            # print(translate("[DEBUG] SEED値の変換エラー: {0}").format(metadata[SEED_KEY]))
                            print(translate("SEED値の変換エラー: {0}").format(metadata[SEED_KEY]))

                    # print(translate("[DEBUG] 更新結果: prompt_update={0}, seed_update={1}").format(prompt_update, seed_update))
                    return [prompt_update, seed_update]
                except Exception as e:
                    # print(translate("[ERROR] メタデータ抽出処理中のエラー: {0}").format(e))
                    # traceback.print_exc()
                    print(translate("メタデータ抽出エラー: {0}").format(e))
                    return [gr.update()] * 2

            # 注意: イベント登録は変数定義後に行うため、後で実行する
            # メタデータ抽出処理の登録は、promptとseed変数の定義後に移動します

            # LoRA設定グループを追加
            with gr.Group(visible=has_lora_support) as lora_settings_group:
                gr.Markdown(f"### " + translate("LoRA設定"))

                # LoRA使用有無のチェックボックス
                use_lora = gr.Checkbox(label=translate("LoRAを使用する"), value=False, info=translate("チェックをオンにするとLoRAを使用します（要16GB VRAM以上）"))

                # LoRA読み込み方式を選択するラジオボタン
                lora_mode = gr.Radio(
                    choices=[translate("ディレクトリから選択"), translate("ファイルアップロード")],
                    value=translate("ディレクトリから選択"),
                    label=translate("LoRA読み込み方式"),
                    visible=False  # 初期状態では非表示
                )

                # ファイルアップロード方式のコンポーネント（グループ化）
                with gr.Group(visible=False) as lora_upload_group:
                    # メインのLoRAファイル
                    lora_files = gr.File(
                        label=translate("LoRAファイル (.safetensors, .pt, .bin)"),
                        file_types=[".safetensors", ".pt", ".bin"]
                    )
                    # 追加のLoRAファイル
                    lora_files2 = gr.File(
                        label=translate("LoRAファイル2 (.safetensors, .pt, .bin)"),
                        file_types=[".safetensors", ".pt", ".bin"]
                    )
                    # さらに追加のLoRAファイル
                    lora_files3 = gr.File(
                        label=translate("LoRAファイル3 (.safetensors, .pt, .bin)"),
                        file_types=[".safetensors", ".pt", ".bin"]
                    )

                # ディレクトリから選択方式のコンポーネント（グループ化）
                with gr.Group(visible=False) as lora_dropdown_group:
                    # ディレクトリからスキャンされたモデルのドロップダウン
                    lora_dropdown1 = gr.Dropdown(
                        label=translate("LoRAモデル選択 1"),
                        choices=[],
                        value=None,
                        allow_custom_value=False
                    )
                    lora_dropdown2 = gr.Dropdown(
                        label=translate("LoRAモデル選択 2"),
                        choices=[],
                        value=None,
                        allow_custom_value=False
                    )
                    lora_dropdown3 = gr.Dropdown(
                        label=translate("LoRAモデル選択 3"),
                        choices=[],
                        value=None,
                        allow_custom_value=False
                    )
                    # スキャンボタン
                    lora_scan_button = gr.Button(translate("LoRAディレクトリを再スキャン"), variant="secondary")
                
                # スケール値の入力フィールド（両方の方式で共通）
                lora_scales_text = gr.Textbox(
                    label=translate("LoRA適用強度 (カンマ区切り)"),
                    value="0.8,0.8,0.8",
                    info=translate("各LoRAのスケール値をカンマ区切りで入力 (例: 0.8,0.5,0.3)"),
                    visible=False
                )
                lora_blocks_type = gr.Dropdown(
                    label=translate("LoRAブロック選択"),
                    choices=["all", "single_blocks", "double_blocks", "db0-9", "db10-19", "sb0-9", "sb10-19", "important"],
                    value="all",
                    info=translate("選択するブロックタイプ（all=すべて、その他=メモリ節約）"),
                    visible=False
                )

                # LoRAディレクトリからモデルを検索する関数
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
                    
                    # 重要: すべての選択肢が確実に文字列型であることを確認
                    for i, choice in enumerate(choices):
                        if not isinstance(choice, str):
                            # 型変換のデバッグログ
                            print(translate("[DEBUG] 選択肢の型変換が必要: インデックス {0}, 型 {1}, 値 {2}").format(
                                i, type(choice).__name__, choice))
                            # 明示的に文字列に変換
                            choices[i] = str(choice)
                    
                    # ファイル内容のデバッグ出力を追加
                    print(translate("[INFO] LoRAディレクトリから{0}個のモデルを検出しました").format(len(choices) - 1))
                    print(translate("[DEBUG] 'なし'の値: {0!r}, 型: {1}").format(choices[0], type(choices[0]).__name__))
                    
                    # 一貫性があるか確認: "なし"がちゃんとに文字列型になっているか確認
                    if not isinstance(choices[0], str):
                        print(translate("[重要警告] 'なし'の選択肢が文字列型ではありません！型: {0}").format(type(choices[0]).__name__))
                    
                    # 数値の0に変換されないようにする
                    if choices[0] == 0 or choices[0] == 0.0:
                        print(translate("[重要警告] 'なし'の選択肢が数値0になっています。修正します。"))
                        choices[0] = none_choice
                    
                    # 戻り値のオブジェクトと型を表示
                    print(translate("[DEBUG] scan_lora_directory戻り値: 型={0}, 最初の要素={1!r}").format(
                        type(choices).__name__, choices[0] if choices else "なし"))
                    
                    return choices
                
                # チェックボックスの状態によって他のLoRA設定の表示/非表示を切り替える関数
                def toggle_lora_settings(use_lora):
                    if use_lora:
                        # LoRA使用時はデフォルトでディレクトリから選択モードを表示
                        choices = scan_lora_directory()
                        print(translate("[DEBUG] toggle_lora_settings - 選択肢リスト: {0}").format(choices))

                        # 選択肢がある場合は確実に文字列型に変換
                        # 型チェックを追加
                        for i, choice in enumerate(choices):
                            if not isinstance(choice, str):
                                print(translate("[DEBUG] toggle_lora_settings - 選択肢を文字列に変換: インデックス {0}, 元の値 {1}, 型 {2}").format(
                                    i, choice, type(choice).__name__))
                                choices[i] = str(choice)

                        # ドロップダウンが初期化時にも確実に更新されるようにする
                        # LoRAを有効にしたときにドロップダウンの選択肢も適切に更新
                        # まずモードを表示してからフラグを返す
                        return [
                            gr.update(visible=True),  # lora_mode
                            gr.update(visible=False),  # lora_upload_group - デフォルトでは非表示
                            gr.update(visible=True),  # lora_dropdown_group - デフォルトで表示
                            gr.update(visible=True),  # lora_scales_text
                        ]
                    else:
                        # LoRA不使用時はすべて非表示
                        return [
                            gr.update(visible=False),  # lora_mode
                            gr.update(visible=False),  # lora_upload_group
                            gr.update(visible=False),  # lora_dropdown_group
                            gr.update(visible=False),  # lora_scales_text
                        ]
                
                # LoRA読み込み方式に応じて表示を切り替える関数
                def toggle_lora_mode(mode):
                    if mode == translate("ディレクトリから選択"):
                        # ディレクトリから選択モードの場合
                        # 最初にディレクトリをスキャン
                        choices = scan_lora_directory()
                        print(translate("[DEBUG] toggle_lora_mode - 選択肢リスト: {0}").format(choices))
                        
                        # 選択肢の型を明示的に確認＆変換
                        for i, choice in enumerate(choices):
                            if not isinstance(choice, str):
                                print(translate("[DEBUG] toggle_lora_mode - 選択肢を文字列に変換: インデックス {0}, 元の値 {1}, 型 {2}").format(
                                    i, choice, type(choice).__name__))
                                choices[i] = str(choice)
                        
                        # 最初の選択肢がちゃんと文字列になっているか再確認
                        first_choice = choices[0]
                        print(translate("[DEBUG] toggle_lora_mode - 変換後の最初の選択肢: {0}, 型: {1}").format(
                            first_choice, type(first_choice).__name__
                        ))
                        
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
                    # LoRAドロップダウンの値を明示的に文字列として設定
                    print(translate("[DEBUG] LoRAドロップダウン更新 - 選択肢: {0}").format(choices))
                    print(translate("[DEBUG] 最初の選択肢: {0}, 型: {1}").format(choices[0], type(choices[0]).__name__))
                    
                    # すべての選択肢が確実に文字列型であることを確認
                    for i, choice in enumerate(choices):
                        if not isinstance(choice, str):
                            print(translate("[DEBUG] update_lora_dropdowns - 選択肢を文字列に変換: インデックス {0}, 値 {1}, 型 {2}").format(
                                i, choice, type(choice).__name__))
                            choices[i] = str(choice)
                    
                    # 各ドロップダウンを更新
                    print(translate("[DEBUG] update_lora_dropdowns - ドロップダウン更新完了。選択肢: {0}").format(choices))
                    
                    return [
                        gr.update(choices=choices, value=choices[0]),  # lora_dropdown1
                        gr.update(choices=choices, value=choices[0]),  # lora_dropdown2
                        gr.update(choices=choices, value=choices[0]),  # lora_dropdown3
                    ]
                    
                # UI初期化後のLoRAドロップダウン初期化関数
                def initialize_lora_dropdowns(use_lora_val):
                    # LoRAが有効で、「ディレクトリから選択」モードの場合のみ更新
                    if use_lora_val:
                        print(translate("[INFO] UIの初期化時にLoRAドロップダウンを更新します"))
                        return update_lora_dropdowns()
                    return [gr.update(), gr.update(), gr.update()]

                # 前回のLoRAモードを記憶するための変数
                previous_lora_mode = translate("ディレクトリから選択")  # デフォルトはディレクトリから選択
                
                # LoRA設定の変更を2ステップで行う関数
                def toggle_lora_full_update(use_lora_val):
                    # グローバル変数でモードを記憶
                    global previous_lora_mode
                    
                    # まずLoRA設定全体の表示/非表示を切り替え
                    # use_loraがオフの場合、まずモード値を保存
                    if not use_lora_val:
                        # モードの現在値を取得
                        current_mode = getattr(lora_mode, 'value', translate("ディレクトリから選択"))
                        if current_mode:
                            previous_lora_mode = current_mode
                            print(translate("[DEBUG] 前回のLoRAモードを保存: {0}").format(previous_lora_mode))
                    
                    # 表示/非表示の設定を取得
                    settings_updates = toggle_lora_settings(use_lora_val)
                    
                    # もしLoRAが有効になった場合
                    if use_lora_val:
                        print(translate("[DEBUG] LoRAが有効になりました。前回のモード: {0}").format(previous_lora_mode))
                        
                        # 前回のモードに基づいて表示を切り替え
                        if previous_lora_mode == translate("ファイルアップロード"):
                            # ファイルアップロードモードだった場合
                            print(translate("[DEBUG] 前回のモードはファイルアップロードだったため、ファイルアップロードUIを表示します"))
                            # モードの設定を上書き（ファイルアップロードに設定）
                            settings_updates[0] = gr.update(visible=True, value=translate("ファイルアップロード"))  # lora_mode
                            settings_updates[1] = gr.update(visible=True)   # lora_upload_group
                            settings_updates[2] = gr.update(visible=False)  # lora_dropdown_group
                            
                            # ドロップダウンは更新しない
                            return settings_updates + [gr.update(), gr.update(), gr.update()]
                        else:
                            # デフォルトまたはディレクトリから選択モードだった場合
                            choices = scan_lora_directory()
                            print(translate("[DEBUG] toggle_lora_full_update - LoRAドロップダウン選択肢: {0}").format(choices))
                            
                            # ドロップダウンの更新を行う
                            dropdown_updates = [
                                gr.update(choices=choices, value=choices[0]),  # lora_dropdown1
                                gr.update(choices=choices, value=choices[0]),  # lora_dropdown2
                                gr.update(choices=choices, value=choices[0])   # lora_dropdown3
                            ]
                            
                            # モードの設定を明示的に上書き
                            settings_updates[0] = gr.update(visible=True, value=translate("ディレクトリから選択"))  # lora_mode
                            return settings_updates + dropdown_updates
                    
                    # LoRAが無効な場合は設定の更新のみ
                    return settings_updates + [gr.update(), gr.update(), gr.update()]
                
                # チェックボックスの変更イベントにLoRA設定全体の表示/非表示を切り替える関数を紐づけ
                use_lora.change(
                    fn=toggle_lora_full_update,
                    inputs=[use_lora],
                    outputs=[lora_mode, lora_upload_group, lora_dropdown_group, lora_scales_text,
                             lora_dropdown1, lora_dropdown2, lora_dropdown3]
                )
                
                # LoRAモードの変更を処理する関数
                def toggle_lora_mode_with_memory(mode_value):
                    # グローバル変数に選択を保存
                    global previous_lora_mode
                    previous_lora_mode = mode_value
                    print(translate("[DEBUG] LoRAモードを変更: {0}").format(mode_value))
                    
                    # 標準のtoggle_lora_mode関数を呼び出し
                    return toggle_lora_mode(mode_value)
                
                # LoRA読み込み方式の変更イベントに表示切替関数を紐づけ
                lora_mode.change(
                    fn=toggle_lora_mode_with_memory,
                    inputs=[lora_mode],
                    outputs=[lora_upload_group, lora_dropdown_group, lora_dropdown1, lora_dropdown2, lora_dropdown3]
                )
                
                # スキャンボタンの処理を紐づけ
                lora_scan_button.click(
                    fn=update_lora_dropdowns,
                    inputs=[],
                    outputs=[lora_dropdown1, lora_dropdown2, lora_dropdown3]
                )
                
                # 代替の初期化方法：チェックボックスの初期値をチェックし、
                # LoRAドロップダウンを明示的に初期化する補助関数
                def lora_ready_init():
                    """LoRAドロップダウンの初期化を行う関数"""
                    print(translate("[INFO] LoRAドロップダウンの初期化を開始します"))
                    
                    # 現在のuse_loraとlora_modeの値を取得
                    use_lora_value = getattr(use_lora, 'value', False)
                    lora_mode_value = getattr(lora_mode, 'value', translate("ディレクトリから選択"))
                    
                    print(translate("[DEBUG] 初期化時の状態 - use_lora: {0}, lora_mode: {1}").format(
                        use_lora_value, lora_mode_value))
                    
                    # グローバル変数を更新
                    global previous_lora_mode
                    previous_lora_mode = lora_mode_value
                    
                    if use_lora_value:
                        # LoRAが有効な場合
                        if lora_mode_value == translate("ディレクトリから選択"):
                            # ディレクトリから選択モードの場合はドロップダウンを初期化
                            print(translate("[INFO] ディレクトリから選択モードでLoRAが有効なため、ドロップダウンを初期化します"))
                            choices = scan_lora_directory()
                            print(translate("[DEBUG] 初期化時のLoRA選択肢: {0}").format(choices))
                            return [
                                gr.update(choices=choices, value=choices[0]),  # lora_dropdown1
                                gr.update(choices=choices, value=choices[0]),  # lora_dropdown2
                                gr.update(choices=choices, value=choices[0])   # lora_dropdown3
                            ]
                        else:
                            # ファイルアップロードモードの場合はドロップダウンを更新しない
                            print(translate("[INFO] ファイルアップロードモードでLoRAが有効なため、ドロップダウンは更新しません"))
                            return [gr.update(), gr.update(), gr.update()]
                    
                    # LoRAが無効な場合は何も更新しない
                    return [gr.update(), gr.update(), gr.update()]
                
                # スキャンボタンの代わりにロード時の更新を行うボタン（非表示）
                lora_init_btn = gr.Button(visible=False, elem_id="lora_init_btn")
                lora_init_btn.click(
                    fn=lora_ready_init,
                    inputs=[],
                    outputs=[lora_dropdown1, lora_dropdown2, lora_dropdown3]
                )
                
                # UIロード後に自動的に初期化ボタンをクリックするJavaScriptを追加
                js_init_code = """
                function initLoraDropdowns() {
                    // UIロード後、少し待ってからボタンをクリック
                    setTimeout(function() {
                        // 非表示ボタンを探して自動クリック
                        var initBtn = document.getElementById('lora_init_btn');
                        if (initBtn) {
                            console.log('LoRAドロップダウン初期化ボタンを自動実行します');
                            initBtn.click();
                        } else {
                            console.log('LoRAドロップダウン初期化ボタンが見つかりません');
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

            prompt = gr.Textbox(label=translate("Prompt"), value=get_default_startup_prompt(), lines=6)

            # プロンプト管理パネルの追加
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

            # プリセットの説明文を削除

            # 互換性のためにQuick Listも残しておくが、非表示にする
            with gr.Row(visible=False):
                example_quick_prompts = gr.Dataset(samples=quick_prompts, label=translate("Quick List"), samples_per_page=1000, components=[prompt])
                example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            # 以下の設定ブロックは右カラムに移動しました

                # セクション設定のリストは既にアコーディオン内で初期化されています
                # section_number_inputs
                # section_image_inputs
                # section_prompt_inputs
                # section_row_groups

                # section_settingsは入力欄の値をまとめてリスト化
                def collect_section_settings(*args):
                    # args: [num1, img1, prompt1, num2, img2, prompt2, ...]
                    return [[args[i], args[i+1], args[i+2]] for i in range(0, len(args), 3)]

                section_settings = gr.State([[None, None, ""] for _ in range(max_keyframes)])
                section_inputs = []
                for i in range(max_keyframes):
                    section_inputs.extend([section_number_inputs[i], section_image_inputs[i], section_prompt_inputs[i]])

                # section_inputsをまとめてsection_settings Stateに格納
                def update_section_settings(*args):
                    return collect_section_settings(*args)

                # section_inputsが変化したらsection_settings Stateを更新
                for inp in section_inputs:
                    inp.change(fn=update_section_settings, inputs=section_inputs, outputs=section_settings)

                # フレームサイズ変更時の処理を追加
                def update_section_calculation(frame_size, mode, length):
                    """フレームサイズ変更時にセクション数を再計算して表示を更新"""
                    # 動画長を取得
                    seconds = get_video_seconds(length)

                    # latent_window_sizeを設定
                    latent_window_size = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                    frame_count = latent_window_size * 4 - 3

                    # セクション数を計算
                    total_frames = int(seconds * 30)
                    total_sections = int(max(round(total_frames / frame_count), 1))

                    # 計算詳細を表示するHTMLを生成
                    html = f"""<div style='padding: 10px; background-color: #f5f5f5; border-radius: 5px; font-size: 14px;'>
                    {translate('<strong>計算詳細</strong>: モード={0}, フレームサイズ={1}, 総フレーム数={2}, セクションあたり={3}フレーム, 必要セクション数={4}').format(length, frame_size, total_frames, frame_count, total_sections)}
                    <br>
                    {translate('動画モード {0} とフレームサイズ {1} で必要なセクション数: <strong>{2}</strong>').format(length, frame_size, total_sections)}
                    </div>"""

                    # デバッグ用ログ
                    print(translate("計算結果: モード={0}, フレームサイズ={1}, latent_window_size={2}, 総フレーム数={3}, 必要セクション数={4}").format(length, frame_size, latent_window_size, total_frames, total_sections))

                    return html

                # 初期化時にも計算を実行
                initial_html = update_section_calculation(frame_size_radio.value, mode_radio.value, length_radio.value)
                section_calc_display = gr.HTML(value=initial_html, label="")

                # フレームサイズ変更イベント - HTML表示の更新とセクションタイトルの更新を行う
                frame_size_radio.change(
                    fn=update_section_calculation,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_calc_display]
                )

                # フレームサイズ変更時にセクションタイトルも更新
                frame_size_radio.change(
                    fn=update_section_title,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_title]
                )

                # セクションの表示/非表示のみを制御する関数
                def update_section_visibility(mode, length, frame_size=None):
                    """画像は初期化せずにセクションの表示/非表示のみを制御する関数"""
                    # フレームサイズに基づくセクション数計算
                    seconds = get_video_seconds(length)
                    latent_window_size_value = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                    frame_count = latent_window_size_value * 4 - 3
                    total_frames = int(seconds * 30)
                    total_sections = int(max(round(total_frames / frame_count), 1))

                    # 通常モードの場合は全てのセクションの赤枠青枠を強制的にクリア
                    is_normal_mode = (mode == MODE_TYPE_NORMAL)
                    section_image_updates = []

                    print(translate("セクション視認性更新: モード={mode}, 長さ={length}, 必要セクション数={total_sections}").format(mode=mode, length=length, total_sections=total_sections))

                    # 現在のキーフレームコピー機能の状態を取得
                    # ループモードでのみキーフレームコピー機能が利用可能
                    # 常にチェックボックスの状態(Value)を尊重し、毎回自動的に有効化しない
                    # 以前の値をそのまま維持するために、現在のチェックボックス値を使用
                    is_copy_enabled = False
                    if not is_normal_mode:  # ループモードの場合のみ
                        try:
                            # チェックボックスの現在の値を取得（内部的な参照であり、イベント間で持続しない）
                            is_copy_enabled = keyframe_copy_checkbox.value
                            print(translate("[DEBUG] キーフレームコピー機能の状態: {state}").format(state=is_copy_enabled))
                        except Exception as e:
                            # エラーが発生した場合はデフォルト値を使用
                            print(translate("[ERROR] チェックボックス状態取得エラー: {error}").format(error=e))
                            is_copy_enabled = False

                    for i in range(len(section_image_inputs)):
                        if is_normal_mode:
                            # 通常モードではすべてのセクション画像の赤枠青枠を強制的にクリア
                            # 重要: 通常モードでは無条件で済む結果を返す
                            # print(translate("  セクション{i}: 通常モードなので赤枠/青枠を強制的にクリア").format(i=i))
                            section_image_updates.append(gr.update(elem_classes=""))  # 必ずelem_classesを空に設定
                        else:
                            # ループモードではセクション0と1に赤枠青枠を設定（ただしチェックボックスがオンのときのみ）
                            if not is_copy_enabled:
                                # チェックボックスがオフの場合は赤枠青枠を表示しない
                                section_image_updates.append(gr.update(elem_classes=""))
                            elif i == 0:
                                # print(translate("  セクション{i}: ループモードのセクション0に赤枠を設定").format(i=i))
                                section_image_updates.append(gr.update(elem_classes="highlighted-keyframe-red"))
                            elif i == 1:
                                # print(translate("  セクション{i}: ループモードのセクション1に青枠を設定").format(i=i))
                                section_image_updates.append(gr.update(elem_classes="highlighted-keyframe-blue"))
                            else:
                                # print(translate("  セクション{i}: ループモードの他セクションは空枠に設定").format(i=i))
                                section_image_updates.append(gr.update(elem_classes=""))

                    # 各セクションの表示/非表示のみを更新
                    section_row_updates = []
                    for i in range(len(section_row_groups)):
                        section_row_updates.append(gr.update(visible=(i < total_sections)))

                    # チェックボックス状態も維持する
                    # 出力リストには含まれていないため、ここではloggingのみ
                    print(translate("[DEBUG] 維持されるキーフレームコピー機能の状態: {state}").format(state=is_copy_enabled))

                    # 返値の設定 - input_imageとend_frameは更新せず
                    return [gr.update()] * 2 + section_image_updates + [gr.update(value=seconds)] + section_row_updates

                # 注意: この関数のイベント登録は、total_second_lengthのUIコンポーネント定義後に行うため、
                # ここでは関数の定義のみ行い、実際のイベント登録はUIコンポーネント定義後に行います。

                # 動画長変更イベントでもセクション数計算を更新
                length_radio.change(
                    fn=update_section_calculation,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_calc_display]
                )

                # 動画長変更時にセクションタイトルも更新
                length_radio.change(
                    fn=update_section_title,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_title]
                )

                # モード変更時にも計算を更新
                mode_radio.change(
                    fn=update_section_calculation,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_calc_display]
                )

                # モード変更時にセクションタイトルも更新
                mode_radio.change(
                    fn=update_section_title,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_title]
                )

                # モード変更時の処理もtotal_second_lengthコンポーネント定義後に行います

                # 動画長変更時のセクション表示更新もtotal_second_lengthコンポーネント定義後に行います

                # 入力画像変更時の処理 - ループモード用に復活
                # 通常モードでセクションにコピーする処理はコメント化したまま
                # ループモードのLastにコピーする処理のみ復活

                # 終端フレームハンドラ関数（FinalからImageへのコピーのみ実装）
                def loop_mode_final_handler(img, mode, length):
                    """end_frameの変更時、ループモードの場合のみコピーを行う関数"""
                    if img is None:
                        # 画像が指定されていない場合は何もしない
                        return gr.update()

                    # ループモードかどうかで処理を分岐
                    if mode == MODE_TYPE_LOOP:
                        # ループモード: ImageにFinalFrameをコピー
                        return gr.update(value=img)  # input_imageにコピー
                    else:
                        # 通常モード: 何もしない
                        return gr.update()

                # 終端フレームの変更ハンドラを登録
                end_frame.change(
                    fn=loop_mode_final_handler,
                    inputs=[end_frame, mode_radio, length_radio],
                    outputs=[input_image]
                )
                
                # エンドフレーム画像の変更をState変数に保存するハンドラを追加
                def update_end_frame_state(image_path):
                    # 画像パスをState変数に保存
                    print(f"[INFO] エンドフレーム画像パスをStateに保存: {image_path}")
                    return image_path
                
                # エンドフレーム変更時にStateも更新
                end_frame.change(
                    fn=update_end_frame_state,
                    inputs=[end_frame],
                    outputs=[end_frame_state]
                )

                # 各キーフレーム画像の変更イベントを個別に設定
                # 一度に複数のコンポーネントを更新する代わりに、個別の更新関数を使用
                def create_single_keyframe_handler(src_idx, target_idx):
                    def handle_single_keyframe(img, mode, length, enable_copy):
                        # ループモード以外では絶対にコピーを行わない
                        if mode != MODE_TYPE_LOOP:
                            # 通常モードでは絶対にコピーしない
                        #   print(translate("通常モードでのコピー要求を拒否: src={src_idx}, target={target_idx}").format(src_idx=src_idx, target_idx=target_idx))
                            return gr.update()

                        # コピー条件をチェック
                        if img is None or not enable_copy:
                            return gr.update()

                        # 現在のセクション数を動的に計算
                        seconds = get_video_seconds(length)
                        # フレームサイズに応じたlatent_window_sizeの調整（ここではUIの設定によらず計算）
                        frame_size = frame_size_radio.value
                        latent_window_size = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                        frame_count = latent_window_size * 4 - 3
                        total_frames = int(seconds * 30)
                        total_sections = int(max(round(total_frames / frame_count), 1))

                        # 対象セクションが有効範囲を超えている場合はコピーしない(項目数的に+1)
                        if target_idx >= total_sections:
                        #   print(translate("コピー対象セクション{target_idx}が有効範囲({total_sections}まで)を超えています").format(target_idx=target_idx, total_sections=total_sections))
                            return gr.update()

                        # コピー先のチェック - セクション0は偶数番号に、セクション1は奇数番号にコピー
                        if src_idx == 0 and target_idx % 2 == 0 and target_idx != 0:
                            # 詳細ログ出力
                        #   print(translate("赤枠(0)から偶数セクション{target_idx}へのコピー実行 (動的セクション数:{total_sections})").format(target_idx=target_idx, total_sections=total_sections))
                            return gr.update(value=img)
                        elif src_idx == 1 and target_idx % 2 == 1 and target_idx != 1:
                            # 詳細ログ出力
                        #   print(translate("青枠(1)から奇数セクション{target_idx}へのコピー実行 (動的セクション数:{total_sections})").format(target_idx=target_idx, total_sections=total_sections))
                            return gr.update(value=img)

                        # 条件に合わない場合
                        return gr.update()
                    return handle_single_keyframe

                # アップロードファイルの内容を各セクション、end_frame、start_frameに反映する関数
                # zipファイルアップロードハンドラを移動したモジュールを使用
                def handle_upload_zipfile(file):
                    # アップロードされたZIPファイルの内容を取得
                    result = upload_zipfile_handler(file, max_keyframes)
                    
                    # デバッグログを追加
                    # print(f"\n[DEBUG] ZIPファイルからの読み込み結果:")
                    
                    # グラフィカルコンポーネント用の値（最後の2つ）
                    # print(f"  エンドフレーム: {result[-2]}")
                    # print(f"  スタートフレーム: {result[-1]}")
                    
                    # セクションプロンプトの処理 - State変数に正しい値を設定
                    # これにより表示が適切になる
                    for i in range(max_keyframes):
                        # 3つずつ繰り返し（番号、プロンプト、画像）
                        section_idx = i * 3
                        if section_idx + 1 < len(result):
                            prompt_value = result[section_idx + 1]
                            # 直接値を取得し、それが文字列であることを確認
                            if isinstance(prompt_value, str):
                                # print(f"[DEBUG] セクション{i}のプロンプト値（修正前）: '{prompt_value}'")
                                # YAMLからのプロンプト値をセクションプロンプトのState変数に格納
                                # これにより、チェックボックスのオン/オフに関わらず常にYAMLの値が優先される
                                if i < len(section_prompt_states) and section_prompt_states[i] is not None:
                                    # 重要: ZIPからの値をStateに直接設定
                                    section_prompt_states[i].value = prompt_value
                                    # print(f"[IMPORTANT] セクション{i}のプロンプト値をStateに格納: '{prompt_value}'")
                            else:
                                pass
                                # print(f"[WARNING] セクション{i}のプロンプト値が文字列ではありません: {prompt_value} (型: {type(prompt_value)})")
                    
                    # 内部データに含まれる追加情報（section_manager.pyにのみ保存）
                    # section_infoに含まれているデフォルトプロンプトとSEED値
                    # これらの値はUIに反映されないが、データとしては存在する
                    
                    # Stateも更新（エンドフレームがある場合）
                    if result[-2] is not None:
                        # print(f"[IMPORTANT] エンドフレーム({result[-2]})をStateにも保存します")
                        # デバッグ: エンドフレーム画像の存在確認
                        try:
                            if os.path.exists(result[-2]):
                                # print(f"[VERIFIED] エンドフレーム画像のパスが有効です: {result[-2]}")
                                pass
                            else:
                                # print(f"[WARNING] エンドフレーム画像のパスが存在しません: {result[-2]}")
                                # 相対パスを絶対パスに変換して試行
                                abs_path = os.path.abspath(result[-2])
                                # print(f"[RETRY] 絶対パスでの確認: {abs_path}")
                                if os.path.exists(abs_path):
                                    # print(f"[FIXED] 絶対パスでエンドフレーム画像を見つけました: {abs_path}")
                                    # 絶対パスに置き換え
                                    result[-2] = abs_path
                        except Exception as e:
                            # print(f"[ERROR] エンドフレームパス検証エラー: {e}")
                            pass
                        
                        # 直接エンドフレームコンポーネントを更新
                        try:
                            # 画像の読み込みを試行
                            from PIL import Image
                            img = Image.open(result[-2])
                            # print(f"[SUCCESS] エンドフレーム画像を読み込み: サイズ={img.size}")
                            # ファイルを読み込んで明示的に設定
                            end_frame.value = result[-2]
                            # print(f"[DIRECT] エンドフレームコンポーネントに直接設定: {result[-2]}")
                        except Exception as e:
                            # print(f"[ERROR] エンドフレーム画像読み込みエラー: {e}")
                            pass
                    else:
                        pass
                        # print(f"[WARNING] エンドフレーム画像が見つかりませんでした")
                        
                    # スタートフレームがある場合
                    if result[-1] is not None:
                        # print(f"[IMPORTANT] スタートフレーム({result[-1]})をStateにも保存します")
                        # デバッグ: スタートフレーム画像の存在確認
                        try:
                            if os.path.exists(result[-1]):
                                # print(f"[VERIFIED] スタートフレーム画像のパスが有効です: {result[-1]}")
                                pass
                            else:
                                # print(f"[WARNING] スタートフレーム画像のパスが存在しません: {result[-1]}")
                                # 相対パスを絶対パスに変換して試行
                                abs_path = os.path.abspath(result[-1])
                                # print(f"[RETRY] 絶対パスでの確認: {abs_path}")
                                if os.path.exists(abs_path):
                                    # print(f"[FIXED] 絶対パスでスタートフレーム画像を見つけました: {abs_path}")
                                    # 絶対パスに置き換え
                                    result[-1] = abs_path
                        except Exception as e:
                            # print(f"[ERROR] スタートフレームパス検証エラー: {e}")
                            pass
                        
                        # 直接スタートフレームコンポーネントを更新
                        try:
                            # 画像の読み込みを試行
                            from PIL import Image
                            img = Image.open(result[-1])
                            # print(f"[SUCCESS] スタートフレーム画像を読み込み: サイズ={img.size}")
                            # ファイルを直接設定
                            input_image.value = result[-1]
                            # print(f"[DIRECT] スタートフレームコンポーネントに直接設定: {result[-1]}")
                        except Exception as e:
                            # print(f"[ERROR] スタートフレーム画像読み込みエラー: {e}")
                            pass
                        
                    # デフォルトプロンプトとSEED値が存在する場合、コンソールに表示するのみ
                    # ただし、変数は保持しておく
                    default_prompt_value = ""
                    seed_value = -1  # デフォルト値
                    
                    # デフォルトプロンプトがある場合（-2の位置）
                    if len(result) > 2 and result[-2] is not None and result[-2] != "":
                        default_prompt_value = result[-2]
                        # print(f"[IMPORTANT] デフォルトプロンプト '{default_prompt_value}' を検出しました")
                        # print(f"[INFO] UIにプロンプトを直接設定するには、次のコマンドを使用: prompt.value = '{default_prompt_value}'")
                        # メッセージのみ表示し、実際の設定はしない（スコープ外アクセスを避けるため）
                    
                    # SEED値がある場合（-1の位置）
                    if len(result) > 1 and result[-1] is not None and isinstance(result[-1], (int, float, str)):
                        try:
                            # 数値に変換（文字列の場合）
                            if isinstance(result[-1], str) and result[-1].strip() == "":
                                seed_value = -1  # 空文字列の場合はデフォルト値に
                            else:
                                seed_value = int(result[-1])
                            # print(f"[IMPORTANT] SEED値 {seed_value} を検出しました")
                            # print(f"[INFO] UIにシード値を直接設定するには、次のコマンドを使用: seed.value = {seed_value}")
                            # メッセージのみ表示し、実際の設定はしない（スコープ外アクセスを避けるため）
                        except (ValueError, TypeError) as e:
                            # print(f"[ERROR] SEED値の変換エラー: {e}, 値: {result[-1]}, 型: {type(result[-1])}")
                            pass
                    
                    # これらの値はJavaScriptを使って設定することも可能
                    # print(f"[INFO] 将来的な改良: ブラウザ側のJavaScriptを使用して値を設定することを検討")
                    # これはクライアントサイドのJavaScriptで処理するべき内容
                    
                    return result

                # ファイルアップロード時のセクション変更
                gr_outputs = []
                # セクション情報
                for i in range(0, max_keyframes):
                    gr_outputs.append(section_number_inputs[i])
                    gr_outputs.append(section_prompt_inputs[i])
                    gr_outputs.append(section_image_inputs[i])
                
                # 末尾に追加（順序が重要）
                # end_frameを設定（重要：名前と順序を正確に）
                gr_outputs.append(end_frame)
                # start_frameを設定（重要：名前と順序を正確に）
                gr_outputs.append(input_image)
                
                # デフォルトプロンプトと種値は別の方法で処理する
                # gr_outputsには含めない
                
                # 詳細デバッグログ
                # print(f"[DEBUG-GR] gr_outputs最後の2つ: {gr_outputs[-2:]}")
                # print(f"[DEBUG-GR] end_frame型: {type(end_frame)}")
                # print(f"[DEBUG-GR] end_frameコンポーネント: {end_frame}")
                
                # ZIPファイルアップロード時のイベント設定
                upload_zipfile.change(
                    fn=handle_upload_zipfile, 
                    inputs=[upload_zipfile], 
                    outputs=gr_outputs
                )
                
                # ダウンロードボタン用のハンドラ関数を定義
                def handle_download_sections(end_frame_state_value, input_image_state_value):
                    # セクション設定情報を取得
                    section_settings = []
                    
                    # 現在表示されているセクション数を取得
                    current_sections_count = get_current_sections_count()
                    # print(f"[INFO] 現在表示されているセクション数: {current_sections_count}")
                    
                    # State変数の値をデバッグ出力
                    # print(f"[INFO] State変数: end_frame_state = {end_frame_state_value}")
                    # print(f"[INFO] State変数: input_image_state = {input_image_state_value}")
                    
                    # デバッグ: セクション入力欄の状態を確認
                    if debug_section_inputs:
                        # print("\n[SECTION DEBUG] セクション入力欄の詳細情報：")
                        for i in range(min(current_sections_count, 5)):  # 表示されている中から最初の5つだけチェック
                            input_obj = section_prompt_inputs[i]
                            # print(f"セクション{i} プロンプト入力: {input_obj}, ID={id(input_obj)}")
                            # print(f"  値: {input_obj.value}, 型: {type(input_obj.value)}")
                            # 値の状態を確認するだけ（強制設定はしない）
                            # if input_obj.value is None:
                            #     print(f"  値はNone")
                            # elif input_obj.value == "":
                            #     print(f"  値は空文字")
                            
                            # print(f"セクション{i} 画像入力: {section_image_inputs[i]}, 値: {section_image_inputs[i].value}")
                            # print(f"セクション{i} 画像State値: {section_image_states[i].value}")
                            pass
                    
                    # ユーザーが入力したプロンプト値をそのまま使用
                    # 表示されているセクションのみを処理
                    for i in range(current_sections_count):
                        # 詳細なデバッグ出力
                        # print(f"セクション {i}: 画像={section_image_inputs[i].value}, プロンプト={section_prompt_inputs[i].value}")
                        # print(f"セクション {i} の画像State値: {section_image_states[i].value}")
                        # print(f"セクション {i} のプロンプトState値: {section_prompt_states[i].value}")
                        
                        # 画像値の優先順位: 
                        # 1. section_image_states[i].valueが存在する場合はそれを使用
                        # 2. section_image_inputs[i].valueが存在する場合はそれを使用
                        # 3. どちらもなければNone
                        img_value = None
                        stored_state = section_image_states[i].value
                        ui_value = section_image_inputs[i].value
                        
                        # UIコンポーネントとStateの値の詳細ログ出力
                        # print(f"[DEBUG-DETAIL] セクション{i}のUI値: {ui_value}, 型: {type(ui_value)}")
                        # print(f"[DEBUG-DETAIL] セクション{i}のState値: {stored_state}, 型: {type(stored_state)}")
                        
                        if stored_state is not None:
                            img_value = stored_state
                            # print(f"[DETAIL] セクション{i}の画像はStateから取得: {img_value}")
                        elif ui_value is not None:
                            img_value = ui_value
                            # print(f"[DETAIL] セクション{i}の画像はUIコンポーネントから取得: {img_value}")
                            # UIから値を取得した場合、Stateにも保存する（安全策）
                            section_image_states[i].value = ui_value
                            # print(f"[DETAIL] セクション{i}の画像値をStateに保存しました: {ui_value}")
                        
                        # 画像の詳細情報を出力
                        # print(f"[DETAIL] セクション{i}の画像情報: タイプ={type(img_value)}")
                        # if img_value is None:
                        #     print(f"[DETAIL] セクション{i}の画像はNoneです")
                        # elif isinstance(img_value, dict):
                        #     print(f"[DETAIL] セクション{i}の画像は辞書型です: キー={img_value.keys()}")
                        #     if 'path' in img_value:
                        #         print(f"[DETAIL] セクション{i}の画像パス: {img_value['path']}")
                        #     if 'name' in img_value:
                        #         print(f"[DETAIL] セクション{i}の画像名: {img_value['name']}")
                        # elif isinstance(img_value, str):
                        #     print(f"[DETAIL] セクション{i}の画像は文字列です: {img_value}")
                        # else:
                        #     print(f"[DETAIL] セクション{i}の画像はその他の型です: {img_value}")
                        
                        # プロンプト値を取得（最大限のエラー対策）
                        # プロンプト値の優先順位（重要な更新）:
                        # 1. カスタム関数で直接セクションプロンプトの値を取得（最優先）
                        # 2. section_prompt_inputs[i].valueが存在する場合はそれを使用（次に優先）
                        # 3. section_prompt_states[i].valueが存在する場合はそれを使用（UIに表示されていない場合）
                        # 4. どれもなければ空文字列
                        prompt_value = ""
                        try:
                            # 直接プロンプト入力欄の値を確認（最も信頼性が高い方法）
                            direct_ui_value = None
                            try:
                                # 直接DOMから値を取得する試み
                                if section_prompt_inputs[i] is not None:
                                    # 直接テキストボックスの値を取得
                                    if hasattr(section_prompt_inputs[i], 'value'):
                                        direct_ui_value = section_prompt_inputs[i].value
                                        # print(f"[DEBUG-DIRECT] セクション{i}の直接取得値: '{direct_ui_value}'")
                            except Exception as e:
                                # print(f"[ERROR-DIRECT] セクション{i}の直接値取得エラー: {e}")
                                pass
                            
                            # UI入力値（標準的な方法）
                            ui_value = None
                            if section_prompt_inputs[i] is not None:
                                # 直接値を取得して空文字列の場合もチェック
                                try:
                                    if hasattr(section_prompt_inputs[i], 'value'):
                                        ui_value = section_prompt_inputs[i].value
                                    # 念のため値が空でないか確認
                                    if ui_value == "" or ui_value is None:
                                        # 更に試行：最新の値を取得
                                        ui_value = section_prompt_inputs[i].get_value() if hasattr(section_prompt_inputs[i], 'get_value') else None
                                        # print(f"[DEBUG-EXTRA] セクション{i}のget_value()で取得した値: '{ui_value}'")
                                except Exception as e:
                                    # print(f"[DEBUG-ERROR] セクション{i}のUI値取得エラー: {e}")
                                    pass
                            
                            # State値を取得
                            state_value = section_prompt_states[i].value
                            # print(f"[DEBUG-STATE] セクション{i}のState値: '{state_value}'") # デバッグ追加
                            
                            # print(f"[DEBUG-DETAIL] セクション{i}の取得結果: 直接UI='{direct_ui_value}', 標準UI='{ui_value}', State='{state_value}'")
                            
                            # 優先順位に従ってプロンプト値を決定
                            # 1. State値を最優先（ユーザーの編集が直接反映される）
                            if state_value and str(state_value).strip() and not str(state_value).startswith('{'):
                                prompt_value = str(state_value)
                                # print(f"[DETAIL] セクション{i}のプロンプトはStateから取得（最優先）: '{prompt_value}'")
                            # 2. 直接取得した値
                            elif direct_ui_value and str(direct_ui_value).strip() and not str(direct_ui_value).startswith('{'):
                                prompt_value = str(direct_ui_value)
                                # print(f"[DETAIL] セクション{i}のプロンプトは直接UIから取得: '{prompt_value}'")
                                # この値をStateにも保存（反映には時間がかかる可能性）
                                section_prompt_states[i].value = prompt_value
                            # 3. 標準的なUI値
                            elif ui_value and str(ui_value).strip() and not str(ui_value).startswith('{'):
                                prompt_value = str(ui_value)
                                # print(f"[DETAIL] セクション{i}のプロンプトは標準UIから取得: '{prompt_value}'")
                                # UIからの値をStateにも保存
                                section_prompt_states[i].value = prompt_value
                            # else:
                                # print(f"[WARNING] セクション{i}のプロンプト値はすべてのソースで無効または空です")
                        except Exception as e:
                            # print(f"[ERROR] セクション{i}のプロンプト取得エラー: {e}")
                            pass
                        
                        # デバッグ情報追加
                        # print(f"[DEBUG] セクション{i}のプロンプト値: '{prompt_value}'")
                        
                        # デバッグ用に詳細出力（セクション0と1）
                        # if i <= 1:
                        #     print(f"セクション{i}のプロンプト: '{prompt_value}', 型: {type(prompt_value)}")
                            
                        # セクション情報に追加
                        section_data = [
                            i,                 # セクション番号
                            img_value,         # セクション画像（Stateから優先取得）
                            prompt_value       # セクションプロンプト（Stateから優先取得）
                        ]
                        section_settings.append(section_data)
                        # print(f"[DEBUG] セクション{i}をsection_settingsに追加: {section_data}")
                    
                    # セクション0と1が含まれていることを確認（特に重要）
                    has_section0 = any(row[0] == 0 for row in section_settings)
                    has_section1 = any(row[0] == 1 for row in section_settings)
                    # print(f"[INFO] セクション0が処理対象に含まれているか: {has_section0}")
                    # print(f"[INFO] セクション1が処理対象に含まれているか: {has_section1}")
                    
                    # endframeとstart_frameもデバッグ出力（State値を優先）
                    # print(f"End Frame: {end_frame_state_value if end_frame_state_value is not None else end_frame.value}")
                    # print(f"Start Frame: {input_image_state_value if input_image_state_value is not None else input_image.value}")
                    
                    # 開始フレームの詳細をデバッグ出力
                    start_frame_value = input_image_state_value if input_image_state_value is not None else input_image.value
                    # if start_frame_value is None:
                    #     print("[DEBUG] 開始フレーム画像はNoneなので、詳細確認ができません")
                    # elif isinstance(start_frame_value, dict):
                    #     print(f"[DEBUG] 開始フレーム画像のキー: {start_frame_value.keys()}")
                    # print(f"デフォルトプロンプト: {prompt.value}")
                    
                    # zipファイルを生成
                    # Gradioデバッグ出力
                    # print(f"end_frame コンポーネント: {end_frame}")
                    # print(f"input_image コンポーネント: {input_image}")
                    
                    # デフォルトプロンプトとシードをセクション設定に含める
                    additional_info = {
                        "default_prompt": prompt.value,
                        "seed": seed.value  # 現在のシード値を追加
                    }
                    
                    # zipファイルを生成（input_imageをstart_frameとして扱う）
                    # プロンプトのデバッグ出力
                    # for i, row in enumerate(section_settings):
                    #     if i == 0:
                    #         print(f"セクション0のプロンプトを確認: '{row[2] if len(row) > 2 else None}'")
                    #     if i == 1:
                    #         print(f"セクション1のプロンプトを確認: '{row[2] if len(row) > 2 else None}'")
                    
                    # 処理対象セクション数を表示
                    # print(f"[INFO] 処理対象のセクション数: {len(section_settings)}")
                    
                    # 重要: UI値とState値を比較して、State値を優先
                    end_frame_value = end_frame_state_value if end_frame_state_value is not None else end_frame.value
                    input_image_value = input_image_state_value if input_image_state_value is not None else input_image.value
                    
                    # print(f"[INFO] ダウンロードに使用するエンドフレーム: {end_frame_value}")
                    # print(f"[INFO] ダウンロードに使用する開始フレーム: {input_image_value}")
                    
                    zip_path = download_zipfile_handler(section_settings, end_frame_value, input_image_value, additional_info)
                    # print(f"生成されたZIPパス: {zip_path}")
                    return gr.update(value=zip_path, visible=True)
                
                # ダウンロードボタンのクリックイベントを登録
                download_sections_button.click(
                    fn=handle_download_sections,
                    inputs=[end_frame_state, input_image_state],  # State変数も入力として受け取る
                    outputs=[download_file]
                )

                # 各キーフレームについて、影響を受ける可能性のある後続のキーフレームごとに個別のイベントを設定
                # ここではイベント登録の定義のみ行い、実際の登録はUIコンポーネント定義後に行う

                # キーフレーム自動コピーの初期値はStateでデフォルトでTrueに設定済み
                # enable_keyframe_copyは既にTrueに初期化されているのでここでは特に何もしない

                # モード切り替え時に赤枠/青枠の表示を切り替える関数
                # トグル関数は不要になったため削除
                # 代わりにcheckbox値のみに依存するシンプルな条件分岐を各関数で直接実装

        with gr.Column():
            result_video = gr.Video(
                label=translate("Finished Frames"),
                autoplay=True,
                show_share_button=False,
                height=512,
                loop=True,
                format="mp4",
                interactive=False,
            )
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            preview_image = gr.Image(label=translate("Next Latents"), height=200, visible=False)

            # フレームサイズ切替用のUIコントロールは上部に移動したため削除

            # 計算結果を表示するエリア
            section_calc_display = gr.HTML("", label="")

            use_teacache = gr.Checkbox(label=translate('Use TeaCache'), value=True, info=translate('Faster speed, but often makes hands and fingers slightly worse.'))
            
            # VAEキャッシュ設定
            use_vae_cache = gr.Checkbox(
                label=translate('VAEキャッシュを使用'),
                value=False,
                info=translate('デコードを1フレームずつ処理し、速度向上（メモリ使用量増加。VRAM24GB以上推奨。それ以下の場合、メモリスワップで逆に遅くなります）')
            )
            print(f"VAEキャッシュCheckbox初期化: id={id(use_vae_cache)}")
            
            # グローバル変数に現在の値を保存するためのイベントハンドラ
            # グローバル変数vae_cache_enabledはファイル先頭で既に宣言済み
            
            def update_vae_cache_state(value):
                global vae_cache_enabled
                vae_cache_enabled = value
                print(f"VAEキャッシュ状態を更新: {vae_cache_enabled}")
                return None
                
            # チェックボックスの状態が変更されたときにグローバル変数を更新
            use_vae_cache.change(fn=update_vae_cache_state, inputs=[use_vae_cache], outputs=[])

            # Use Random Seedの初期値
            use_random_seed_default = True
            seed_default = random.randint(0, 2**32 - 1) if use_random_seed_default else 1

            use_random_seed = gr.Checkbox(label=translate("Use Random Seed"), value=use_random_seed_default)

            n_prompt = gr.Textbox(label=translate("Negative Prompt"), value="", visible=False)  # Not used
            seed = gr.Number(label=translate("Seed"), value=seed_default, precision=0)

            # ここで、メタデータ取得処理の登録を移動する
            # ここでは、promptとseedの両方が定義済み
            input_image.change(
                fn=update_from_image_metadata,
                inputs=[input_image, copy_metadata],
                outputs=[prompt, seed]
            )
            
            # 開始フレーム画像の変更をState変数に保存するハンドラを追加
            def update_input_image_state(image_path):
                # 画像パスをState変数に保存
                # print(f"[INFO] 開始フレーム画像パスをStateに保存: {image_path}")
                return image_path
            
            # 開始フレーム変更時にStateも更新
            input_image.change(
                fn=update_input_image_state,
                inputs=[input_image],
                outputs=[input_image_state]
            )

            # チェックボックスの変更時に再読み込みを行う
            def check_metadata_on_checkbox_change(copy_enabled, image_path):
                if not copy_enabled or image_path is None:
                    return [gr.update()] * 2
                # チェックボックスオン時に、画像があれば再度メタデータを読み込む
                return update_from_image_metadata(image_path, copy_enabled)

            # セクション画像のメタデータをチェックボックス変更時に再読み込みする関数
            def update_section_metadata_on_checkbox_change(copy_enabled, *section_images):
                if not copy_enabled:
                    # チェックボックスがオフの場合は、何も変更せずに現在の値を維持する
                    print("[INFO] チェックボックスがオフのため、現在のプロンプト値を維持します")
                    
                    # gr.updateの配列を返す - valueを指定しないとUI値が維持される
                    updates = []
                    for _ in range(max_keyframes):
                        updates.append(gr.update())
                    
                    return updates

                # 各セクションの画像があれば、それぞれのメタデータを再取得する
                updates = []
                for i, section_image in enumerate(section_images):
                    if section_image is not None:
                        # セクションメタデータハンドラを直接利用してメタデータを取得
                        # section_prompt_inputsからセクションのプロンプト欄を取得して渡す
                        if i < len(section_prompt_inputs):
                            section_prompt_input = section_prompt_inputs[i]
                        else:
                            section_prompt_input = None
                            
                        handler = create_section_metadata_handler(i, section_prompt_input)
                        # メタデータを取得 - 戻り値の最初の要素（プロンプト値）のみを使用
                        update_result = handler(section_image, copy_enabled)
                        
                        # update_resultは(プロンプト値, 画像パス)のタプル
                        # プロンプト値のみをリストに追加
                        if isinstance(update_result, tuple) and len(update_result) > 0:
                            prompt_value = update_result[0]
                            # gr.update()の場合は空文字列に置き換え
                            if hasattr(prompt_value, '__type__') and prompt_value.__type__ == 'update':
                                prompt_value = ""
                            updates.append(prompt_value)
                        else:
                            # 予期せぬ戻り値の場合は空文字列
                            updates.append("")
                    else:
                        # 画像がなければ空文字列
                        updates.append("")

                # 不足分を追加
                while len(updates) < max_keyframes:
                    updates.append("")

                return updates[:max_keyframes]

            copy_metadata.change(
                fn=check_metadata_on_checkbox_change,
                inputs=[copy_metadata, input_image],
                outputs=[prompt, seed]
            )

            # セクション画像のメタデータを再読み込みするイベントを追加
            copy_metadata.change(
                fn=update_section_metadata_on_checkbox_change,
                inputs=[copy_metadata] + section_image_inputs,
                outputs=section_prompt_inputs
            )

            def set_random_seed(is_checked):
                if is_checked:
                    return random.randint(0, 2**32 - 1)
                else:
                    return gr.update()
            use_random_seed.change(fn=set_random_seed, inputs=use_random_seed, outputs=seed)

            total_second_length = gr.Slider(label=translate("Total Video Length (Seconds)"), minimum=1, maximum=120, value=1, step=1)
            latent_window_size = gr.Slider(label=translate("Latent Window Size"), minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
            steps = gr.Slider(label=translate("Steps"), minimum=1, maximum=100, value=25, step=1, info=translate('Changing this value is not recommended.'))

            cfg = gr.Slider(label=translate("CFG Scale"), minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
            gs = gr.Slider(label=translate("Distilled CFG Scale"), minimum=1.0, maximum=32.0, value=10.0, step=0.01, info=translate('Changing this value is not recommended.'))
            rs = gr.Slider(label=translate("CFG Re-Scale"), minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

            available_cuda_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
            default_gpu_memory_preservation_gb = 6 if available_cuda_memory_gb >= 20 else (8 if available_cuda_memory_gb > 16 else 10)
            gpu_memory_preservation = gr.Slider(label=translate("GPU Memory to Preserve (GB) (smaller = more VRAM usage)"), minimum=6, maximum=128, value=default_gpu_memory_preservation_gb, step=0.1, info=translate("空けておくGPUメモリ量を指定。小さい値=より多くのVRAMを使用可能=高速、大きい値=より少ないVRAMを使用=安全"))

            # MP4圧縮設定スライダーを追加
            mp4_crf = gr.Slider(label=translate("MP4 Compression"), minimum=0, maximum=100, value=16, step=1, info=translate("数値が小さいほど高品質になります。0は無圧縮。黒画面が出る場合は16に設定してください。"))
            
            # VAEタイリング設定（ゴースト対策）
            from eichi_utils import create_vae_settings_ui, get_current_vae_settings_display
            vae_settings_accordion, vae_controls = create_vae_settings_ui(translate)
            
            # VAEの実際の設定値を表示する関数を実装
            def update_vae_settings_display():
                global vae
                if vae is not None:
                    current_settings = get_current_vae_settings_display(vae)
                    return current_settings
                return "VAEがロードされていません"
            
            # 初回表示時に実行
            vae_controls['current_settings_md'].value = update_vae_settings_display()

            # セクションごとの動画保存チェックボックスを追加（デフォルトOFF）
            keep_section_videos = gr.Checkbox(label=translate("完了時にセクションごとの動画を残す - チェックがない場合は最終動画のみ保存されます（デフォルトOFF）"), value=False)

            # テンソルデータ保存チェックボックス违加
            save_tensor_data = gr.Checkbox(
                label=translate("完了時にテンソルデータ(.safetensors)も保存 - このデータを別の動画の後に結合可能"),
                value=False,
                info=translate("チェックすると、生成されたテンソルデータを保存します。アップロードされたテンソルがあれば、結合したテンソルデータも保存されます。")
            )

            # セクションごとの静止画保存チェックボックスを追加（デフォルトOFF）
            save_section_frames = gr.Checkbox(label=translate("Save Section Frames"), value=False, info=translate("各セクションの最終フレームを静止画として保存します（デフォルトOFF）"))
            
            # フレーム画像保存のラジオボタンを追加（デフォルトは「保存しない」）
            # gr.Groupで囲むことで灰色背景のスタイルに統一
            with gr.Group():
                gr.Markdown(f"### " + translate("フレーム画像保存設定"))
                frame_save_mode = gr.Radio(
                    label=translate("フレーム画像保存モード"),
                    choices=[
                        translate("保存しない"),
                        translate("全フレーム画像保存"),
                        translate("最終セクションのみ全フレーム画像保存")
                    ],
                    value=translate("保存しない"),
                    info=translate("フレーム画像の保存方法を選択します。過去セクション分も含めて保存します。全セクションか最終セクションのみか選択できます。")
                )

            # UIコンポーネント定義後のイベント登録
            # mode_radio.changeの登録 - セクションの表示/非表示と赤枠青枠の表示を同時に更新
            mode_radio.change(
                fn=update_section_visibility,
                inputs=[mode_radio, length_radio, frame_size_radio],
                outputs=[input_image, end_frame] + section_image_inputs + [total_second_length] + section_row_groups
            )

            # 設定変更時にキーフレームコピー機能の状態を取得して保持する関数
            def update_section_preserve_checkbox(mode, length, frame_size, copy_state):
                """セクションの表示/非表示を更新し、キーフレームコピー機能の状態を維持する"""
                # 通常のセクション表示/非表示の更新を行う
                updates = update_section_visibility(mode, length, frame_size)
                
                # ループモードの場合のみキーフレームコピー機能が利用可能
                is_loop_mode = (mode == MODE_TYPE_LOOP)
                
                # 現在の状態を維持（ループモードでない場合は常にFalse）
                preserved_state = copy_state if is_loop_mode else False
                
                # 赤枠/青枠の表示状態を更新（キーフレームコピー機能の状態に応じて）
                section_updates = []
                for i in range(len(section_image_inputs)):
                    if is_loop_mode and preserved_state:
                        if i == 0:
                            # セクション0は赤枠
                            section_updates.append(gr.update(elem_classes="highlighted-keyframe-red"))
                        elif i == 1:
                            # セクション1は青枠
                            section_updates.append(gr.update(elem_classes="highlighted-keyframe-blue"))
                        else:
                            section_updates.append(gr.update(elem_classes=""))
                    else:
                        # 通常モードまたはキーフレームコピー機能オフの場合は枠を非表示
                        section_updates.append(gr.update(elem_classes=""))
                
                # キーフレームコピー機能のチェックボックス状態を更新
                copy_checkbox_update = gr.update(value=preserved_state)
                
                # total_second_lengthの更新
                seconds = get_video_seconds(length)
                total_second_update = gr.update(value=seconds)
                
                # セクション行の表示/非表示を計算
                latent_window_size_value = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                frame_count = latent_window_size_value * 4 - 3
                total_frames = int(seconds * 30)
                total_sections = int(max(round(total_frames / frame_count), 1))
                
                section_row_updates = []
                for i in range(len(section_row_groups)):
                    section_row_updates.append(gr.update(visible=(i < total_sections)))
                
                # 返値を構築
                return [gr.update(), gr.update()] + section_updates + [total_second_update] + [copy_checkbox_update] + section_row_updates
            
            # frame_size_radio.changeの登録 - セクションの表示/非表示とキーフレームコピーの状態を維持
            frame_size_radio.change(
                fn=update_section_preserve_checkbox,
                inputs=[mode_radio, length_radio, frame_size_radio, enable_keyframe_copy],
                outputs=[input_image, end_frame] + section_image_inputs + [total_second_length] + [keyframe_copy_checkbox] + section_row_groups
            )

            # length_radio.changeの登録 - セクションの表示/非表示とキーフレームコピーの状態を維持
            length_radio.change(
                fn=update_section_preserve_checkbox,
                inputs=[mode_radio, length_radio, frame_size_radio, enable_keyframe_copy],
                outputs=[input_image, end_frame] + section_image_inputs + [total_second_length] + [keyframe_copy_checkbox] + section_row_groups
            )

            # mode_radio.changeの登録 - 拡張モード変更ハンドラを使用
            # モード変更時は個別にキーフレームコピー機能のチェックボックスを更新する
            def update_mode_and_checkbox_state(mode, length):
                # 拡張モード変更ハンドラを呼び出してセクション表示を更新
                section_updates = extended_mode_length_change_handler(
                    mode, length, section_number_inputs, section_row_groups, frame_size_radio.value
                )
                
                # ループモードならチェックボックスをオン、通常モードならオフにする
                is_loop_mode = (mode == MODE_TYPE_LOOP)
                checkbox_update = gr.update(value=is_loop_mode, visible=is_loop_mode)
                
                return section_updates + [checkbox_update]
                
            mode_radio.change(
                fn=update_mode_and_checkbox_state,
                inputs=[mode_radio, length_radio],
                outputs=[input_image, end_frame] + section_image_inputs + [total_second_length] + section_row_groups + [keyframe_copy_checkbox]
            )

            # EndFrame影響度調整スライダー
            with gr.Group():
                gr.Markdown(f"### " + translate("EndFrame影響度調整"))
                end_frame_strength = gr.Slider(
                    label=translate("EndFrame影響度"),
                    minimum=0.01,
                    maximum=1.00,
                    value=1.00,
                    step=0.01,
                    info=translate("最終フレームが動画全体に与える影響の強さを調整します。値を小さくすると最終フレームの影響が弱まり、最初のフレームに早く移行します。1.00が通常の動作です。")
                )

            # 出力フォルダ設定
            gr.Markdown(translate("※ 出力先は `webui` 配下に限定されます"))
            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    # フォルダ名だけを入力欄に設定
                    output_dir = gr.Textbox(
                        label=translate("出力フォルダ名"),
                        value=output_folder_name,  # 設定から読み込んだ値を使用
                        info=translate("動画やキーフレーム画像の保存先フォルダ名"),
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

            # フォルダを開くボタンのイベント
            def handle_open_folder_btn(folder_name):
                """フォルダ名を保存し、そのフォルダを開く"""
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

            open_folder_btn.click(fn=handle_open_folder_btn, inputs=[output_dir], outputs=[output_dir, path_display])

            # プロンプト管理パネル（右カラムから左カラムに移動済み）

    # 実行前のバリデーション関数
    def validate_and_process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_random_seed, mp4_crf=16, all_padding_value=1.0, end_frame=None, end_frame_strength=1.0, frame_size_setting="1秒 (33フレーム)", keep_section_videos=False, lora_files=None, lora_files2=None, lora_files3=None, lora_scales_text="0.8,0.8,0.8", output_dir=None, save_section_frames=False, section_settings=None, use_all_padding=False, use_lora=False, lora_mode=None, lora_dropdown1=None, lora_dropdown2=None, lora_dropdown3=None, save_tensor_data=False, tensor_data_input=None, fp8_optimization=False, resolution=640, batch_count=1, save_latent_frames=False, save_last_section_frames=False, use_vae_cache=False, use_queue=False, prompt_queue_file=None):
        # グローバル変数の宣言 - 関数の先頭で行う
        global batch_stopped, queue_enabled, queue_type, prompt_queue_file_path, vae_cache_enabled, image_queue_files
        # すべての入力パラメーターの型情報をデバッグ出力（問題診断用）
        print("=== 入力パラメーター型情報 ===")
        print(f"use_queue: 型={type(use_queue).__name__}")
        print(f"prompt_queue_file: 型={type(prompt_queue_file).__name__}")
        if hasattr(use_queue, 'value'):
            print(f"use_queue.value: {use_queue.value}, 型={type(use_queue.value).__name__}")
        if hasattr(prompt_queue_file, 'name'):
            print(f"prompt_queue_file.name: {prompt_queue_file.name}, 型={type(prompt_queue_file.name).__name__}")
        print("=========================")
        """入力画像または最後のキーフレーム画像のいずれかが有効かどうかを確認し、問題がなければ処理を実行する"""
        # 引数のデバッグ出力
        print(f"validate_and_process 引数数: {len(locals())}")
        
        # LoRA関連の引数を確認
        print(f"[DEBUG] LoRA関連の引数確認")
        print(f"[DEBUG] lora_files: {lora_files}, 型: {type(lora_files)}")
        print(f"[DEBUG] lora_files2: {lora_files2}, 型: {type(lora_files2)}")
        print(f"[DEBUG] lora_files3: {lora_files3}, 型: {type(lora_files3)}")
        print(f"[DEBUG] lora_scales_text: {lora_scales_text}, 型: {type(lora_scales_text)}")
        print(f"[DEBUG] use_lora: {use_lora}, 型: {type(use_lora)}")
        print(f"[DEBUG] lora_mode: {lora_mode}, 型: {type(lora_mode)}")
        print(f"[DEBUG] lora_dropdown1: {lora_dropdown1}, 型: {type(lora_dropdown1)}")
        print(f"[DEBUG] lora_dropdown2: {lora_dropdown2}, 型: {type(lora_dropdown2)}")
        print(f"[DEBUG] lora_dropdown3: {lora_dropdown3}, 型: {type(lora_dropdown3)}")
            
        # バッチカウント引数の確認
        print(f"[DEBUG] batch_count: {batch_count}, 型: {type(batch_count)}")
        
        # フレーム保存モードとVAEキャッシュの引数位置を確認
        print(f"[DEBUG] フレーム保存モード設定値: {save_latent_frames}, 型: {type(save_latent_frames)}")
        print(f"[DEBUG] VAEキャッシュ設定値: {use_vae_cache}, 型: {type(use_vae_cache)}")
        
        # グローバル変数宣言
        global vae_cache_enabled

        input_img = input_image  # 入力の最初が入力画像
        
        # デバッグ: 引数の型と値を表示
        print(translate("[DEBUG] resolution_value 型: {0}, 値: {1}").format(type(resolution).__name__, resolution))
        print(translate("[DEBUG] batch_count 型: {0}, 値: {1}").format(type(batch_count).__name__, batch_count))
        
        # フレーム保存モードの値を取得
        frame_save_mode = save_latent_frames
        
        # VAEキャッシュを取得（グローバル変数を優先）
        use_vae_cache_ui_value = use_vae_cache
        
        # UIの値よりもグローバル変数を優先
        use_vae_cache_value = vae_cache_enabled
        
        print(f"VAEキャッシュ設定値(UI): {use_vae_cache_ui_value}, 型: {type(use_vae_cache_ui_value)}")
        print(f"VAEキャッシュ設定値(グローバル変数): {vae_cache_enabled}, 型: {type(vae_cache_enabled)}")
        print(f"最終的なVAEキャッシュ設定値: {use_vae_cache_value}, 型: {type(use_vae_cache_value)}")
        
        print(f"フレーム保存モード: {frame_save_mode}(index=32)")
        
        # デバッグ：フレーム保存モードの型と値を確認
        print(translate("[DEBUG] frame_save_modeの型: {0}, 値: {1}").format(type(frame_save_mode).__name__, frame_save_mode))
        
        # Gradioのラジオボタンオブジェクトが直接渡されているか、文字列値が渡されているかを確認
        if hasattr(frame_save_mode, 'value'):
            # Gradioオブジェクトの場合は値を取得
            frame_save_mode_value = frame_save_mode.value
            print(translate("[DEBUG] Gradioオブジェクトから値を取得: {0}").format(frame_save_mode_value))
        else:
            # 文字列などの通常の値の場合はそのまま使用
            frame_save_mode_value = frame_save_mode
            print(translate("[DEBUG] 通常の値として使用: {0}").format(frame_save_mode_value))
        
        # モード選択に基づいてフラグを設定（必ずブール値を設定する）
        # 選択肢は3つ: "保存しない", "全フレーム画像保存", "最終セクションのみ全フレーム画像保存"
        
        # UIのデバッグ出力用に変数を作成（これらは実際には使用しない）
        save_latent_frames_display = False
        save_last_section_frames_display = False
        
        # 実際に使用するフラグ（必ずブール値を設定）
        save_latent_frames = False  # 最初にFalseに設定
        save_last_section_frames = False  # 最初にFalseに設定
        
        # 選択された値に基づいてフラグを設定
        if frame_save_mode_value == translate("全フレーム画像保存"):
            save_latent_frames = True
            save_latent_frames_display = translate("全フレーム画像保存")
        elif frame_save_mode_value == translate("最終セクションのみ全フレーム画像保存"):
            save_last_section_frames = True
            save_last_section_frames_display = translate("最終セクションのみ全フレーム画像保存")
        
        # UIデバッグ用の出力
        print(translate("[DEBUG] フレーム保存モード (オリジナル): {0}").format(frame_save_mode))
        print(translate("[DEBUG] フレーム保存モード (実際の値): {0}").format(frame_save_mode_value))
        print(translate("[DEBUG] save_latent_frames: {0}").format(save_latent_frames_display))
        print(translate("[DEBUG] save_last_section_frames: {0}").format(save_last_section_frames_display))
        
        # 実際のプログラム内部で使用する値の出力
        print(translate("[DEBUG] 設定済みフラグ (内部値) - save_latent_frames型: {0}, 値: {1}, save_last_section_frames型: {2}, 値: {3}").format(
            type(save_latent_frames).__name__, save_latent_frames, 
            type(save_last_section_frames).__name__, save_last_section_frames
        ))

        # バッチ回数を有効な範囲に制限
        # 型チェックしてから変換（数値でない場合はデフォルト値の1を使用）
        try:
            batch_count_val = int(batch_count)
            batch_count = max(1, min(batch_count_val, 100))  # 1〜100の間に制限
        except (ValueError, TypeError):
            print(translate("[WARN] validate_and_process: バッチ処理回数が無効です。デフォルト値の1を使用します: {0}").format(batch_count))
            batch_count = 1  # デフォルト値

        # ドロップダウン選択に基づいてuse_loraフラグを調整は既に引数で受け取り済み
        # 詳細なデバッグ出力を追加
        print(translate("[DEBUG] validate_and_process詳細："))
        print(f"[DEBUG] lora_dropdown1 詳細: 値={repr(lora_dropdown1)}, 型={type(lora_dropdown1).__name__}")
        print(f"[DEBUG] lora_dropdown2 詳細: 値={repr(lora_dropdown2)}, 型={type(lora_dropdown2).__name__}")
        print(f"[DEBUG] lora_dropdown3 詳細: 値={repr(lora_dropdown3)}, 型={type(lora_dropdown3).__name__}")
        
        # ディレクトリ選択モードの場合の処理
        if lora_mode == translate("ディレクトリから選択") and has_lora_support:
            # ドロップダウン選択があるか確認
            has_dropdown_selection = False
            dropdown_values = [(1, lora_dropdown1), (2, lora_dropdown2), (3, lora_dropdown3)]
            
            for idx, dropdown in dropdown_values:
                # 詳細なデバッグ情報
                print(translate("[DEBUG] ドロップダウン{0}の検出処理: 値={1!r}, 型={2}").format(
                    idx, dropdown, type(dropdown).__name__
                ))
                
                # 処理用にローカル変数にコピー（元の値を保持するため）
                processed_value = dropdown
                
                # 通常の値が0や0.0などの数値の場合の特別処理（GradioのUIの問題によるもの）
                if processed_value == 0 or processed_value == "0" or processed_value == 0.0:
                    # 数値の0を"なし"として扱う
                    print(translate("[DEBUG] validate_and_process: ドロップダウン{0}の値が数値0として検出されました。'なし'として扱います").format(idx))
                    processed_value = translate("なし")
                
                # 文字列でない場合は変換
                if processed_value is not None and not isinstance(processed_value, str):
                    processed_value = str(processed_value)
                    print(translate("[DEBUG] validate_and_process: ドロップダウン{0}の値を文字列に変換: {1!r}").format(idx, processed_value))
                
                # 有効な選択かチェック
                if processed_value and processed_value != translate("なし"):
                    has_dropdown_selection = True
                    print(translate("[DEBUG] validate_and_process: ドロップダウン{0}で有効な選択を検出: {1!r}").format(idx, processed_value))
                    break
            
            # 選択があれば有効化
            if has_dropdown_selection:
                use_lora = True
                print(translate("[INFO] validate_and_process: ドロップダウンでLoRAが選択されているため、LoRA使用を自動的に有効化しました"))
        
        # フラグの最終状態を出力
        print(translate("[DEBUG] LoRA使用フラグの最終状態: {0}").format(use_lora))
                
        # section_settingsがブール値の場合は空のリストで初期化
        if isinstance(section_settings, bool):
            print(f"[DEBUG] section_settings is bool ({section_settings}), initializing as empty list")
            section_settings = [[None, None, ""] for _ in range(50)]

        # 現在の動画長設定とフレームサイズ設定を渡す
        is_valid, error_message = validate_images(input_img, section_settings, length_radio, frame_size_radio)

        if not is_valid:
            # 画像が無い場合はエラーメッセージを表示して終了
            yield None, gr.update(visible=False), translate("エラー: 画像が選択されていません"), error_message, gr.update(interactive=True), gr.update(interactive=False), gr.update()
            return

        # 画像がある場合は通常の処理を実行
        # LoRA関連の引数をログ出力
        print(translate("[DEBUG] LoRA関連の引数を確認:"))
        print(translate("  - lora_files: {0}").format(lora_files))
        print(translate("  - lora_files2: {0}").format(lora_files2))
        print(translate("  - lora_files3: {0}").format(lora_files3))
        print(translate("  - use_lora: {0}").format(use_lora))
        print(translate("  - lora_mode: {0}").format(lora_mode))

        # resolutionが整数であることを確認
        try:
            resolution_int = int(float(resolution))
            resolution_value = resolution_int
        except (ValueError, TypeError):
            resolution_value = 640
            
        # グローバル変数vae_cache_enabledは既に宣言済み
        
        # フレーム保存モードのブール値変換
        save_latent_frames = False
        save_last_section_frames = False
        if isinstance(frame_save_mode, str):
            if frame_save_mode == translate("全フレーム画像保存"):
                save_latent_frames = True
            elif frame_save_mode == translate("最終セクションのみ全フレーム画像保存"):
                save_last_section_frames = True
        
        # グローバル変数からVAEキャッシュ設定を取得
        use_vae_cache = vae_cache_enabled
        print(f"最終的なVAEキャッシュ設定フラグ: {use_vae_cache}")
        
        # 最終的なフラグを設定（名前付き引数で渡すため変数の整理のみ）
        print(translate("[DEBUG] 最終的なuse_loraフラグを{0}に設定しました").format(use_lora))
        # 最終的な設定値の確認
        print(f"[DEBUG] 最終的なフラグ設定 - save_latent_frames: {save_latent_frames}, save_last_section_frames: {save_last_section_frames}, use_vae_cache: {use_vae_cache}, use_lora: {use_lora}")

        # キュー機能の状態を確認
        # グローバル変数の状態を先に取得
        global queue_enabled, queue_type

        # UIから受け取ったパラメータに応じてグローバル変数を更新
        if isinstance(use_queue, bool) and use_queue:
            # パラメータで有効化された場合はグローバル変数を更新
            queue_enabled = True

        # キュー機能の状態をログに出力
        print(f"キュー変数の型: {type(use_queue).__name__}")
        print(f"キュー状態: {queue_enabled}, タイプ: {queue_type}")

        # イメージキューの場合は、事前に画像ファイルリストを更新
        if queue_enabled and queue_type == "image":
            # inputs フォルダから画像ファイルリストを更新
            get_image_queue_files()
            image_queue_count = len(image_queue_files)
            print(f"イメージキュー使用: inputs フォルダの画像 {image_queue_count} 個を使用します")

            # バッチ数を画像数+1（入力画像を含む）に合わせる
            if image_queue_count > 0:
                # 入力画像を使う1回 + 画像ファイル分のバッチ数
                total_needed_batches = 1 + image_queue_count

                # 設定されたバッチ数より必要数が多い場合は調整
                if total_needed_batches > batch_count:
                    print(f"画像キュー数+1に合わせてバッチ数を自動調整: {batch_count} → {total_needed_batches}")
                    batch_count = total_needed_batches

        # パラメータもチェックして整合性を確認（診断用）
        param_enabled = False
        if isinstance(use_queue, bool):
            param_enabled = use_queue
            print(f"【最新版】パラメータからの値: {param_enabled} (ブール型)")
        elif hasattr(use_queue, 'value'):
            param_enabled = bool(use_queue.value)
            print(f"【最新版】パラメータからの値: {param_enabled} (Gradioオブジェクト)")
        elif isinstance(use_queue, str) and use_queue.lower() in ('true', 'false', 't', 'f', 'yes', 'no', 'y', 'n', '1', '0'):
            param_enabled = use_queue.lower() in ('true', 't', 'yes', 'y', '1')
            print(f"【最新版】パラメータからの値: {param_enabled} (文字列)")
        else:
            print(f"【最新版】パラメータは使用できない形式: {type(use_queue).__name__}")

        # グローバル変数と渡されたパラメータに不一致がある場合は警告
        if param_enabled != queue_enabled and not isinstance(use_queue, str):
            print(f"【警告】グローバル変数とパラメータの値が一致しません: グローバル={queue_enabled}, パラメータ={param_enabled}")

        # グローバル変数の値を優先して使用
        print(f"【最新版】最終的なプロンプトキュー状態: {queue_enabled} (グローバル変数優先)")

        print(f"プロンプトキュー使用状態: {queue_enabled}")

        # チェックボックスがオンでファイルがあれば処理
        queue_prompts_count = 0

        # グローバル変数のファイルパスを使用し、バッチ処理用にコピー
        global prompt_queue_file_path
        # ファイルアップロード状態の詳細診断
        print(f"prompt_queue_file の型: {type(prompt_queue_file).__name__}")
        if prompt_queue_file_path:
            print(f"プロンプトキューファイルパス: {prompt_queue_file_path}")

        # パラメータから得られるファイル情報
        if hasattr(prompt_queue_file, 'name'):
            print(f"【ファイル診断】prompt_queue_file.name: {prompt_queue_file.name}")
            # グローバル変数がなければパラメータから更新
            if prompt_queue_file_path is None:
                prompt_queue_file_path = prompt_queue_file.name
                print(f"【ファイル診断】グローバル変数を更新: {prompt_queue_file_path}")

        # グローバル変数が有効でチェックボックスがオンなら処理
        if prompt_queue_file_path is not None and queue_enabled:
            # グローバル変数からファイルパスを取得しつつ、saved変数も設定
            queue_file_path = prompt_queue_file_path
            # グローバル変数にすでに保存済み
            print(f"【ファイル診断】グローバル変数からファイルパス取得: {queue_file_path}")

            print(f"プロンプトキューファイル: {queue_file_path}")

            # ファイルパスが有効かチェック
            if queue_file_path and os.path.exists(queue_file_path):
                print(f"プロンプトキューファイルの内容を読み込みます: {queue_file_path}")
                try:
                    with open(queue_file_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        queue_prompts_count = len(lines)
                        print(f"有効なプロンプト行数: {queue_prompts_count}")

                        if queue_prompts_count > 0:
                            # サンプルとして最初の数行を表示
                            sample_lines = lines[:min(3, queue_prompts_count)]
                            print(f"プロンプトサンプル: {sample_lines}")

                            # バッチ数をプロンプト数に合わせる
                            if queue_prompts_count > batch_count:
                                print(f"プロンプト数に合わせてバッチ数を自動調整: {batch_count} → {queue_prompts_count}")
                                batch_count = queue_prompts_count
                        else:
                            print("プロンプトキューファイルに有効なプロンプトがありません")
                except Exception as e:
                    print(f"プロンプトキューファイル読み込みエラー: {str(e)}")
            else:
                print(f"プロンプトキューファイルが存在しないか無効です: {queue_file_path}")
        else:
            # エラーメッセージの改善
            if not queue_enabled:
                print("プロンプトキュー無効: チェックボックスがオフです")
            if prompt_queue_file_path is None:
                print("プロンプトキュー無効: ファイルが正しくアップロードされていません")
                print("【ヒント】もう一度ファイルをアップロードしてみてください")

        # process関数のジェネレータを返す - 明示的に全ての引数を渡す
        yield from process(
            input_image=input_image,
            prompt=prompt,
            n_prompt=n_prompt,
            seed=seed,
            total_second_length=total_second_length,
            latent_window_size=latent_window_size,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            gpu_memory_preservation=gpu_memory_preservation,
            use_teacache=use_teacache,
            use_random_seed=use_random_seed,
            mp4_crf=mp4_crf,
            all_padding_value=all_padding_value,
            end_frame=end_frame,
            end_frame_strength=end_frame_strength,
            frame_size_setting=frame_size_setting,
            keep_section_videos=keep_section_videos,
            lora_files=lora_files,
            lora_files2=lora_files2,
            lora_files3=lora_files3,
            lora_scales_text=lora_scales_text,
            output_dir=output_dir,
            save_section_frames=save_section_frames,
            section_settings=section_settings,
            use_all_padding=use_all_padding,
            use_lora=use_lora,
            lora_mode=lora_mode,
            lora_dropdown1=lora_dropdown1,
            lora_dropdown2=lora_dropdown2,
            lora_dropdown3=lora_dropdown3,
            save_tensor_data=save_tensor_data,
            tensor_data_input=tensor_data_input,
            fp8_optimization=fp8_optimization,
            resolution=resolution_value,
            batch_count=batch_count,
            save_latent_frames=save_latent_frames,
            save_last_section_frames=save_last_section_frames,
            use_vae_cache=use_vae_cache,
            use_queue=bool(queue_enabled),  # 確実にブール値として渡す
            prompt_queue_file=prompt_queue_file
        )

    # 実行ボタンのイベント
    # UIから渡されるパラメーターリスト
    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_random_seed, mp4_crf, all_padding_value, end_frame, end_frame_strength, frame_size_radio, keep_section_videos, lora_files, lora_files2, lora_files3, lora_scales_text, output_dir, save_section_frames, section_settings, use_all_padding, use_lora, lora_mode, lora_dropdown1, lora_dropdown2, lora_dropdown3, save_tensor_data, tensor_data_input, fp8_optimization, resolution, batch_count, frame_save_mode, use_vae_cache, use_queue, prompt_queue_file]
    
    # デバッグ: チェックボックスの現在値を出力
    print(f"use_vae_cacheチェックボックス値: {use_vae_cache.value if hasattr(use_vae_cache, 'value') else 'no value attribute'}, id={id(use_vae_cache)}")
    start_button.click(fn=validate_and_process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, seed])
    end_button.click(fn=end_process, outputs=[end_button])

    # キーフレーム画像変更時のイベント登録
    # セクション0（赤枚)からの自動コピー処理
    for target_idx in range(1, max_keyframes):
        # 偶数セクションにのみコピー
        if target_idx % 2 == 0:  # 偶数先セクション
            single_handler = create_single_keyframe_handler(0, target_idx)
            section_image_inputs[0].change(
                fn=single_handler,
                inputs=[section_image_inputs[0], mode_radio, length_radio, enable_keyframe_copy],
                outputs=[section_image_inputs[target_idx]]
            )

    # セクション1（青枠)からの自動コピー処理
    for target_idx in range(2, max_keyframes):
        # 奇数セクションにのみコピー
        if target_idx % 2 == 1:  # 奇数先セクション
            single_handler = create_single_keyframe_handler(1, target_idx)
            section_image_inputs[1].change(
                fn=single_handler,
                inputs=[section_image_inputs[1], mode_radio, length_radio, enable_keyframe_copy],
                outputs=[section_image_inputs[target_idx]]
            )

    # 注: create_single_keyframe_handler関数はフレームサイズや動画長に基づいた動的セクション数を計算します
    # UIでフレームサイズや動画長を変更すると、動的に計算されたセクション数に従ってコピー処理が行われます

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

    # 保存ボタンのクリックイベントを接続
    save_btn.click(
        fn=save_button_click_handler,
        inputs=[edit_name, edit_prompt],
        outputs=[result_message, preset_dropdown, prompt]
    )

    # クリアボタン処理
    def clear_fields():
        return gr.update(value=""), gr.update(value="")

    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[edit_name, edit_prompt]
    )

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

    preset_dropdown.change(
        fn=load_preset_handler_wrapper,
        inputs=[preset_dropdown],
        outputs=[edit_name, edit_prompt]
    )

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

    apply_preset_btn.click(
        fn=apply_to_prompt,
        inputs=[edit_prompt],
        outputs=[prompt]
    )

    delete_preset_btn.click(
        fn=delete_preset_handler,
        inputs=[preset_dropdown],
        outputs=[result_message, preset_dropdown]
    )

# enable_keyframe_copyの初期化（グローバル変数）
enable_keyframe_copy = True

allowed_paths = [os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './outputs')))]

# 起動コード
try:
    block.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        allowed_paths=allowed_paths,
        inbrowser=args.inbrowser,
    )
except OSError as e:
    if "Cannot find empty port" in str(e):
        print("\n======================================================")
        print(translate("エラー: FramePack-eichiは既に起動しています。"))
        print(translate("同時に複数のインスタンスを実行することはできません。"))
        print(translate("現在実行中のアプリケーションを先に終了してください。"))
        print("======================================================\n")
        input(translate("続行するには何かキーを押してください..."))
    else:
        # その他のOSErrorの場合は元のエラーを表示
        print(translate("\nエラーが発生しました: {e}").format(e=e))
        input(translate("続行するには何かキーを押してください..."))

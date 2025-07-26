import os
import sys
sys.path.append(os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './submodules/FramePack'))))

# Windows環境で loop再生時に [WinError 10054] の warning が出るのを回避する設定
import asyncio
if sys.platform in ('win32', 'cygwin'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from diffusers_helper.hf_login import login

import os
import random
import time
import subprocess
import traceback  # ログ出力用
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
except ImportError:
    print(translate("LoRAサポートが無効です（lora_utilsモジュールがインストールされていません）"))

# 設定管理のインポートと読み込み
from eichi_utils.settings_manager import load_app_settings_f1
saved_app_settings = load_app_settings_f1()

# 読み込んだ設定をログに出力
if saved_app_settings:
    pass
else:
    print(translate(" 保存された設定が見つかりません。デフォルト設定を使用します"))

# 設定モジュールをインポート（ローカルモジュール）
import os.path
from eichi_utils.video_mode_settings import (
    VIDEO_MODE_SETTINGS, get_video_modes, get_video_seconds, get_important_keyframes,
    get_copy_targets, get_max_keyframes_count, get_total_sections, generate_keyframe_guide_html,
    handle_mode_length_change, process_keyframe_change, MODE_TYPE_NORMAL
    # F1モードでは不要な機能のインポートを削除
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

# ログ管理モジュールをインポート
from eichi_utils.log_manager import (
    enable_logging, disable_logging, is_logging_enabled, 
    get_log_folder, set_log_folder, open_log_folder,
    get_default_log_settings, load_log_settings, apply_log_settings
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
    unified_input_image_change_handler
)

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
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
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

from eichi_utils.config_queue_manager import ConfigQueueManager

from pathlib import Path

current_ui_components = {}



free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 100

print(translate('Free VRAM {0} GB').format(free_mem_gb))
print(translate('High-VRAM Mode: {0}').format(high_vram))

# モデルを並列ダウンロードしておく
from eichi_utils.model_downloader import ModelDownloader
ModelDownloader().download_f1()

# グローバルなモデル状態管理インスタンスを作成
# F1モードではuse_f1_model=Trueを指定
transformer_manager = TransformerManager(device=gpu, high_vram_mode=high_vram, use_f1_model=True)
text_encoder_manager = TextEncoderManager(device=gpu, high_vram_mode=high_vram)

# ==============================================================================
# CONFIG QUEUE SYSTEM - MAIN INTEGRATION
# ==============================================================================


# Queue processing state tracking
config_queue_manager = None  # Initialized later in main code
current_loaded_config = None  # Currently loaded config name
queue_processing_active = False  # Global processing state flag
current_processing_config_name = None  # For video file naming
current_batch_progress = {"current": 0, "total": 0}  # Batch progress tracking
queue_ui_settings = None  # Captured UI settings for queue processing
pending_lora_config_data = None  # For delayed LoRA configuration loading

# Configuration constants for queue display
CONST_queued_shown_count = 5  # Number of queued items shown in status
CONST_latest_finish_count = 2  # Number of completed items shown in status

# Language-independent constants for LoRA config storage
LORA_MODE_DIRECTORY = "directory_selection"
LORA_MODE_UPLOAD = "file_upload"
LORA_NONE_OPTION = "none_option"

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

if not high_vram:
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

# LoRAプリセットの初期化
from eichi_utils.lora_preset_manager import initialize_lora_presets
initialize_lora_presets()

# ベースパスを定義
base_path = os.path.dirname(os.path.abspath(__file__))

# 設定から出力フォルダを取得
app_settings = load_settings()
output_folder_name = app_settings.get('output_folder', 'outputs')
print(translate("設定から出力フォルダを読み込み: {0}").format(output_folder_name))

# ログ設定を読み込み適用
log_settings = app_settings.get('log_settings', get_default_log_settings())
print(translate("ログ設定を読み込み: 有効={0}, フォルダ={1}").format(
    log_settings.get('log_enabled', False), 
    log_settings.get('log_folder', 'logs')
))
if log_settings.get('log_enabled', False):
    # 現在のファイル名を渡す
    enable_logging(log_settings.get('log_folder', 'logs'), source_name="endframe_ichi_f1")
    print(translate("ログ出力を有効化しました"))

# キュー関連のグローバル変数
queue_enabled = False  # キュー機能の有効/無効フラグ
queue_type = "prompt"  # キューのタイプ（"prompt" または "image"）
prompt_queue_file_path = None  # プロンプトキューのファイルパス
image_queue_files = []  # イメージキューのファイルリスト
input_folder_name_value = app_settings.get('input_folder', 'inputs')  # 入力フォルダ名の設定値

# 入力フォルダも存在確認（作成はボタン押下時のみ）
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_folder_name_value)
print(translate("設定から入力フォルダを読み込み: {0}").format(input_folder_name_value))

# 出力フォルダのフルパスを生成
outputs_folder = get_output_folder_path(output_folder_name)
os.makedirs(outputs_folder, exist_ok=True)

# ==============================================================================
# CORE QUEUE CONFIGURATION FUNCTIONS
# ==============================================================================

def get_current_ui_settings_for_queue():

    global current_ui_components
    
    try:
        settings = {}
        
        # Helper function to safely get component values with explicit type conversion
        def get_component_value(component_name, default_value, value_type=None):
            if component_name in current_ui_components:
                component = current_ui_components[component_name]
                if hasattr(component, 'value'):
                    value = component.value
                    print(translate("🔍 Getting {0}: {1} (type: {2})").format(component_name, value, type(value)))
                    # Type conversion if specified
                    if value_type == bool:
                        return bool(value)
                    elif value_type == int:
                        try:
                            return int(float(value)) if value is not None else default_value
                        except (ValueError, TypeError):
                            print(translate("⚠️ Error converting {0} to int: {1}, using default: {2}").format(component_name, value, default_value))
                            return default_value
                    elif value_type == float:
                        try:
                            return float(value) if value is not None else default_value
                        except (ValueError, TypeError):
                            print(translate("⚠️ Error converting {0} to float: {1}, using default: {2}").format(component_name, value, default_value))
                            return default_value
                    else:
                        return value if value is not None else default_value
                else:
                    print(translate("⚠️ Component {0} has no value attribute").format(component_name))
                    return default_value
            else:
                print(translate("⚠️ Component {0} not found in registered components").format(component_name))
                return default_value
        
        # ===== DURATION SETTINGS - DETAILED LOGGING =====
        
        # Get slider value with detailed logging
        total_second_length_value = get_component_value('total_second_length', 1, int)
        settings['total_second_length'] = max(1, total_second_length_value)
        
        # Get radio value for comparison
        length_radio_value = get_component_value('length_radio', translate("1秒"))
        
        print(translate("🕒 Duration settings for queue:"))
        print(translate("   length_radio: '{0}' (for reference only)").format(length_radio_value))
        print(translate("   total_second_length slider: {0}s → final: {1}s").format(total_second_length_value, settings['total_second_length']))
        
        # Determine duration source for logging
        duration_source = "total_second_length"  # Since we're prioritizing the slider
        
        # Frame size settings
        frame_size_setting = get_component_value('frame_size_radio', translate("1秒 (33フレーム)"))
        settings['frame_size_setting'] = frame_size_setting
        
        # Convert frame size to latent_window_size
        if frame_size_setting == translate("0.5秒 (17フレーム)"):
            settings['latent_window_size'] = 4.5
        else:
            settings['latent_window_size'] = 9
        
        print(translate("🎬 Frame settings: {0} → latent_window_size={1}").format(frame_size_setting, settings['latent_window_size']))
        
        # ===== QUALITY SETTINGS =====
        settings['steps'] = get_component_value('steps', 25, int)
        settings['cfg'] = get_component_value('cfg', 1.0, float)
        settings['gs'] = get_component_value('gs', 10, float)
        settings['rs'] = get_component_value('rs', 0.0, float)
        settings['resolution'] = get_component_value('resolution', 640, int)
        settings['mp4_crf'] = get_component_value('mp4_crf', 16, int)
        
        # ===== GENERATION SETTINGS =====
        base_seed = get_component_value('seed', 1, int)
        use_random_seed = get_component_value('use_random_seed', False, bool)
        
        if use_random_seed:
            import random
            settings['seed'] = random.randint(0, 2**32 - 1)
            print(translate("🎲 Generated new random seed for queue item: {0}").format(settings['seed']))
        else:
            settings['seed'] = base_seed
            
        settings['use_random_seed'] = False  # Always False for queue processing since we handle it above
        settings['use_teacache'] = get_component_value('use_teacache', True, bool)
        settings['image_strength'] = get_component_value('image_strength', 1.0, float)
        settings['fp8_optimization'] = get_component_value('fp8_optimization', True, bool)
        
        # ===== BATCH COUNT - EXPLICIT HANDLING =====
        batch_count_raw = get_component_value('batch_count', 1, int)
        # Ensure it's definitely an integer and within valid range
        batch_count_final = max(1, min(int(batch_count_raw), 100))
        settings['batch_count'] = batch_count_final
        
        print(translate("🔢 Batch count processing: raw={0} (type: {1}) → final={2} (type: {3})").format(batch_count_raw, type(batch_count_raw), batch_count_final, type(batch_count_final)))
        
        
        # ===== SYSTEM SETTINGS =====
        settings['gpu_memory_preservation'] = get_component_value('gpu_memory_preservation', 6.0, float)
        
        # ===== OUTPUT SETTINGS =====
        settings['keep_section_videos'] = get_component_value('keep_section_videos', False, bool)
        settings['save_section_frames'] = get_component_value('save_section_frames', False, bool)
        settings['save_tensor_data'] = get_component_value('save_tensor_data', False, bool)
        settings['frame_save_mode'] = get_component_value('frame_save_mode', translate("保存しない"))
        settings['output_dir'] = get_component_value('output_dir', "outputs")
        settings['alarm_on_completion'] = get_component_value('alarm_on_completion', False, bool)
        
        # ===== F1 MODE SETTINGS =====
        settings['all_padding_value'] = get_component_value('all_padding_value', 1.0, float)
        settings['use_all_padding'] = get_component_value('use_all_padding', False, bool)
        
        # ===== FIXED VALUES FOR QUEUE PROCESSING =====
        settings['n_prompt'] = ""  # Ignored in F1 mode
        settings['tensor_data_input'] = None  # Not supported in queue
        settings['use_queue'] = False
        settings['prompt_queue_file'] = None
        settings['batch_count'] = 1
        settings['save_settings_on_start'] = False
        
        print(translate("📋 Queue settings summary:"))
        print(translate("   Duration: {0}s ({1}), Frames: {2}").format(settings['total_second_length'], duration_source, frame_size_setting))
        print(translate("   Quality: steps={0}, CFG={1}, Distilled={2}").format(settings['steps'], settings['cfg'], settings['gs']))
        print(translate("   Output: resolution={0}, CRF={1}").format(settings['resolution'], settings['mp4_crf']))
        print(translate("   Generation: seed={0} (random: {1})").format(settings['seed'], use_random_seed))
        print(translate("   Performance: TeaCache={0}, FP8={1}").format(settings['use_teacache'], settings['fp8_optimization']))
        
        return settings
        
    except Exception as e:
        print(translate("❌ Error getting current UI settings: {0}").format(e))
        import traceback
        traceback.print_exc()
        
        # Return safe defaults
        return {
            'total_second_length': 1,
            'latent_window_size': 9,
            'frame_size_setting': translate("1秒 (33フレーム)"),
            'steps': 25,
            'cfg': 1.0,
            'gs': 10,
            'rs': 0.0,
            'resolution': 640,
            'mp4_crf': 16,
            'seed': 1,
            'use_random_seed': False,
            'use_teacache': True,
            'image_strength': 1.0,
            'fp8_optimization': True,
            'gpu_memory_preservation': 6.0,
            'keep_section_videos': False,
            'save_section_frames': False,
            'save_tensor_data': False,
            'frame_save_mode': translate("保存しない"),
            'output_dir': "outputs",
            'alarm_on_completion': False,
            'all_padding_value': 1.0,
            'use_all_padding': False,
            'n_prompt': "",
            'tensor_data_input': None,
            'use_queue': False,
            'prompt_queue_file': None,
            'batch_count': 1,
            'save_settings_on_start': False
        }

def cancel_operation_handler():
    return (
        "❌ " + translate("Operation cancelled"),
        gr.update(),  # Don't change config dropdown
        gr.update(),  # Don't change queue status
        gr.update(visible=False),  # Hide confirmation group
        None  # Clear operation data
    )

def merged_refresh_handler_standardized():
    try:
        if config_queue_manager is None:
            return translate("❌ Config queue manager not initialized"), gr.update(), gr.update()
        
        # Refresh config list
        available_configs = config_queue_manager.get_available_configs()
        
        # Get enhanced queue status with auto-correction
        queue_status = config_queue_manager.get_queue_status()
        
        # Auto-correction logic (same as before)
        global queue_processing_active
        manager_processing = config_queue_manager.is_processing
        has_current_work = bool(queue_status.get('current_config'))
        has_queued_items = queue_status.get('queue_count', 0) > 0
        
        needs_correction = False
        
        if manager_processing and not has_current_work and not has_queued_items:
            print(translate("🔧 Merged refresh: Manager processing but no work - correcting"))
            config_queue_manager.is_processing = False
            config_queue_manager.current_config = None
            queue_processing_active = False
            needs_correction = True
        
        if queue_processing_active and not manager_processing:
            print(translate("🔧 Merged refresh: Global active but manager idle - syncing"))
            queue_processing_active = False
            needs_correction = True
        
        if needs_correction:
            queue_status = config_queue_manager.get_queue_status()
        
        # Use the same enhanced formatting function for consistency
        status_text = format_queue_status_with_batch_progress(queue_status)
        
        # Add correction note if needed
        if needs_correction:
            status_text += "\n🔧 Auto-corrected processing state"
        
        queue_count = queue_status['queue_count']
        print(translate("🔄 Merged refresh completed: {0} configs, {1} queued").format(len(available_configs), queue_count))
        
        return (
            translate("✅ Refreshed: {0} configs, {1} queued").format(len(available_configs), queue_count),
            gr.update(choices=available_configs),
            gr.update(value=status_text)
        )
        
    except Exception as e:
        return translate("❌ Error during refresh: {0}").format(str(e)), gr.update(), gr.update()

# ==============================================================================
# QUEUE CONTROL HANDLERS
# ==============================================================================

def queue_config_handler_with_confirmation(config_dropdown):

    global current_loaded_config
    
    config_name = current_loaded_config or config_dropdown
    if not config_name:
        return "❌ Error: No config loaded", gr.update(), gr.update(visible=False), None
    
    if config_queue_manager is None:
        return translate("❌ Error: Config queue manager not initialized"), gr.update(), gr.update(visible=False), None

    # Check if already in queue
    queue_file = os.path.join(config_queue_manager.queue_dir, f"{config_name}.json")
    
    if os.path.exists(queue_file):
        # Store operation details for confirmation
        operation_data = {
            'type': 'queue_overwrite',
            'config_name': config_name
        }
        
        # Show confirmation message
        confirmation_msg = f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #fff3cd; border: 1px solid #ffeaa7; margin: 10px 0;">
            <h3 style="color: #856404; margin: 0 0 10px 0;">⚠️ {translate('Queue Overwrite Confirmation')}</h3>
            <p style="margin: 10px 0;">{translate('Config "{0}" is already in the queue. Do you want to overwrite it with the current config settings?').format(config_name)}</p>
            <p style="margin: 10px 0; font-weight: bold; color: #856404;">
                {translate('This will replace the queued config with your current settings.')}
            </p>
        </div>
        """
        
        return (
            confirmation_msg,
            gr.update(),
            gr.update(visible=True),  # Show confirmation group
            operation_data  # Store operation data
        )
    else:
        # Not in queue, proceed normally
        success, message = config_queue_manager.queue_config(config_name)
        
        if success:
            queue_status = config_queue_manager.get_queue_status()
            status_text = format_queue_status_with_batch_progress(queue_status)
            return f"✅ {message}", gr.update(value=status_text), gr.update(visible=False), None
        else:
            return f"❌ {message}", gr.update(), gr.update(visible=False), None
  
def stop_queue_processing_handler_fixed():

    global queue_processing_active

    if config_queue_manager is None:
        return translate("❌ Error: Config queue manager not initialized"), gr.update(), gr.update(visible=True), gr.update(visible=True)
    
    if not queue_processing_active and not config_queue_manager.is_processing:
        return translate("❌ Queue processing is not running"), gr.update(), gr.update(visible=True), gr.update(visible=True)
    
    print(translate("🛑 Stopping queue processing..."))
    
    success, message = config_queue_manager.stop_queue_processing()
    
    if success:
        # Force reset both flags
        queue_processing_active = False
        config_queue_manager.is_processing = False
        config_queue_manager.current_config = None
        print(translate("✅ Queue processing stopped and flags reset"))
        
    queue_status = config_queue_manager.get_queue_status()
    status_text = format_queue_status_with_batch_progress(queue_status)
    
    # Return with visibility restored
    return message, gr.update(value=status_text), gr.update(visible=True), gr.update(visible=True)

def clear_queue_handler():

    if config_queue_manager is None:  # ← Add this line
        return translate("Error: Config queue manager not initialized/clear_queue_handler"), gr.update(), gr.update()    
    
    success, message = config_queue_manager.clear_queue()
    
    queue_status = config_queue_manager.get_queue_status()
    status_text = format_queue_status_with_batch_progress(queue_status)
    
    return message, gr.update(value=status_text)

# ==============================================================================
# CONFIG FILE OPERATIONS (SAVE/LOAD/DELETE)
# ==============================================================================

def save_current_config_handler_v3(config_name_input, add_timestamp, input_image, prompt, use_lora, lora_mode, 
                                  lora_dropdown1, lora_dropdown2, lora_dropdown3, lora_files, 
                                  lora_files2, lora_files3, lora_scales_text):

    global current_loaded_config
    
    try:
        # Validate inputs
        if not input_image:
            return translate("❌ Error: No image selected"), gr.update(), gr.update(), gr.update(visible=False), None
            
        if not prompt or not prompt.strip():
            return translate("❌ Error: No prompt entered"), gr.update(), gr.update(), gr.update(visible=False), None
        
        if config_queue_manager is None:
            return translate("❌ Error: Config queue manager not initialized"), gr.update(), gr.update(), gr.update(visible=False), None
        
        # Get the config name to use
        config_name_to_use = config_name_input.strip() if config_name_input and config_name_input.strip() else ""
        
        # Check if config already exists (only relevant when NOT adding timestamp)
        will_overwrite = False
        if config_name_to_use and not add_timestamp:
            will_overwrite = config_queue_manager.config_exists(config_name_to_use)
        
        if will_overwrite:
            # Store operation details for confirmation
            operation_data = {
                'type': 'overwrite_exact',
                'config_name': config_name_to_use,
                'config_name_input': config_name_input,
                'add_timestamp': add_timestamp,
                'input_image': input_image,
                'prompt': prompt,
                'use_lora': use_lora,
                'lora_mode': lora_mode,
                'lora_dropdown1': lora_dropdown1,
                'lora_dropdown2': lora_dropdown2,
                'lora_dropdown3': lora_dropdown3,
                'lora_files': lora_files,
                'lora_files2': lora_files2,
                'lora_files3': lora_files3,
                'lora_scales_text': lora_scales_text
            }
            
            # Show confirmation message
            confirmation_msg = f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #fff3cd; border: 1px solid #ffeaa7; margin: 10px 0;">
                <h3 style="color: #856404; margin: 0 0 10px 0;">⚠️ {translate('Overwrite Confirmation')}</h3>
                <p style="margin: 10px 0;">{translate('Config file "{0}.json" already exists. Do you want to overwrite it?').format(config_name_to_use)}</p>
                <p style="margin: 10px 0; font-weight: bold; color: #856404;">
                    {translate('Use the buttons above to confirm or cancel the operation.')}
                </p>
            </div>
            """
            
            return (
                confirmation_msg, 
                gr.update(), 
                gr.update(), 
                gr.update(visible=True),  # Show confirmation group
                operation_data  # Store operation data
            )
        
        # No overwrite needed - proceed directly
        return perform_save_operation_v3(
            config_name_input, add_timestamp, input_image, prompt, use_lora, lora_mode,
            lora_dropdown1, lora_dropdown2, lora_dropdown3, lora_files,
            lora_files2, lora_files3, lora_scales_text
        )
        
    except Exception as e:
        return f"❌ Error saving config: {str(e)}", gr.update(), gr.update(), gr.update(visible=False), None
    
def perform_save_operation_v3(config_name_input, add_timestamp, input_image, prompt, use_lora, lora_mode,
                            lora_dropdown1, lora_dropdown2, lora_dropdown3, lora_files,
                            lora_files2, lora_files3, lora_scales_text):

    global current_loaded_config

    try:
        config_name_to_use = config_name_input.strip() if config_name_input and config_name_input.strip() else ""
        
        # Get LoRA settings (with language-independent storage and auto-conversion)
        lora_settings = get_current_lora_settings(
            use_lora, lora_mode, lora_dropdown1, lora_dropdown2, lora_dropdown3,
            lora_files, lora_files2, lora_files3, lora_scales_text
        )
        
        # Save config with timestamp option
        success, message = config_queue_manager.save_config_with_timestamp_option(
            config_name_to_use, input_image, prompt, lora_settings, add_timestamp, other_params=None
        )
        
        if success:
            # FIXED: Parse the message to extract ONLY the config name, not system messages
            # Expected message format: "Config saved: {actual_name}"
            actual_config_name = config_name_to_use
            
            # Parse the clean config name from the success message
            if ": " in message:
                # Extract everything after "Config saved: "
                temp_name = message.split(": ")[1].strip()
                # Remove any system messages that might have been concatenated
                if " (" in temp_name:
                    # Remove everything from the first parenthesis onwards
                    actual_config_name = temp_name.split(" (")[0].strip()
                else:
                    actual_config_name = temp_name
            
            # VALIDATION: Ensure the config name is clean and doesn't contain system messages
            if "(" in actual_config_name or ")" in actual_config_name:
                print(translate(f"⚠️ Warning: Config name contains parentheses, cleaning: '{actual_config_name}'"))
                # Remove everything from the first parenthesis onwards
                actual_config_name = actual_config_name.split("(")[0].strip()
            
            print(translate("✅ Config saved successfully: '{0}' (from message: '{1}')").format(actual_config_name, message))
            
            current_loaded_config = actual_config_name
            
            # CRITICAL FIX: Always refresh the available configs list after saving
            # This ensures the dropdown includes any newly created configs (including case variations)
            available_configs = config_queue_manager.get_available_configs()
            
            # CASE-SENSITIVE CHECK: Ensure the actual_config_name exists in the choices
            # This handles case-sensitive file systems where "abc" and "ABC" are different
            if actual_config_name not in available_configs:
                print(translate("⚠️ Warning: Saved config '{0}' not found in available configs").format(actual_config_name))
                print(translate("   Available configs: {0}...").format(available_configs[:10]))  # Show first 10 for debugging
                
                # Try case-insensitive search as fallback
                for config in available_configs:
                    if config.lower() == actual_config_name.lower():
                        print(f"   Found case-insensitive match: '{config}'")
                        actual_config_name = config
                        current_loaded_config = config
                        break
            
            queue_status = config_queue_manager.get_queue_status()
            status_text = format_queue_status_with_batch_progress(queue_status)
            
            # Format user message with timestamp info - keep it separate from config name
            user_message = ""
            if add_timestamp and config_name_to_use:
                user_message = f"✅ {translate('Config saved with timestamp')}: {actual_config_name}.json"
            elif not add_timestamp and config_name_to_use:
                user_message = f"✅ {translate('Config saved with exact name')}: {actual_config_name}.json"
            else:
                user_message = f"✅ {translate('Config saved with auto-generated name')}: {actual_config_name}.json"
            
            # Add LoRA conversion notification if applicable (but separate from config name)
            if lora_settings.get("lora_mode_key") == LORA_MODE_DIRECTORY and lora_settings.get("lora_files"):
                # If we have LoRA files, show what was configured
                lora_files_list = lora_settings.get("lora_files", [])
                if lora_files_list:
                    filenames = [os.path.basename(path) for path in lora_files_list]
                    user_message += translate("\n📦 LoRA files configured: {0}").format(', '.join(filenames))
            
            return (
                user_message,
                gr.update(choices=available_configs, value=actual_config_name),  # Use CLEAN config name with refreshed choices
                gr.update(value=status_text),
                gr.update(visible=False),  # Hide confirmation group
                None  # Clear operation data
            )
        else:
            return translate("❌ {0}").format(message), gr.update(), gr.update(), gr.update(visible=False), None
            
    except Exception as e:
        return translate("❌ Error saving config: {0}").format(str(e)), gr.update(), gr.update(), gr.update(visible=False), None

def load_config_with_delayed_lora_application_fixed(config_name):

    global pending_lora_config_data, current_loaded_config
    
    if not config_name or config_queue_manager is None:
        return [
            translate("Error: Config queue manager not initialized"),
            gr.update(), gr.update(), gr.update(), gr.update(), 
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        ]
    
    success, config_data, message = config_queue_manager.load_config_for_editing(config_name)
    
    if success:
        current_loaded_config = config_name
        
        image_path = config_data.get('image_path')
        prompt = config_data.get('prompt', '')
        lora_settings = config_data.get('lora_settings', {})
        
        use_lora = lora_settings.get('use_lora', False)
        
        # Convert language-independent key to current language
        lora_mode_key = lora_settings.get('lora_mode_key')
        if lora_mode_key:
            lora_mode = get_lora_mode_text(lora_mode_key)
        else:
            # Fallback: try to convert old language-dependent format
            old_lora_mode = lora_settings.get('lora_mode', translate("ディレクトリから選択"))
            lora_mode = translate("ディレクトリから選択")  # Safe default
            print(translate("📦 Config uses old language-dependent format, using default: {0}").format(lora_mode))
        
        print(translate("📂 Loading config: {0}").format(config_name))
        print(translate("    use_lora: {0}, lora_mode: {1}").format(use_lora, lora_mode))
        
        # Handle LoRA configuration (now always directory mode due to auto-conversion)
        if use_lora and lora_mode == translate("ディレクトリから選択"):
            # FIXED: Always scan directory first to ensure we have current choices
            choices = scan_lora_directory()
            print(translate("📦 Scanned LoRA directory, found {0} choices").format(len(choices)))
            
            # Try new format first (language-independent)
            lora_dropdown_files = lora_settings.get('lora_dropdown_files')
            if lora_dropdown_files:
                print(translate("📦 Using language-independent format"))
                lora_dropdown_values = []
                
                for dropdown_file in lora_dropdown_files[:3]:
                    if dropdown_file == LORA_NONE_OPTION:
                        # FIXED: Handle both constants and string literals
                        lora_dropdown_values.append(translate("なし"))
                    elif dropdown_file in choices:
                        lora_dropdown_values.append(dropdown_file)
                    else:
                        # FIXED: If file not found, use "none" instead of the missing file
                        print(translate("⚠️ LoRA file not found in directory: {0}, using 'none'").format(dropdown_file))
                        lora_dropdown_values.append(translate("なし"))
                
                # Pad with "none" if needed
                while len(lora_dropdown_values) < 3:
                    lora_dropdown_values.append(translate("なし"))
                
                applied_files = [f for f in lora_dropdown_values if f != translate("なし")]
                
            else:
                # Fallback: use old format (file paths)
                print(translate("📦 Using fallback file path method"))
                lora_files = lora_settings.get('lora_files', [])
                if lora_files:
                    # FIXED: Use enhanced validation in apply_lora_config_to_dropdowns
                    choices, lora_dropdown_values, applied_files = apply_lora_config_to_dropdowns_safe(lora_files, choices)
                else:
                    lora_dropdown_values = [translate("なし")] * 3
                    applied_files = []
            
            if applied_files:
                print(translate("✅ Applied LoRA files: {0}").format(applied_files))
                
                # Store for potential reuse
                pending_lora_config_data = {
                    'files': lora_settings.get('lora_files', []),
                    'scales': lora_settings.get('lora_scales', '0.8,0.8,0.8'),
                    'mode': lora_mode,
                    'config_name': config_name,
                    'applied_values': lora_dropdown_values
                }
                
                # FIXED: Return with proper choices and values, ensuring all values are in choices
                return [
                    translate("✅ Loaded config: {0} (LoRA: {1}, Str: {2})").format(config_name, ', '.join(applied_files), lora_settings.get('lora_scales')),
                    gr.update(value=image_path if image_path and os.path.exists(image_path) else None),
                    gr.update(value=prompt),
                    gr.update(value=use_lora),
                    gr.update(value=lora_mode),
                    gr.update(value=lora_settings.get('lora_scales', '0.8,0.8,0.8')),
                    gr.update(choices=choices, value=lora_dropdown_values[0] if lora_dropdown_values[0] in choices else choices[0]),
                    gr.update(choices=choices, value=lora_dropdown_values[1] if lora_dropdown_values[1] in choices else choices[0]),
                    gr.update(choices=choices, value=lora_dropdown_values[2] if lora_dropdown_values[2] in choices else choices[0]),
                    gr.update(value=config_name)
                ]
            else:
                print(translate("📦 Config has LoRA enabled but no files"))
                pending_lora_config_data = None
                # FIXED: Still return proper choices to avoid warnings
                return [
                    translate("✅ Loaded config: {0}").format(config_name),
                    gr.update(value=image_path if image_path and os.path.exists(image_path) else None),
                    gr.update(value=prompt),
                    gr.update(value=use_lora),
                    gr.update(value=lora_mode),
                    gr.update(value=lora_settings.get('lora_scales', '0.8,0.8,0.8')),
                    gr.update(choices=choices, value=choices[0] if choices else translate("なし")),
                    gr.update(choices=choices, value=choices[0] if choices else translate("なし")),
                    gr.update(choices=choices, value=choices[0] if choices else translate("なし")),
                    gr.update(value=config_name)
                ]
        else:
            print(translate("📦 No LoRA configuration needed"))
            pending_lora_config_data = None

        # Default return for non-LoRA configs
        return [
            translate("✅ Loaded config: {0}").format(config_name),
            gr.update(value=image_path if image_path and os.path.exists(image_path) else None),
            gr.update(value=prompt),
            gr.update(value=use_lora),
            gr.update(value=lora_mode),
            gr.update(value=lora_settings.get('lora_scales', '0.8,0.8,0.8')),
            gr.update(), gr.update(), gr.update(),
            gr.update(value=config_name)
        ]
    else:
        pending_lora_config_data = None
        return [
            translate("❌ {0}").format(message), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        ]

def delete_config_handler_v2(config_dropdown):

    if not config_dropdown:
        return translate("❌ No config selected for deletion"), gr.update(), gr.update(), gr.update(visible=False), None
    
    if config_queue_manager is None:
        return translate("❌ Config queue manager not initialized"), gr.update(), gr.update(), gr.update(visible=False), None
    
    # Check if file actually exists
    if not config_queue_manager.config_exists(config_dropdown):
        available_configs = config_queue_manager.get_available_configs()
        queue_status = config_queue_manager.get_queue_status()
        status_text = format_queue_status_with_batch_progress(queue_status)
        
        return (
            f"❌ Config file {config_dropdown}.json not found (refreshing list)", 
            gr.update(choices=available_configs, value=None), 
            gr.update(value=status_text), 
            gr.update(visible=False),
            None
        )
    
    # Store operation details for confirmation
    operation_data = {
        'type': 'delete',
        'config_name': config_dropdown
    }
    
    # Show confirmation message
    confirmation_msg = f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; margin: 10px 0;">
        <h3 style="color: #721c24; margin: 0 0 10px 0;">🗑️ {translate('Delete Confirmation')}</h3>
        <p style="margin: 10px 0;">{translate('Are you sure you want to delete config file "{0}.json"? This action cannot be undone.').format(config_dropdown)}</p>
        <p style="margin: 10px 0; font-weight: bold; color: #721c24;">
            {translate('Use the buttons below to confirm or cancel the operation.')}
        </p>
    </div>
    """
    
    return (
        confirmation_msg,
        gr.update(),
        gr.update(),
        gr.update(visible=True),  # Show confirmation group
        operation_data  # Store operation data - ONLY 5 VALUES RETURNED
    )

# ==============================================================================
# INTEGRATION WITH MANUAL GENERATION SYSTEM
# ==============================================================================

def validate_and_process_with_queue_check(*args):

    global queue_processing_active
    
    # Check if queue processing is active
    if queue_processing_active or (config_queue_manager and config_queue_manager.is_processing):
        # Return error message with button states (7 outputs to match start_button.click)
        yield (
            gr.skip(),  # result_video
            gr.update(visible=False),  # preview_image
            "Cannot start manual generation: Queue processing is active",  # progress_desc
            '<div style="color: red;">Queue processing is running. Please wait for completion or stop the queue.</div>',  # progress_bar
            gr.update(interactive=False, value=translate("队列处理中，手动生成已禁用")),  # start_button
            gr.update(interactive=False),  # end_button
            gr.update(interactive=False, value=translate("队列处理中...")),  # queue_start_button
            gr.update()  # seed
        )
        return
        
    # If no queue processing, proceed with normal validation
    for result in validate_and_process(*args):
        # result is a tuple: (video, preview, desc, progress, start_btn, end_btn, seed)
        if len(result) >= 6:
            video, preview, desc, progress, start_btn, end_btn = result[:6]
            seed_update = result[6] if len(result) > 6 else gr.update()
            
            # During manual generation, manage queue start button state
            if isinstance(start_btn, dict) and not start_btn.get('interactive', True):
                # Manual generation is running, disable queue start
                queue_start_state = gr.update(interactive=False, value=translate("手动生成中..."))
            else:
                # Manual generation finished, re-enable queue start
                queue_start_state = gr.update(interactive=True, value=translate("▶️ Start Queue"))
            
            # Return 8 outputs to match the expected outputs
            yield (video, preview, desc, progress, start_btn, end_btn, queue_start_state, seed_update)
        else:
            # Fallback for unexpected result format
            yield result + (gr.update(),) * (8 - len(result))

def end_process_enhanced():

    global stream
    global batch_stopped

    batch_stopped = True
    print(translate("停止ボタンが押されました。バッチ処理を停止します..."))
    stream.input_queue.push('end')

    # Return updated button states
    return (
        gr.update(value=translate("停止処理中...")),  # End button (temporary message)
        gr.update(interactive=True, value=translate("▶️ Start Queue"))  # Re-enable queue start
    )
  
# ==============================================================================
# UI CREATION AND EVENT SETUP
# ==============================================================================

def create_enhanced_config_queue_ui():
    
    with gr.Group():
        gr.Markdown(f"### " + translate("Config Queue System"))
        
        with gr.Row():
            with gr.Column(scale=2):
                config_name_input = gr.Textbox(
                    label=translate("Config Name (optional)"),
                    placeholder=translate("Leave blank for auto-generation"),
                    value="",
                    info=translate("Use existing name to overwrite, or new name to create")
                )
            with gr.Column(scale=1):
                # LOAD SAVED SETTING FOR TIMESTAMP CHECKBOX
                saved_settings = load_app_settings_f1()
                default_add_timestamp = saved_settings.get("add_timestamp_to_config", True)
                
                add_timestamp_to_config = gr.Checkbox(
                    label=translate("Add timestamp to config name"),
                    value=default_add_timestamp,  # Use saved setting
                    info=translate("Uncheck to use exact input name (may overwrite existing)")
                )

                save_config_btn = gr.Button(
                    value=translate("💾 Save Config"),
                    variant="primary"
                )
                
        # Config selection with merged refresh
        with gr.Row():
            with gr.Column(scale=2):
                available_configs = config_queue_manager.get_available_configs()
                if not available_configs:
                    available_configs = [translate("No configs available")]
                
                config_dropdown = gr.Dropdown(
                    label=translate("Select Config"),
                    choices=available_configs,
                    value=None,
                    allow_custom_value=False,
                    info=translate("Select a config file to load, queue, or delete")
                )
            with gr.Column(scale=1):
                with gr.Row():
                    load_config_btn = gr.Button(value=translate("📂 Load"), variant="secondary", scale=1)
                    delete_config_btn = gr.Button(value=translate("🗑️ Delete"), variant="secondary", scale=1)
                    merged_refresh_btn = gr.Button(value=translate("🔄 Refresh All"), variant="secondary", scale=1)
        
      
        # Queue control buttons with enhanced start
        with gr.Row():
            with gr.Column(scale=1):
                queue_config_btn = gr.Button(value=translate("📋 Queue Config"), variant="primary")
            with gr.Column(scale=1):
                clear_queue_btn = gr.Button(value=translate("🗑️ Clear Queue"), variant="secondary")
            with gr.Column(scale=1):
                enhanced_start_queue_btn = gr.Button(value=translate("▶️ Start Queue"), variant="primary")
            with gr.Column(scale=1):
                stop_queue_btn = gr.Button(value=translate("⏹️ Stop Queue"), variant="secondary")


        # Messages
        config_message = gr.Markdown("")

        # State and confirmation (same as before)
        pending_operation = gr.State(None)
        with gr.Group(visible=False) as confirmation_group:
            confirmation_html = gr.HTML("")
            with gr.Row():
                confirm_btn = gr.Button(translate("✅ Confirm"), variant="primary", scale=1)
                cancel_btn = gr.Button(translate("❌ Cancel"), variant="secondary", scale=1)

        # Enhanced status display
        queue_status_display = gr.Textbox(
            label=translate("Queue & Config Status"),
            value="",
            lines=10,
            interactive=False
        )

        
    # Initialize status
    try:
        initial_status = config_queue_manager.get_queue_status()
        initial_status_text = format_queue_status_with_batch_progress(initial_status)
        queue_status_display.value = initial_status_text
    except Exception as e:
        queue_status_display.value = translate("Status: Ready")
    
    return {
        'config_name_input': config_name_input,
        'add_timestamp_to_config': add_timestamp_to_config,
        'save_config_btn': save_config_btn,
        'config_dropdown': config_dropdown,
        'load_config_btn': load_config_btn,
        'delete_config_btn': delete_config_btn,
        'merged_refresh_btn': merged_refresh_btn,  # Changed from separate buttons
        'pending_operation': pending_operation,
        'confirmation_group': confirmation_group,
        'confirmation_html': confirmation_html,
        'confirm_btn': confirm_btn,
        'cancel_btn': cancel_btn,
        'queue_config_btn': queue_config_btn,
        'enhanced_start_queue_btn': enhanced_start_queue_btn,  # Enhanced start button
        'stop_queue_btn': stop_queue_btn,
        'clear_queue_btn': clear_queue_btn,
        'queue_status_display': queue_status_display,
        'config_message': config_message
    }

def setup_enhanced_config_queue_events(components, ui_components):
    
    # Config management events (unchanged)
    components['load_config_btn'].click(
        fn=load_config_with_delayed_lora_application_fixed,
        inputs=[components['config_dropdown']],
        outputs=[
            components['config_message'],
            ui_components['input_image'],
            ui_components['prompt'],
            ui_components['use_lora'],
            ui_components['lora_mode'],
            ui_components['lora_scales_text'],
            ui_components['lora_dropdown1'],
            ui_components['lora_dropdown2'],
            ui_components['lora_dropdown3'],
            components['config_name_input']
        ]
    )
    
    components['save_config_btn'].click(
        fn=save_current_config_handler_v3,
        inputs=[
            components['config_name_input'],
            components['add_timestamp_to_config'],  # NEW INPUT
            ui_components['input_image'],
            ui_components['prompt'],
            ui_components['use_lora'],
            ui_components['lora_mode'],
            ui_components['lora_dropdown1'],
            ui_components['lora_dropdown2'],
            ui_components['lora_dropdown3'],
            ui_components['lora_files'],
            ui_components['lora_files2'],
            ui_components['lora_files3'],
            ui_components['lora_scales_text']
        ],
        outputs=[
            components['config_message'],
            components['config_dropdown'],
            components['queue_status_display'],
            components['confirmation_group'],
            components['pending_operation']
        ]
    )
    
    
    components['delete_config_btn'].click(
        fn=delete_config_handler_v2,
        inputs=[components['config_dropdown']],
        outputs=[
            components['config_message'],
            components['config_dropdown'],
            components['queue_status_display'],
            components['confirmation_group'],
            components['pending_operation']
        ]
    )
    
    # Confirmation handlers (unchanged)
    components['confirm_btn'].click(
        fn=confirm_operation_handler_fixed,
        inputs=[components['pending_operation']],
        outputs=[
            components['config_message'],
            components['config_dropdown'],
            components['queue_status_display'],
            components['confirmation_group'],
            components['pending_operation'],
            components['config_name_input']
        ]
    )
    
    components['cancel_btn'].click(
        fn=cancel_operation_handler,
        inputs=[],
        outputs=[
            components['config_message'],
            components['config_dropdown'],
            components['queue_status_display'],
            components['confirmation_group'],
            components['pending_operation']
        ]
    )

    # UPDATED QUEUE START HANDLER - Now includes batch_count
    components['enhanced_start_queue_btn'].click(
        fn=start_queue_processing_with_current_ui_values,
        inputs=[
            # Duration settings - both controls  
            length_radio, total_second_length,
            # Frame settings
            frame_size_radio,
            # Quality settings
            steps, cfg, gs, rs, resolution, mp4_crf,
            # Generation settings
            seed, use_random_seed, use_teacache, image_strength, fp8_optimization,
            # System settings
            gpu_memory_preservation,
            # Output settings
            keep_section_videos, save_section_frames, save_tensor_data,
            frame_save_mode, output_dir, alarm_on_completion,
            # F1 mode settings
            all_padding_value, use_all_padding,
            # ADD BATCH COUNT INPUT
            batch_count
        ],
        outputs=[
            components['config_message'],
            components['queue_status_display'],
            ui_components['progress_desc'],
            ui_components['progress_bar'],
            ui_components['preview_image'],
            ui_components['result_video'],
            start_button,  # ADD: Manual start button state
            end_button     # ADD: Manual end button state  
        ]
    )
    
    components['queue_config_btn'].click(
        fn=queue_config_handler_with_confirmation,
        inputs=[components['config_dropdown']],
        outputs=[
            components['config_message'],
            components['queue_status_display'],
            components['confirmation_group'],    
            components['pending_operation']  
        ]
    )
    
    components['stop_queue_btn'].click(
        fn=stop_queue_processing_handler_fixed,
        inputs=[],
        outputs=[
            components['config_message'],
            components['queue_status_display'],
            ui_components['preview_image'],
            ui_components['result_video'] 
        ]
    )
    
    components['clear_queue_btn'].click(
        fn=clear_queue_handler,
        inputs=[],
        outputs=[
            components['config_message'],
            components['queue_status_display']
        ]
    )
        
    # Merged refresh button (unchanged)
    components['merged_refresh_btn'].click(
        fn=merged_refresh_handler_standardized,
        inputs=[],
        outputs=[
            components['config_message'],
            components['config_dropdown'],
            components['queue_status_display']
        ]
    )

def setup_periodic_queue_status_check():

    import threading
    import time
    
    def periodic_check():
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds (reduced from 30 for better responsiveness)
                
                if config_queue_manager and hasattr(config_queue_manager, 'is_processing'):
                    status = config_queue_manager.get_queue_status()
                    
                    # Check for stuck state
                    if (config_queue_manager.is_processing and 
                        status.get('queue_count', 0) == 0 and 
                        not status.get('current_config') and
                        not status.get('processing')):
                        
                        print(translate("🔧 Periodic check: Detected stuck queue state - auto-correcting"))
                        globals()['queue_processing_active'] = False
                        config_queue_manager.is_processing = False
                        config_queue_manager.current_config = None
                        
                        # Log the correction for debugging
                        print(translate("🔧 Periodic correction applied at {0}").format(time.strftime('%H:%M:%S')))
                        
            except Exception as e:
                print(translate("Periodic queue check error: {0}").format(e))
    
    # Start the periodic check thread
    check_thread = threading.Thread(target=periodic_check, daemon=True)
    check_thread.start()
    print(translate("🔧 Started periodic queue status checker (10s intervals)"))


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_lora_mode_text(lora_mode_key):

    if lora_mode_key == LORA_MODE_DIRECTORY:
        return translate("ディレクトリから選択")
    elif lora_mode_key == LORA_MODE_UPLOAD:
        return translate("ファイルアップロード")
    else:
        return translate("ディレクトリから選択")  # Default fallback

def get_current_lora_settings(use_lora, lora_mode, lora_dropdown1, lora_dropdown2, lora_dropdown3, 
                             lora_files, lora_files2, lora_files3, lora_scales_text):

    lora_settings = {
        "use_lora": use_lora,
        "lora_scales": lora_scales_text
    }
    
    if not use_lora:
        lora_settings["lora_mode_key"] = LORA_MODE_DIRECTORY
        lora_settings["lora_files"] = []
        return lora_settings
    
    if lora_mode == translate("ディレクトリから選択"):
        # Directory selection mode - handle normally
        print(translate("📁 Saving config: Directory selection mode"))
        lora_settings["lora_mode_key"] = LORA_MODE_DIRECTORY
        
        lora_paths = []
        lora_dropdown_files = []
        
        for dropdown in [lora_dropdown1, lora_dropdown2, lora_dropdown3]:
            if dropdown and dropdown != translate("なし"):
                lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
                lora_path = os.path.join(lora_dir, dropdown)
                if os.path.exists(lora_path):
                    lora_paths.append(lora_path)
                    lora_dropdown_files.append(dropdown)
                else:
                    lora_dropdown_files.append(LORA_NONE_OPTION)
            else:
                lora_dropdown_files.append(LORA_NONE_OPTION)
        
        lora_settings["lora_files"] = lora_paths
        lora_settings["lora_dropdown_files"] = lora_dropdown_files
        
    else:  # File upload mode - AUTO-CONVERT to directory mode
        print(translate("📁 Saving config: Converting file uploads to directory mode"))
        
        import shutil
        lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
        os.makedirs(lora_dir, exist_ok=True)
        
        lora_paths = []
        lora_dropdown_files = []
        copied_files = []
        
        for lora_file in [lora_files, lora_files2, lora_files3]:
            if lora_file and hasattr(lora_file, 'name'):
                try:
                    src_path = lora_file.name
                    original_filename = os.path.basename(src_path)
                    dest_path = os.path.join(lora_dir, original_filename)
                    
                    # Handle filename conflicts
                    if os.path.exists(dest_path):
                        if os.path.getsize(src_path) == os.path.getsize(dest_path):
                            print(translate("   📄 File already exists (same size): {0}").format(original_filename))
                        else:
                            name, ext = os.path.splitext(original_filename)
                            counter = 1
                            while os.path.exists(dest_path):
                                new_filename = f"{name}_copy{counter}{ext}"
                                dest_path = os.path.join(lora_dir, new_filename)
                                counter += 1
                            original_filename = os.path.basename(dest_path)
                            print(translate("   📄 Renamed to avoid conflict: {0}").format(original_filename))
                    
                    if not os.path.exists(dest_path):
                        shutil.copy2(src_path, dest_path)
                        print(translate("   ✅ Copied LoRA file: {0}").format(original_filename))
                        copied_files.append(original_filename)
                    else:
                        print(translate("   ✅ Using existing file: {0}").format(original_filename))
                        copied_files.append(original_filename)
                    
                    lora_paths.append(dest_path)
                    lora_dropdown_files.append(original_filename)
                    
                except Exception as e:
                    print(translate("   ❌ Error copying LoRA file {0}: {1}").format(lora_file.name, e))
                    continue
        
        while len(lora_dropdown_files) < 3:
            lora_dropdown_files.append(LORA_NONE_OPTION)
        
        lora_settings["lora_mode_key"] = LORA_MODE_DIRECTORY  # AUTO-CONVERTED
        lora_settings["lora_files"] = lora_paths
        lora_settings["lora_dropdown_files"] = lora_dropdown_files
        # REMOVED: Don't store conversion info in lora_settings to avoid message contamination
        
        if copied_files:
            print(translate("   📦 Auto-converted file uploads: {0}").format(', '.join(copied_files)))
    
    return lora_settings

def apply_lora_config_to_dropdowns_safe(lora_files, existing_choices=None):
    
    # Use provided choices or scan fresh
    if existing_choices is None:
        choices = scan_lora_directory()
        print(translate("🔄 Fresh scan found {0} choices: {1}...").format(len(choices), choices[:5]))
    else:
        choices = existing_choices
        print(translate("🔄 Using provided choices: {0} choices").format(len(choices)))
    
    # Initialize dropdown values
    lora_dropdown_values = [translate("なし"), translate("なし"), translate("なし")]
    applied_files = []
    
    # Apply each LoRA file
    for i, lora_path in enumerate(lora_files[:3]):
        if lora_path and os.path.exists(lora_path):
            lora_filename = os.path.basename(lora_path)
            
            # Check if filename exists in choices
            if lora_filename in choices:
                lora_dropdown_values[i] = lora_filename
                applied_files.append(lora_filename)
                print(translate("  ✅ Applied LoRA {0}: {1}").format(i+1, lora_filename))
            else:
                print(translate("  ❌ LoRA file not found in directory: {0}").format(lora_filename))
                print(translate("      Available choices: {0}...").format(choices[:10]))  # Show first 10 for debugging
                # Keep default "なし" value instead of setting invalid value
        else:
            print(translate("  ⚠️ LoRA {0} file not found or invalid: {1}").format(i+1, lora_path))
    
    # Validate all values are in choices before returning
    for i, value in enumerate(lora_dropdown_values):
        if value not in choices:
            print(translate("  🔧 Correcting invalid dropdown value: {0} -> {1}").format(value, choices[0]))
            lora_dropdown_values[i] = choices[0] if choices else translate("なし")
    
    return choices, lora_dropdown_values, applied_files

def scan_lora_directory():

    lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
    choices = []
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(lora_dir):
        os.makedirs(lora_dir, exist_ok=True)
        print(translate("LoRAディレクトリが存在しなかったため作成しました: {0}").format(lora_dir))
    
    # ディレクトリ内のファイルをリストアップ
    try:
        for filename in os.listdir(lora_dir):
            if filename.endswith(('.safetensors', '.pt', '.bin')):
                # Validate file is readable
                file_path = os.path.join(lora_dir, filename)
                if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
                    choices.append(filename)
    except Exception as e:
        print(translate("Error scanning LoRA directory: {0}").format(e))
    
    # 空の選択肢がある場合は"なし"を追加
    choices = sorted(choices)
    
    # なしの選択肢を最初に追加
    none_choice = translate("なし")
    if none_choice not in choices:
        choices.insert(0, none_choice)
    
    # 重要: すべての選択肢が確実に文字列型であることを確認
    validated_choices = []
    for choice in choices:
        if isinstance(choice, str) and choice.strip():
            validated_choices.append(choice)
        else:
            print(translate("⚠️ Skipping invalid choice: {0} (type: {1})").format(choice,type(choice)))
    
    # Ensure we always have at least the "none" option
    if not validated_choices:
        validated_choices = [translate("なし")]
    
    print(translate("🔍 Scanned LoRA directory: found {0} files + none option").format(len(validated_choices)-1))
    return validated_choices

# ==============================================================================
# INITIALIZATION AND STARTUP
# ==============================================================================

# Initialize config queue manager with error handling
try:
    config_queue_manager = ConfigQueueManager(os.path.dirname(os.path.abspath(__file__)))
    print(translate("Config queue manager initialized successfully"))
except Exception as e:
    print(translate("Error initializing config queue manager: {0}").format(e))
    config_queue_manager = None
# Setup monitoring systems if manager available
if config_queue_manager is not None:
    setup_periodic_queue_status_check()

# ==============================================================================
# QUEUE PROCESSING FUNCTIONS
# ==============================================================================

def start_queue_processing_with_current_ui_values(
    # Duration settings - both controls
    length_radio, total_second_length,
    # Frame settings
    frame_size_radio,
    # Quality settings  
    steps, cfg, gs, rs, resolution, mp4_crf,
    # Generation settings
    seed, use_random_seed, use_teacache, image_strength, fp8_optimization,
    # System settings
    gpu_memory_preservation,
    # Output settings
    keep_section_videos, save_section_frames, save_tensor_data, 
    frame_save_mode, output_dir, alarm_on_completion,
    # F1 mode settings
    all_padding_value, use_all_padding,
    # Batch count parameter
    batch_count
):
    
    if config_queue_manager is None:
        yield (
            translate("❌ Config queue manager not initialized"),  # 1. markdown (config_message)
            gr.update(),                                 # 2. textbox (queue_status_display)
            gr.update(),                                 # 3. markdown (progress_desc)
            gr.update(),                                 # 4. html (progress_bar)
            gr.update(visible=False),                    # 5. image (preview_image) - HIDE
            gr.update(visible=False),                    # 6. video (result_video) - HIDE
            gr.update(interactive=True),                 # 7. button (manual start_button)
            gr.update(interactive=False)                 # 8. button (manual end_button)
        )
        return
    
    queue_status = config_queue_manager.get_queue_status()
    has_items = queue_status.get('queue_count', 0) > 0
    
    if not has_items:
        yield (
            "❌ No items in queue",                      # 1. markdown (config_message)
            gr.update(),                                 # 2. textbox (queue_status_display)
            gr.update(),                                 # 3. markdown (progress_desc)
            gr.update(),                                 # 4. html (progress_bar)
            gr.update(visible=False),                    # 5. image (preview_image) - HIDE
            gr.update(visible=False),                    # 6. video (result_video) - HIDE
            gr.update(interactive=True),                 # 7. button (manual start_button)
            gr.update(interactive=False)                 # 8. button (manual end_button)
        )
        return
    
    
    # Store settings globally for queue worker to access
    global queue_ui_settings
    queue_ui_settings = {
        'total_second_length': max(1, int(total_second_length)),
        'length_radio': length_radio,
        'frame_size_setting': frame_size_radio,
        'latent_window_size': 4.5 if frame_size_radio == translate("0.5秒 (17フレーム)") else 9,
        'steps': int(steps),
        'cfg': float(cfg),
        'gs': float(gs), 
        'rs': float(rs),
        'resolution': int(resolution),
        'mp4_crf': int(mp4_crf),
        'seed': int(seed),
        'use_random_seed': bool(use_random_seed),
        'use_teacache': bool(use_teacache),
        'image_strength': float(image_strength),
        'fp8_optimization': bool(fp8_optimization),
        'gpu_memory_preservation': float(gpu_memory_preservation),
        'keep_section_videos': bool(keep_section_videos),
        'save_section_frames': bool(save_section_frames),
        'save_tensor_data': bool(save_tensor_data),
        'frame_save_mode': frame_save_mode,
        'output_dir': output_dir,
        'alarm_on_completion': bool(alarm_on_completion),
        'all_padding_value': float(all_padding_value),
        'use_all_padding': bool(use_all_padding),
        'batch_count': max(1, int(batch_count)),
        'n_prompt': "",
        'tensor_data_input': None,
        'use_queue': False,
        'prompt_queue_file': None,
        'save_settings_on_start': False
    }
    
    total_expected_videos = queue_status['queue_count'] * queue_ui_settings['batch_count']
    print(translate("📋 Queue starting: {0} configs × {1} batches = {2} total videos").format(queue_status['queue_count'], queue_ui_settings['batch_count'], total_expected_videos))
    
    # Start processing with batch-aware processor
    success, message = config_queue_manager.start_queue_processing(process_config_item_with_batch_support)
    
    if not success:
        yield (
            translate("❌ Failed to start: {0}").format(message),            # 1. markdown (config_message)
            gr.update(),                                 # 2. textbox (queue_status_display)
            gr.update(),                                 # 3. markdown (progress_desc)
            gr.update(),                                 # 4. html (progress_bar)
            gr.update(visible=False),                    # 5. image (preview_image) - HIDE
            gr.update(visible=False),                    # 6. video (result_video) - HIDE
            gr.update(interactive=True),                 # 7. button (manual start_button)
            gr.update(interactive=False)                 # 8. button (manual end_button)
        )
        return
    
    global queue_processing_active
    queue_processing_active = True
    initial_count = has_items
    
    # Return initial status with queue processing UI
    yield (
        translate("✅ Queue started ({0} configs × {1} batches = {2} videos)").format(queue_status['queue_count'], queue_ui_settings['batch_count'], total_expected_videos),  # 1. markdown (config_message)
        gr.update(value=format_queue_status_with_batch_progress(queue_status)),  # 2. textbox (queue_status_display)
        translate("Queue processing started: {0} total videos to generate...").format(total_expected_videos),  # 3. markdown (progress_desc)
        f'<div style="color: blue; font-weight: bold;">{translate("📋 Queue processing active - Progress UI disabled. Check console for details.")}</div>',  # 4. html (progress_bar)
        gr.update(visible=False),                        # 5. image (preview_image) - HIDE
        gr.update(visible=False),                        # 6. video (result_video) - HIDE
        gr.update(interactive=False, value=translate("队列处理中...")),  # 7. button (manual start_button)
        gr.update(interactive=False)                     # 8. button (manual end_button)
    )
    
    # Monitor with periodic updates using batch progress
    import time
    last_count = initial_count
    last_current_config = None
    
    while queue_processing_active:
        time.sleep(3.0)
        
        try:
            status = config_queue_manager.get_queue_status()
            current_count = status['queue_count']
            is_processing = status['is_processing']
            current_config = status.get('current_config')
            
            # Get batch progress for enhanced status
            batch_progress = current_batch_progress.copy()
            
            if current_count != last_count or current_config != last_current_config:
                # Calculate remaining videos with batch progress
                unprocessed_config_count = current_count
                if current_config and batch_progress['total'] > 0:
                    # Current config is being processed, subtract completed batches
                    current_config_remaining_batches = batch_progress['total'] - batch_progress['current']
                    remaining_videos = current_config_remaining_batches + (unprocessed_config_count * queue_ui_settings['batch_count'])
                else:
                    remaining_videos = unprocessed_config_count * queue_ui_settings['batch_count']
                
                if current_config:
                    if batch_progress['total'] > 0:
                        batch_info = translate("Batch {0}/{1}").format(batch_progress['current'], batch_progress['total'])
                        status_msg = translate("📋 Processing: {0} ({1}) - {2} videos remaining").format(current_config, batch_info, remaining_videos)
                        desc_msg = translate("Processing {0} - {1} - {2} videos remaining").format(current_config, batch_info, remaining_videos)
                    else:
                        status_msg = translate("📋 Processing: {0} - {1} videos remaining").format(current_config, remaining_videos)
                        desc_msg = translate("Processing {0} - {1} videos remaining").format(current_config, remaining_videos)
                else:
                    status_msg = translate("📋 Queue: {0} videos remaining").format(remaining_videos)
                    desc_msg = translate("Queue processing... {0} videos remaining").format(remaining_videos)
                
                yield (
                    status_msg,                              # 1. markdown (config_message)
                    gr.update(value=format_queue_status_with_batch_progress(status)),  # 2. textbox (queue_status_display)
                    desc_msg,                                # 3. markdown (progress_desc)
                    f'<div style="color: blue; font-weight: bold;">📋 {translate("Queue processing active - Progress UI disabled. Check console for details.")}</div>',  # 4. html (progress_bar)
                    gr.update(visible=False),                # 5. image (preview_image) - HIDE
                    gr.update(visible=False),                # 6. video (result_video) - HIDE
                    gr.update(interactive=False, value=translate("队列处理中...")),  # 7. button (manual start_button)
                    gr.update(interactive=False)             # 8. button (manual end_button)
                )
                
                last_count = current_count
                last_current_config = current_config
            
            if not is_processing and current_count == 0:
                print(translate("✅ Queue processing completed"))
                yield (
                    translate("✅ Queue completed - All {0} videos processed successfully").format(total_expected_videos),  # 1. markdown (config_message)
                    gr.update(value=format_queue_status_with_batch_progress(status)),  # 2. textbox (queue_status_display)
                    "All queue items and batches have been processed",  # 3. markdown (progress_desc)
                    '<div style="color: green; font-weight: bold;">✅ Queue processing completed</div>',  # 4. html (progress_bar)
                    gr.update(visible=True),  # preview_image - RESTORE VISIBILITY
                    gr.update(visible=True),  # result_video - RESTORE VISIBILITY
                    gr.update(interactive=True, value=translate("Start Generation")),  # 7. button (manual start_button) - RE-ENABLE
                    gr.update(interactive=False)             # 8. button (manual end_button)
                )
                break
                
        except Exception as e:
            print(translate("❌ Queue monitoring error: {0}").format(e))
            continue
    
    queue_processing_active = False
    print(translate("🏁 Queue processing monitor finished"))

# def process_config_item_with_batch_support(config_data):

#     global queue_ui_settings, current_processing_config_name, current_batch_progress
    
#     try:
#         config_name = config_data.get('config_name', 'unknown_config')
#         print(translate("🎬 Processing config: {0}").format(config_name))
        
#         # Validate that image exists for generation
#         image_path = config_data.get('image_path')
#         if not image_path or not os.path.exists(image_path):
#             print(translate("❌ Cannot generate video: Image missing for config {0}").format(config_name))
#             print(translate("    Expected path: {0}").format(image_path))
#             return False
        
#         print(translate("✅ Image validated: {0}").format(os.path.basename(image_path)))
            
#         # Get batch count from UI settings with debug logging
#         batch_count_raw = queue_ui_settings.get('batch_count', 1)
        
#         # Ensure batch_count is definitely an integer
#         if isinstance(batch_count_raw, bool):
#             print(translate("⚠️ Warning: batch_count is boolean ({0}), converting to integer").format(batch_count_raw))
#             batch_count = 1 if batch_count_raw else 1
#         else:
#             try:
#                 batch_count = int(batch_count_raw)
#             except (ValueError, TypeError):
#                 print(translate("⚠️ Warning: Could not convert batch_count to int: {0} (type: {1})").format(batch_count_raw, type(batch_count_raw)))
#                 batch_count = 1
        
#         batch_count = max(1, min(batch_count, 100))  # Ensure valid range
        
        
#         # Set the current config name for worker function to use
#         current_processing_config_name = config_name
        
#         # Initialize batch progress with validated integer
#         current_batch_progress = {"current": 0, "total": batch_count}
#         print(translate("📊 Initialized batch progress: {0}").format(current_batch_progress))
        
#         # Use the stored UI settings
#         if queue_ui_settings is None:
#             print(translate("❌ No UI settings available - using defaults"))
#             queue_ui_settings = get_current_ui_settings_for_queue()
        
#         current_ui_settings = queue_ui_settings
        
#         print(translate("🕒 Using duration from UI: {0}s").format(current_ui_settings['total_second_length']))
        
#         # Extract config data
#         image_path = config_data['image_path']
#         prompt = config_data['prompt']
#         lora_settings = config_data['lora_settings']
        
#         # Handle LoRA configuration (existing code...)
#         use_lora = lora_settings.get('use_lora', False)
#         lora_mode_key = lora_settings.get('lora_mode_key')
#         if lora_mode_key:
#             lora_mode = get_lora_mode_text(lora_mode_key)
#         else:
#             old_lora_mode = lora_settings.get('lora_mode')
#             if old_lora_mode:
#                 if 'ディレクトリ' in old_lora_mode or 'directory' in old_lora_mode.lower() or '目錄' in old_lora_mode:
#                     lora_mode = translate("ディレクトリから選択")
#                 elif 'ファイル' in old_lora_mode or 'file' in old_lora_mode.lower() or '檔案' in old_lora_mode:
#                     lora_mode = translate("ファイルアップロード")
#                 else:
#                     lora_mode = translate("ディレクトリから選択")
#             else:
#                 lora_mode = translate("ディレクトリから選択")
            
#         lora_scales_text = lora_settings.get('lora_scales', '0.8,0.8,0.8')
        
#         # Initialize LoRA parameters (existing code...)
#         lora_files_obj = None
#         lora_files2_obj = None
#         lora_files3_obj = None
#         lora_dropdown1_val = None
#         lora_dropdown2_val = None
#         lora_dropdown3_val = None
        
#         if use_lora:
#             lora_files_list = lora_settings.get('lora_files', [])
            
#             if lora_mode == translate("ディレクトリから選択"):
#                 lora_dropdown_files = lora_settings.get('lora_dropdown_files')
#                 if lora_dropdown_files:
#                     lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
                    
#                     for i, dropdown_file in enumerate(lora_dropdown_files[:3]):
#                         if dropdown_file != "none_option":
#                             lora_file_path = os.path.join(lora_dir, dropdown_file)
#                             if os.path.exists(lora_file_path):
#                                 if i == 0:
#                                     lora_dropdown1_val = dropdown_file
#                                 elif i == 1:
#                                     lora_dropdown2_val = dropdown_file
#                                 elif i == 2:
#                                     lora_dropdown3_val = dropdown_file
#                 else:
#                     if lora_files_list:
#                         if len(lora_files_list) > 0 and lora_files_list[0] and os.path.exists(lora_files_list[0]):
#                             lora_dropdown1_val = os.path.basename(lora_files_list[0])
#                         if len(lora_files_list) > 1 and lora_files_list[1] and os.path.exists(lora_files_list[1]):
#                             lora_dropdown2_val = os.path.basename(lora_files_list[1])
#                         if len(lora_files_list) > 2 and lora_files_list[2] and os.path.exists(lora_files_list[2]):
#                             lora_dropdown3_val = os.path.basename(lora_files_list[2])
#             else:
#                 if lora_files_list:
#                     if len(lora_files_list) > 0 and os.path.exists(lora_files_list[0]):
#                         lora_files_obj = type('MockFile', (), {'name': lora_files_list[0]})()
#                     if len(lora_files_list) > 1 and os.path.exists(lora_files_list[1]):
#                         lora_files2_obj = type('MockFile', (), {'name': lora_files_list[1]})()
#                     if len(lora_files_list) > 2 and os.path.exists(lora_files_list[2]):
#                         lora_files3_obj = type('MockFile', (), {'name': lora_files_list[2]})()
        
#         print(translate("🎯 Calling process() with config: {0}, batch_count: {1}, duration: {2}s").format(config_name, batch_count, current_ui_settings['total_second_length']))
        
#         def process_with_batch_tracking(*args):
#             """Simplified wrapper - let process() handle its own batch logic"""
#             print(translate("📊 Starting video generation for config: {0}").format(config_name))
            
#             # Initialize batch progress - we're starting the entire config processing
#             update_batch_progress(0, batch_count)
            
#             # Just consume the process generator without trying to intercept individual results
#             for result in process(*args):
#                 # Don't try to detect "batch completion" here - process() handles batch logic internally
#                 yield result
            
#             # When the generator is fully consumed, the entire config (all batches) is complete
#             update_batch_progress(batch_count, batch_count)
#             print(translate("✅ All {0} batch(es) completed for config: {1}").format(batch_count, config_name))
        
#         # Call the enhanced process function with batch tracking
#         result_generator = process_with_batch_tracking(
#             image_path,  # input_image
#             prompt,  # prompt
#             current_ui_settings['n_prompt'],  # n_prompt
#             current_ui_settings['seed'],  # seed
#             current_ui_settings['total_second_length'],  # total_second_length
#             current_ui_settings['latent_window_size'],  # latent_window_size
#             current_ui_settings['steps'],  # steps
#             current_ui_settings['cfg'],  # cfg
#             current_ui_settings['gs'],  # gs
#             current_ui_settings['rs'],  # rs
#             current_ui_settings['gpu_memory_preservation'],  # gpu_memory_preservation
#             current_ui_settings['use_teacache'],  # use_teacache
#             current_ui_settings['use_random_seed'],  # use_random_seed
#             current_ui_settings['mp4_crf'],  # mp4_crf
#             current_ui_settings['all_padding_value'],  # all_padding_value
#             current_ui_settings['image_strength'],  # image_strength
#             current_ui_settings['frame_size_setting'],  # frame_size_setting
#             current_ui_settings['keep_section_videos'],  # keep_section_videos
#             lora_files_obj,  # lora_files
#             lora_files2_obj,  # lora_files2
#             lora_files3_obj,  # lora_files3
#             lora_scales_text,  # lora_scales_text
#             current_ui_settings['output_dir'],  # output_dir
#             current_ui_settings['save_section_frames'],  # save_section_frames
#             current_ui_settings['use_all_padding'],  # use_all_padding
#             use_lora,  # use_lora
#             lora_mode,  # lora_mode
#             lora_dropdown1_val,  # lora_dropdown1
#             lora_dropdown2_val,  # lora_dropdown2
#             lora_dropdown3_val,  # lora_dropdown3
#             current_ui_settings['save_tensor_data'],  # save_tensor_data
#             [[None, None, ""] for _ in range(50)],  # section_settings (F1 specific)
#             current_ui_settings['tensor_data_input'],  # tensor_data_input
#             current_ui_settings['fp8_optimization'],  # fp8_optimization
#             current_ui_settings['resolution'],  # resolution
#             batch_count,  # USE VALIDATED batch_count DIRECTLY
#             current_ui_settings['frame_save_mode'],  # frame_save_mode
#             current_ui_settings['use_queue'],  # use_queue
#             current_ui_settings['prompt_queue_file'],  # prompt_queue_file
#             current_ui_settings['save_settings_on_start'],  # save_settings_on_start
#             current_ui_settings['alarm_on_completion']  # alarm_on_completion
#         )
        
#         # Consume the generator - RESTORED
#         step_count = 0
#         for result in result_generator:
#             step_count += 1
        
#         # Reset batch progress when done
#         current_batch_progress = {"current": 0, "total": 0}
        
#         return True
        
#     except Exception as e:
#         print(translate("❌ Config processing error: {0}").format(e))
#         import traceback
#         traceback.print_exc()
#         return False
#     finally:
#         # Clear the config name when done
#         current_processing_config_name = None
#         current_batch_progress = {"current": 0, "total": 0}
#         return True

def process_config_item_with_batch_support(config_data):
    global queue_ui_settings, current_processing_config_name, current_batch_progress
    
    try:
        config_name = config_data.get('config_name', 'unknown_config')
        print(translate("🎬 Processing config: {0}").format(config_name))
        
        # FIXED: Use ConfigQueueManager's load_config_for_generation method
        # This method handles path resolution for both images and LoRA files
        if config_queue_manager is None:
            print(translate("❌ Config queue manager not available"))
            return False
            
        # Load config with proper path resolution
        success, resolved_config_data, message = config_queue_manager.load_config_for_generation(config_name)
        
        if not success:
            print(translate("❌ Cannot load config for generation: {0}").format(config_name))
            print(translate("    Error: {0}").format(message))
            return False
            
        # Use the resolved config data instead of the original
        config_data = resolved_config_data
        
        # Validate that image exists for generation (now with resolved path)
        image_path = config_data.get('image_path')
        if not image_path or not os.path.exists(image_path):
            print(translate("❌ Cannot generate video: Image missing for config {0}").format(config_name))
            print(translate("    Expected path: {0}").format(image_path))
            return False
        
        print(translate("✅ Image validated: {0}").format(os.path.basename(image_path)))
            
        # Get batch count from UI settings with debug logging
        batch_count_raw = queue_ui_settings.get('batch_count', 1)
        
        # Ensure batch_count is definitely an integer
        if isinstance(batch_count_raw, bool):
            print(translate("⚠️ Warning: batch_count is boolean ({0}), converting to integer").format(batch_count_raw))
            batch_count = 1 if batch_count_raw else 1
        else:
            try:
                batch_count = int(batch_count_raw)
            except (ValueError, TypeError):
                print(translate("⚠️ Warning: Could not convert batch_count to int: {0} (type: {1})").format(batch_count_raw, type(batch_count_raw)))
                batch_count = 1
        
        batch_count = max(1, min(batch_count, 100))  # Ensure valid range
        
        # Set the current config name for worker function to use
        current_processing_config_name = config_name
        
        # Initialize batch progress with validated integer
        current_batch_progress = {"current": 0, "total": batch_count}
        print(translate("📊 Initialized batch progress: {0}").format(current_batch_progress))
        
        # Use the stored UI settings
        if queue_ui_settings is None:
            print(translate("❌ No UI settings available - using defaults"))
            queue_ui_settings = get_current_ui_settings_for_queue()
        
        current_ui_settings = queue_ui_settings
        
        print(translate("🕒 Using duration from UI: {0}s").format(current_ui_settings['total_second_length']))
        
        # Extract config data (now using resolved paths)
        image_path = config_data['image_path']  # Already resolved
        prompt = config_data['prompt']
        lora_settings = config_data['lora_settings']
        
        # Handle LoRA configuration with resolved paths
        use_lora = lora_settings.get('use_lora', False)
        lora_mode_key = lora_settings.get('lora_mode_key')
        if lora_mode_key:
            lora_mode = get_lora_mode_text(lora_mode_key)
        else:
            old_lora_mode = lora_settings.get('lora_mode')
            if old_lora_mode:
                if 'ディレクトリ' in old_lora_mode or 'directory' in old_lora_mode.lower() or '目錄' in old_lora_mode:
                    lora_mode = translate("ディレクトリから選択")
                elif 'ファイル' in old_lora_mode or 'file' in old_lora_mode.lower() or '檔案' in old_lora_mode:
                    lora_mode = translate("ファイルアップロード")
                else:
                    lora_mode = translate("ディレクトリから選択")
            else:
                lora_mode = translate("ディレクトリから選択")
            
        lora_scales_text = lora_settings.get('lora_scales', '0.8,0.8,0.8')
        
        # Initialize LoRA parameters
        lora_files_obj = None
        lora_files2_obj = None
        lora_files3_obj = None
        lora_dropdown1_val = None
        lora_dropdown2_val = None
        lora_dropdown3_val = None
        
        if use_lora:
            # LoRA files are now resolved to absolute paths by load_config_for_generation
            lora_files_list = lora_settings.get('lora_files', [])
            
            if lora_mode == translate("ディレクトリから選択"):
                lora_dropdown_files = lora_settings.get('lora_dropdown_files')
                if lora_dropdown_files:
                    # Use the resolved absolute paths, but extract filenames for dropdown values
                    for i, lora_file_path in enumerate(lora_files_list[:3]):
                        if lora_file_path and os.path.exists(lora_file_path):
                            filename = os.path.basename(lora_file_path)
                            if i == 0:
                                lora_dropdown1_val = filename
                            elif i == 1:
                                lora_dropdown2_val = filename
                            elif i == 2:
                                lora_dropdown3_val = filename
                else:
                    # Fallback: extract filenames from resolved paths
                    if lora_files_list:
                        if len(lora_files_list) > 0 and lora_files_list[0] and os.path.exists(lora_files_list[0]):
                            lora_dropdown1_val = os.path.basename(lora_files_list[0])
                        if len(lora_files_list) > 1 and lora_files_list[1] and os.path.exists(lora_files_list[1]):
                            lora_dropdown2_val = os.path.basename(lora_files_list[1])
                        if len(lora_files_list) > 2 and lora_files_list[2] and os.path.exists(lora_files_list[2]):
                            lora_dropdown3_val = os.path.basename(lora_files_list[2])
            else:
                # File upload mode - create mock file objects from resolved paths
                if lora_files_list:
                    if len(lora_files_list) > 0 and os.path.exists(lora_files_list[0]):
                        lora_files_obj = type('MockFile', (), {'name': lora_files_list[0]})()
                    if len(lora_files_list) > 1 and os.path.exists(lora_files_list[1]):
                        lora_files2_obj = type('MockFile', (), {'name': lora_files_list[1]})()
                    if len(lora_files_list) > 2 and os.path.exists(lora_files_list[2]):
                        lora_files3_obj = type('MockFile', (), {'name': lora_files_list[2]})()
        
        print(translate("🎯 Calling process() with config: {0}, batch_count: {1}, duration: {2}s").format(config_name, batch_count, current_ui_settings['total_second_length']))
        
        def process_with_batch_tracking(*args):
            """Simplified wrapper - let process() handle its own batch logic"""
            print(translate("📊 Starting video generation for config: {0}").format(config_name))
            
            # Initialize batch progress - we're starting the entire config processing
            update_batch_progress(0, batch_count)
            
            # Just consume the process generator without trying to intercept individual results
            for result in process(*args):
                # Don't try to detect "batch completion" here - process() handles batch logic internally
                yield result
            
            # When the generator is fully consumed, the entire config (all batches) is complete
            update_batch_progress(batch_count, batch_count)
            print(translate("✅ All {0} batch(es) completed for config: {1}").format(batch_count, config_name))
        
        # Call the enhanced process function with batch tracking
        result_generator = process_with_batch_tracking(
            image_path,  # input_image (now resolved)
            prompt,  # prompt
            current_ui_settings['n_prompt'],  # n_prompt
            current_ui_settings['seed'],  # seed
            current_ui_settings['total_second_length'],  # total_second_length
            current_ui_settings['latent_window_size'],  # latent_window_size
            current_ui_settings['steps'],  # steps
            current_ui_settings['cfg'],  # cfg
            current_ui_settings['gs'],  # gs
            current_ui_settings['rs'],  # rs
            current_ui_settings['gpu_memory_preservation'],  # gpu_memory_preservation
            current_ui_settings['use_teacache'],  # use_teacache
            current_ui_settings['use_random_seed'],  # use_random_seed
            current_ui_settings['mp4_crf'],  # mp4_crf
            current_ui_settings['all_padding_value'],  # all_padding_value
            current_ui_settings['image_strength'],  # image_strength
            current_ui_settings['frame_size_setting'],  # frame_size_setting
            current_ui_settings['keep_section_videos'],  # keep_section_videos
            lora_files_obj,  # lora_files
            lora_files2_obj,  # lora_files2
            lora_files3_obj,  # lora_files3
            lora_scales_text,  # lora_scales_text
            current_ui_settings['output_dir'],  # output_dir
            current_ui_settings['save_section_frames'],  # save_section_frames
            current_ui_settings['use_all_padding'],  # use_all_padding
            use_lora,  # use_lora
            lora_mode,  # lora_mode
            lora_dropdown1_val,  # lora_dropdown1
            lora_dropdown2_val,  # lora_dropdown2
            lora_dropdown3_val,  # lora_dropdown3
            current_ui_settings['save_tensor_data'],  # save_tensor_data
            [[None, None, ""] for _ in range(50)],  # section_settings (F1 specific)
            current_ui_settings['tensor_data_input'],  # tensor_data_input
            current_ui_settings['fp8_optimization'],  # fp8_optimization
            current_ui_settings['resolution'],  # resolution
            batch_count,  # USE VALIDATED batch_count DIRECTLY
            current_ui_settings['frame_save_mode'],  # frame_save_mode
            current_ui_settings['use_queue'],  # use_queue
            current_ui_settings['prompt_queue_file'],  # prompt_queue_file
            current_ui_settings['save_settings_on_start'],  # save_settings_on_start
            current_ui_settings['alarm_on_completion']  # alarm_on_completion
        )
        
        # Consume the generator
        step_count = 0
        for result in result_generator:
            step_count += 1
        
        # Reset batch progress when done
        current_batch_progress = {"current": 0, "total": 0}
        
        return True
        
    except Exception as e:
        print(translate("❌ Config processing error: {0}").format(e))
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clear the config name when done
        current_processing_config_name = None
        current_batch_progress = {"current": 0, "total": 0}
        return True

# ==============================================================================
# QUEUE STATUS AND MONITORING
# ==============================================================================

def format_queue_status_with_batch_progress(status):

    if "error" in status:
        return f"❌ Error: {status['error']}"
   
    lines = []
   
    # Processing status with batch information
    if status['is_processing']:
        lines.append(translate("🔄 Status: PROCESSING"))
       
        current_config = status.get('current_config')
        batch_progress = status.get('batch_progress', {"current": 0, "total": 0})
        configs_remaining = status.get('configs_remaining', 0)  # Use the field from status
       
        if current_config:
            if batch_progress['total'] > 0:
                batch_info = translate("Batch ({0}/{1})").format(batch_progress['current'], batch_progress['total'])
                queue_info = translate("{0} file(s) in queue").format(configs_remaining)
                lines.append(translate("📹 Processing: {0}, {1}, {2}").format(current_config, batch_info, queue_info))
            else:
                lines.append(translate("📹 Processing: {0}, {1} file(s) in queue").format(current_config, configs_remaining))
        elif status.get('processing'):
            lines.append(translate("📹 Current: {0}").format(status['processing']))
    else:
        lines.append(translate("⏸️ Status: IDLE"))
   
    # Queue information
    queue_count = status['queue_count']
    lines.append(translate("📋 Queue: {0} items").format(queue_count))
   
    # Available configs count
    try:
        if config_queue_manager:
            available_configs = config_queue_manager.get_available_configs()
            lines.append(translate("📁 Configs: {0} available").format(len(available_configs)))
    except:
        pass
   
    # Pending items (limited display)
    if status['queued']:
        lines.append(translate("⏳ Pending:"))
        for i, config in enumerate(status['queued'][:CONST_queued_shown_count]):
            lines.append(f"   {i+1}. {config}")
        if len(status['queued']) > CONST_queued_shown_count:
            lines.append(translate("   ... and {0} more").format(len(status['queued']) - CONST_queued_shown_count))
   
    # Recently completed (newest first)
    if status['completed']:
        lines.append(translate("✅ Recently completed: {0} (newest first)").format(len(status['completed'])))
        for config in status['completed'][:CONST_latest_finish_count]:
            lines.append(f"   ✓ {config}")
   
    # Timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    lines.append(translate("🕒 Last updated: {0}").format(timestamp))
   
    return "\n".join(lines)

def update_batch_progress(current_batch, total_batches):

    global current_batch_progress
    current_batch_progress = {"current": current_batch, "total": total_batches}
    print(translate("📊 Batch progress updated: {0}/{1}").format(current_batch, total_batches))

# ==============================================================================
# CONFIRMATION SYSTEM AND UI EVENT HANDLERS
# ==============================================================================

def confirm_operation_handler_fixed(operation_data):

    if not operation_data:
        return "❌ No pending operation", gr.update(), gr.update(), gr.update(visible=False), None, gr.update()
    
    try:
        if operation_data['type'] == 'overwrite' or operation_data['type'] == 'overwrite_exact':
            # Existing config save overwrite logic
            result = perform_save_operation_v3(
                operation_data['config_name_input'],
                operation_data['add_timestamp'],
                operation_data['input_image'],
                operation_data['prompt'],
                operation_data['use_lora'],
                operation_data['lora_mode'],
                operation_data['lora_dropdown1'],
                operation_data['lora_dropdown2'],
                operation_data['lora_dropdown3'],
                operation_data['lora_files'],
                operation_data['lora_files2'],
                operation_data['lora_files3'],
                operation_data['lora_scales_text']
            )
            # Add empty config name update for save operations
            return result + (gr.update(),)
        
        elif operation_data['type'] == 'queue_overwrite':
            # Queue overwrite logic
            config_name = operation_data['config_name']
            
            # Remove existing queued file
            queue_file = os.path.join(config_queue_manager.queue_dir, f"{config_name}.json")
            if os.path.exists(queue_file):
                os.remove(queue_file)
                print(translate("🔄 Removed existing queued config: {0}").format(config_name))
            
            # Queue the config using existing method
            success, message = config_queue_manager.queue_config(config_name)
            
            if success:
                queue_status = config_queue_manager.get_queue_status()
                status_text = format_queue_status_with_batch_progress(queue_status)
                
                return (
                    f"✅ Config overwritten in queue: {config_name}",  # config_message
                    gr.update(),  # config_dropdown (no change)
                    gr.update(value=status_text),  # queue_status_display
                    gr.update(visible=False),  # confirmation_group
                    None,  # pending_operation
                    gr.update()  # config_name_input (6th output - MISSING IN ORIGINAL)
                )
            else:
                return f"❌ {message}", gr.update(), gr.update(), gr.update(visible=False), None, gr.update()
        
        elif operation_data['type'] == 'delete':
            # DELETE OPERATION - CLEAR CONFIG NAME INPUT
            config_name = operation_data['config_name']
            success, message = config_queue_manager.delete_config(config_name)
            
            if success:
                available_configs = config_queue_manager.get_available_configs()
                queue_status = config_queue_manager.get_queue_status()
                status_text = format_queue_status_with_batch_progress(queue_status)
                new_value = available_configs[0] if available_configs else None
                
                return (
                    f"✅ {translate('Config deleted successfully')}: {config_name}.json",
                    gr.update(choices=available_configs, value=new_value),
                    gr.update(value=status_text),
                    gr.update(visible=False),  # Hide confirmation group
                    None,  # Clear operation data
                    gr.update(value="")  # CLEAR the config name input textbox
                )
            else:
                return f"❌ {message}", gr.update(), gr.update(), gr.update(visible=False), None, gr.update()
        
        else:
            return "❌ Unknown operation type", gr.update(), gr.update(), gr.update(visible=False), None, gr.update()
            
    except Exception as e:
        return f"❌ Error confirming operation: {str(e)}", gr.update(), gr.update(), gr.update(visible=False), None, gr.update()

def toggle_lora_full_update(use_lora_val):

    global previous_lora_mode, pending_lora_config_data

    print(translate("🔄 toggle_lora_full_update called: use_lora={0}").format(use_lora_val))
    
    # Get basic visibility settings
    settings_updates = toggle_lora_settings(use_lora_val)
    
    if not use_lora_val:
        # LoRA disabled - save current mode and clear pending data
        current_mode = getattr(lora_mode, 'value', translate("ディレクトリから選択"))
        if current_mode:
            previous_lora_mode = current_mode
        pending_lora_config_data = None
        print(translate("    LoRA disabled, cleared pending data"))
        return settings_updates + [gr.update(), gr.update(), gr.update()]
    
    # LoRA enabled
    print(translate("    LoRA enabled..."))
    
    # Check for pending configuration
    if pending_lora_config_data is not None:
        print(translate("    Found pending LoRA config for: {0}").format(pending_lora_config_data.get('config_name')))
        
        # Use the already-applied values from the config loading
        if 'applied_values' in pending_lora_config_data:
            lora_dropdown_values = pending_lora_config_data['applied_values']
            choices = scan_lora_directory()  # Fresh scan
            
            print(translate("    Reapplying stored values: {0}").format(lora_dropdown_values))
            
            # Set directory mode with stored values
            settings_updates[0] = gr.update(visible=True, value=translate("ディレクトリから選択"))
            settings_updates[1] = gr.update(visible=False)
            settings_updates[2] = gr.update(visible=True)
            
            dropdown_updates = [
                gr.update(choices=choices, value=lora_dropdown_values[0]),
                gr.update(choices=choices, value=lora_dropdown_values[1]),
                gr.update(choices=choices, value=lora_dropdown_values[2])
            ]
            
            return settings_updates + dropdown_updates
        else:
            # Fallback: reapply from file paths
            lora_files = pending_lora_config_data.get('files', [])
            choices, lora_dropdown_values, applied_files = apply_lora_config_to_dropdowns_safe(lora_files)
            
            settings_updates[0] = gr.update(visible=True, value=translate("ディレクトリから選択"))
            settings_updates[1] = gr.update(visible=False)
            settings_updates[2] = gr.update(visible=True)
            
            dropdown_updates = [
                gr.update(choices=choices, value=lora_dropdown_values[0]),
                gr.update(choices=choices, value=lora_dropdown_values[1]),
                gr.update(choices=choices, value=lora_dropdown_values[2])
            ]
            
            return settings_updates + dropdown_updates
    
    # No pending config - use default behavior
    print(translate("    No pending config, using default behavior"))
    
    if previous_lora_mode == translate("ファイルアップロード"):
        settings_updates[0] = gr.update(visible=True, value=translate("ファイルアップロード"))
        settings_updates[1] = gr.update(visible=True)
        settings_updates[2] = gr.update(visible=False)
        return settings_updates + [gr.update(), gr.update(), gr.update()]
    else:
        # Default to directory mode
        choices = scan_lora_directory()
        settings_updates[0] = gr.update(visible=True, value=translate("ディレクトリから選択"))
        settings_updates[1] = gr.update(visible=False)
        settings_updates[2] = gr.update(visible=True)
        
        dropdown_updates = [
            gr.update(choices=choices, value=choices[0] if choices else translate("なし")),
            gr.update(choices=choices, value=choices[0] if choices else translate("なし")),
            gr.update(choices=choices, value=choices[0] if choices else translate("なし"))
        ]
        
        return settings_updates + dropdown_updates

def fix_prompt_preset_dropdown_initialization():
    
    # This should be called during UI setup to fix the choices
    try:
        # Get presets data
        from eichi_utils.preset_manager import load_presets
        
        presets_data = load_presets()
        choices = [preset["name"] for preset in presets_data["presets"]]
        
        # Separate default and user presets
        default_presets = [name for name in choices if any(p["name"] == name and p.get("is_default", False) for p in presets_data["presets"])]
        user_presets = [name for name in choices if name not in default_presets]
        
        # Create sorted choices
        sorted_choices = [(name, name) for name in sorted(default_presets) + sorted(user_presets)]
        
        
        # Check if "起動時デフォルト" is in choices
        startup_default = translate("起動時デフォルト")
        choice_names = [choice[1] for choice in sorted_choices]
        
        if startup_default not in choice_names: 
            # Add it if missing
            sorted_choices.insert(0, (startup_default, startup_default))
        # else:
        #     print(f"✅ '{startup_default}' found in preset choices")
        
        return sorted_choices, startup_default
        
    except Exception as e:
        print(translate("Error fixing prompt preset dropdown: {0}").format(e))
        return [], ""

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
            image_path = os.path.join(input_dir, file)
            image_files.append(image_path)

    print(translate("入力ディレクトリから画像ファイル{0}個を読み込みました").format(len(image_files)))

    image_queue_files = image_files
    return image_files

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf=16, all_padding_value=1.0, image_strength=1.0, keep_section_videos=False, lora_files=None, lora_files2=None, lora_files3=None, lora_scales_text="0.8,0.8,0.8", output_dir=None, save_section_frames=False, use_all_padding=False, use_lora=False, lora_mode=None, lora_dropdown1=None, lora_dropdown2=None, lora_dropdown3=None, save_tensor_data=False, tensor_data_input=None, fp8_optimization=False, resolution=640, batch_index=None, frame_save_mode=None):

    # frame_save_modeに基づいてフラグを設定
    save_latent_frames = False
    save_last_section_frames = False
    
    if frame_save_mode == translate("全フレーム画像保存"):
        save_latent_frames = True
    elif frame_save_mode == translate("最終セクションのみ全フレーム画像保存"):
        save_last_section_frames = True

    # 入力画像または表示されている最後のキーフレーム画像のいずれかが存在するか確認
    if isinstance(input_image, str):
        has_any_image = (input_image is not None)
    else:
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
        except Exception as e:
            print(translate("セクション数計算エラー: {0}").format(e))

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

    has_any_image = has_any_image or (last_visible_section_image is not None)
    if not has_any_image:
        raise ValueError("入力画像または表示されている最後のキーフレーム画像のいずれかが必要です")

    # 入力画像がない場合はキーフレーム画像を使用
    if input_image is None and last_visible_section_image is not None:
        print(translate("入力画像が指定されていないため、セクション{0}のキーフレーム画像を使用します").format(last_visible_section_num))
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


    #Get config file name
    def get_job_id_with_config_name(batch_index=None):
        """Generate job ID with config name if processing queue, otherwise use timestamp"""
        global current_processing_config_name
        
        batch_suffix = f"_batch{batch_index+1}" if batch_index is not None else ""
        
        if current_processing_config_name:
            # Queue processing - use config name + timestamp
            timestamp = generate_timestamp()
            job_id = f"{current_processing_config_name}_{timestamp}{batch_suffix}"
            print(translate("📁 Queue video naming: {0}").format(job_id))
        else:
            # Manual processing - use original naming
            job_id = generate_timestamp() + batch_suffix
            print(translate("📁 Manual video naming: {0}").format(job_id))
        
        return job_id

    # Then in the worker function, replace the job_id line with:
    job_id = get_job_id_with_config_name(batch_index)

    # 現在のバッチ番号が指定されていれば使用する
    # endframe_ichiの仕様に合わせて+1した値を使用
    batch_suffix = f"_batch{batch_index+1}" if batch_index is not None else ""
    #job_id = generate_timestamp() + batch_suffix

    # F1モードでは順生成を行うため、latent_paddingsのロジックは使用しない
    # 全セクション数を設定
    total_sections = total_latent_sections
    
    # 正確なセクション数の再計算と確認（トラブルシューティング用）
    if total_second_length > 0:
        sections_by_frames = int(max(round((total_second_length * 30) / (latent_window_size * 4 - 3)), 1))
        if sections_by_frames != total_sections:
            print(translate("セクション数に不一致があります！計算値を優先します"))
            total_sections = sections_by_frames

    print(translate("セクション生成詳細 (F1モード):"))
    print(translate("  - 合計セクション数: {0} (最終確定値)").format(total_sections))
    frame_count = latent_window_size * 4 - 3
    print(translate("  - 各セクションのフレーム数: 約{0}フレーム (latent_window_size: {1})").format(frame_count, latent_window_size))


    # All stream.output_queue.push() calls should now go through the proxy correctly
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # F1モードのプロンプト処理
        section_map = None
        section_numbers_sorted = []

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
        
        # イメージキューのカスタムプロンプトの場合、詳細ログを追加
        # リクエスト内のキュー情報からカスタムプロンプトの使用をいつでも確認できるよう変数からチェック
        # ローカル変数ではなく、グローバル設定から確認する
        using_custom_txt = False
        if queue_enabled and queue_type == "image" and batch_index is not None and batch_index > 0:
            if batch_index - 1 < len(image_queue_files):
                img_path = image_queue_files[batch_index - 1]
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                if os.path.exists(txt_path):
                    using_custom_txt = True
        
        # 実際に使用されるプロンプトを必ず表示
        actual_prompt = prompt  # 実際に使用するプロンプト
        prompt_source = translate("共通プロンプト")  # プロンプトの種類

        # プロンプトソースの判定
        if queue_enabled and queue_type == "prompt" and batch_index is not None:
            # プロンプトキューの場合
            prompt_source = translate("プロンプトキュー")
            print(translate("プロンプトキューからのプロンプトをエンコードしています..."))
        elif using_custom_txt:
            # イメージキューのカスタムプロンプトの場合
            actual_prompt = prompt  # カスタムプロンプトを使用
            prompt_source = translate("カスタムプロンプト(イメージキュー)")
            print(translate("カスタムプロンプトをエンコードしています..."))
        else:
            # 通常の共通プロンプトの場合
            print(translate("共通プロンプトをエンコードしています..."))
        
        # プロンプトの内容とソースを表示
        print(translate("プロンプト情報: ソース: {0}").format(prompt_source))
        print(translate("プロンプト情報: 内容: {0}").format(actual_prompt))
        
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)


        # これ以降の処理は text_encoder, text_encoder_2 は不要なので、メモリ解放してしまって構わない
        if not high_vram:
            text_encoder, text_encoder_2 = None, None
            text_encoder_manager.dispose_text_encoders()

        # テンソルデータのアップロードがあれば読み込み
        uploaded_tensor = None
        if tensor_data_input is not None:
            try:
                # リスト型の場合、最初の要素を取得
                if isinstance(tensor_data_input, list):
                    if tensor_data_input and hasattr(tensor_data_input[0], 'name'):
                        tensor_data_input = tensor_data_input[0]
                    else:
                        tensor_data_input = None
                
                if tensor_data_input is not None and hasattr(tensor_data_input, 'name'):
                    tensor_path = tensor_data_input.name
                    print(translate("テンソルデータを読み込み: {0}").format(os.path.basename(tensor_path)))
                    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate('Loading tensor data ...')))))

                    # safetensorsからテンソルを読み込み
                    tensor_dict = sf.load_file(tensor_path)

                    # テンソルに含まれているキーとシェイプを確認
                    print(translate("テンソルデータの内容:"))
                    for key, tensor in tensor_dict.items():
                        print(translate("  - {0}: shape={1}, dtype={2}").format(key, tensor.shape, tensor.dtype))

                    # history_latentsと呼ばれるキーが存在するか確認
                    if "history_latents" in tensor_dict:
                        uploaded_tensor = tensor_dict["history_latents"]
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

        # バッチ処理で対応するために入力画像を使用
        # worker関数に渡される入力画像を直接使用（input_image）
        input_image_np, input_image_pt, height, width = preprocess_image(input_image, resolution=resolution)

        # 入力画像にメタデータを埋め込んで保存
        # endframe_ichiの仕様に完全に合わせる - バッチ番号を追加しない
        initial_image_path = os.path.join(outputs_folder, f'{job_id}.png')
        Image.fromarray(input_image_np).save(initial_image_path)

        # メタデータの埋め込み
        metadata = {
            PROMPT_KEY: prompt,
            SEED_KEY: seed
        }
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

                    decoded_image = vae_decode(preview_latent, vae)
                    decoded_image = (decoded_image[0, :, 0] * 127.5 + 127.5).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
                    # デコードした画像を保存
                    Image.fromarray(decoded_image).save(os.path.join(outputs_folder, f'{job_id}_tensor_preview.png'))
                    # デコードした画像を入力画像として設定
                    input_image = decoded_image
                    # 前処理用のデータも生成
                    input_image_np, input_image_pt, height, width = preprocess_image(input_image)
                    print(translate("テンソルからデコードした画像を生成しました: {0}x{1}").format(height, width))
                except Exception as e:
                    print(translate("テンソルからのデコード中にエラーが発生しました: {0}").format(e))
                    # デコードに失敗した場合は通常の処理を続行

            # UI上でテンソルデータの情報を表示
            tensor_info = translate("テンソルデータ ({0}フレーム) を検出しました。動画生成後に後方に結合します。").format(uploaded_tensor.shape[2])
            stream.output_queue.push(('progress', (None, tensor_info, make_progress_bar_html(10, translate('テンソルデータを後方に結合')))))

        # 常に入力画像から通常のエンコーディングを行う
        start_latent = vae_encode(input_image_pt, vae)

        # 簡略化設計: section_latents機能を削除

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

        # 初期フレーム準備
        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None

        # 開始フレームをhistory_latentsに追加
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1  # 最初のフレームを含むので1から開始

        # -------- LoRA 設定 START ---------

        # UI設定のuse_loraフラグ値を保存
        original_use_lora = use_lora

        # LoRAの環境変数設定（PYTORCH_CUDA_ALLOC_CONF）
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            old_env = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            print(translate("CUDA環境変数設定: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (元の値: {0})").format(old_env))

        # 次回のtransformer設定を更新
        current_lora_paths = []
        current_lora_scales = []
        
        if use_lora and has_lora_support:
            # モードに応じてLoRAファイルを処理
            if lora_mode == translate("ディレクトリから選択"):
                print(translate("ディレクトリから選択モードでLoRAを処理します"))
                # ドロップダウンの値を取得
                for dropdown in [lora_dropdown1, lora_dropdown2, lora_dropdown3]:
                    if dropdown is not None and dropdown != translate("なし") and dropdown != 0:
                        # なし以外が選択されている場合、パスを生成
                        lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
                        lora_path = os.path.join(lora_dir, dropdown)
                        if os.path.exists(lora_path):
                            current_lora_paths.append(lora_path)
                            print(translate("LoRAファイルを追加: {0}").format(lora_path))
                        else:
                            print(translate("LoRAファイルが見つかりません: {0}").format(lora_path))
                
                # 数値0の特別処理（インデックス0の要素として解釈）
                if lora_dropdown2 == 0:
                    try:
                        # ディレクトリから選択が入ってるはずなので、選択肢からインデックス0の項目（なし）を取得
                        choices = scan_lora_directory()
                        if choices and len(choices) > 0:
                            if choices[0] != translate("なし"):
                                print(translate("予期しない選択肢リスト: 最初の要素が「なし」ではありません: {0}").format(choices[0]))
                    except Exception as e:
                        print(translate("ドロップダウン2の特別処理でエラー: {0}").format(e))
            else:
                # ファイルアップロードモード
                print(translate("ファイルアップロードモードでLoRAを処理します"))
                # LoRAファイルを収集
                if lora_files is not None:
                    if isinstance(lora_files, list):
                        # 複数のLoRAファイル（将来のGradioバージョン用）
                        current_lora_paths.extend([file.name for file in lora_files])
                    else:
                        # 単一のLoRAファイル
                        current_lora_paths.append(lora_files.name)
                
                # 2つ目のLoRAファイルがあれば追加
                if lora_files2 is not None:
                    if isinstance(lora_files2, list):
                        # 複数のLoRAファイル（将来のGradioバージョン用）
                        current_lora_paths.extend([file.name for file in lora_files2])
                    else:
                        # 単一のLoRAファイル
                        current_lora_paths.append(lora_files2.name)
                
                # 3つ目のLoRAファイルがあれば追加（F1版でも対応）
                if lora_files3 is not None:
                    if isinstance(lora_files3, list):
                        current_lora_paths.extend([file.name for file in lora_files3])
                    else:
                        current_lora_paths.append(lora_files3.name)
            
            # スケール値をテキストから解析
            if current_lora_paths:  # LoRAパスがある場合のみ解析
                try:
                    scales_text = lora_scales_text.strip()
                    if scales_text:
                        # カンマ区切りのスケール値を解析
                        scales = [float(scale.strip()) for scale in scales_text.split(',')]
                        current_lora_scales = scales
                        
                        # 足りない場合は0.8で埋める
                        if len(scales) < len(current_lora_paths):
                            current_lora_scales.extend([0.8] * (len(current_lora_paths) - len(scales)))
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
            print(translate("UIでLoRA使用が有効化されているため、LoRA使用を有効にします"))

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
        print(translate("セクション処理開始前のtransformer状態チェック..."))
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

        # セクション順次処理
        for i_section in range(total_sections):
            # 先に変数を定義
            is_first_section = i_section == 0

            # 単純なインデックスによる判定
            is_last_section = i_section == total_sections - 1

            # F1モードではオールパディング機能は無効化されているため、常に固定値を使用
            # この値はF1モードでは実際には使用されないが、ログ出力のために計算する
            latent_padding = 1  # 固定値

            latent_padding_size = int(latent_padding * latent_window_size)

            # 定義後にログ出力（F1モードではオールパディングは常に無効）
            padding_info = translate("パディング値: {0} (F1モードでは影響なし)").format(latent_padding)
            print(translate("■ セクション{0}の処理開始 ({1})").format(i_section, padding_info))
            print(translate("  - 現在の生成フレーム数: {0}フレーム").format(total_generated_latent_frames * 4 - 3))
            print(translate("  - 生成予定フレーム数: {0}フレーム").format(num_frames))
            print(translate("  - 最初のセクション?: {0}").format(is_first_section))
            print(translate("  - 最後のセクション?: {0}").format(is_last_section))
            # set current_latent here
            # 常に開始フレームを使用
            current_latent = start_latent


            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            # 共通プロンプトを使用
            current_llama_vec, current_clip_l_pooler, current_llama_attention_mask = llama_vec, clip_l_pooler, llama_attention_mask

            print(translate('latent_padding_size = {0}, is_last_section = {1}').format(latent_padding_size, is_last_section))


            # COMMENTED OUT: セクション処理前のメモリ解放（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()

            # latent_window_sizeが4.5の場合は特別に5を使用
            effective_window_size = 5 if latent_window_size == 4.5 else int(latent_window_size)
            # 必ず整数のリストを使用
            indices = torch.arange(0, sum([1, 16, 2, 1, effective_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, effective_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

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

            # Image影響度を計算：大きい値ほど始点の影響が強くなるよう変換
            # 1.0/image_strengthを使用し、最小値を0.01に制限
            strength_value = max(0.01, 1.0 / image_strength)
            print(translate('Image影響度: UI値={0:.2f}（{1:.0f}%）→計算値={2:.4f}（値が小さいほど始点の影響が強い）').format(
                image_strength, image_strength * 100, strength_value))

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
                initial_latent=current_latent,  # 開始潜在空間を設定
                strength=strength_value,        # 計算した影響度を使用
                callback=callback,
            )

            # if is_last_section:
            #     generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            # 後方にフレームを追加
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram:
                # 減圧時に使用するGPUメモリ値も明示的に浮動小数点に設定
                preserved_memory_offload = 8.0  # こちらは固定値のまま
                print(translate('Offloading transformer with memory preservation: {0} GB').format(preserved_memory_offload))
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=preserved_memory_offload)
                load_model_as_complete(vae, target_device=gpu)

            # 最新フレームは末尾から切り出し
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            # COMMENTED OUT: VAEデコード前のメモリクリア（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()
            #     print(translate("VAEデコード前メモリ: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                # latent_window_sizeが4.5の場合は特別に5を使用
                if latent_window_size == 4.5:
                    section_latent_frames = 11 if is_last_section else 10  # 5 * 2 + 1 = 11, 5 * 2 = 10
                    overlapped_frames = 17  # 5 * 4 - 3 = 17
                else:
                    # +1は逆方向生成時の start_latent 分なのでカット
                    section_latent_frames = int(latent_window_size * 2) if is_last_section else int(latent_window_size * 2)
                    overlapped_frames = int(latent_window_size * 4 - 3)

                # F1モードでは最新フレームは末尾にあるため、後方のセクションを取得
                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()

                # 引数の順序を修正 - history_pixelsが先、新しいcurrent_pixelsが後
                if history_pixels is None:
                    history_pixels = current_pixels
                else:
                    history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

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
                    section_metadata = {
                        PROMPT_KEY: prompt,  # メインプロンプト
                        SEED_KEY: seed,
                        SECTION_NUMBER_KEY: i_section
                    }

                    # セクション固有のプロンプトがあれば取得
                    if section_map and i_section in section_map:
                        _, section_prompt = section_map[i_section]
                        if section_prompt and section_prompt.strip():
                            section_metadata[SECTION_PROMPT_KEY] = section_prompt

                    # 画像の保存とメタデータの埋め込み
                    if is_first_section:
                        frame_path = os.path.join(outputs_folder, f'{job_id}_{i_section}_end.png')
                        Image.fromarray(last_frame).save(frame_path)
                        embed_metadata_to_png(frame_path, section_metadata)
                    else:
                        frame_path = os.path.join(outputs_folder, f'{job_id}_{i_section}.png')
                        Image.fromarray(last_frame).save(frame_path)
                        embed_metadata_to_png(frame_path, section_metadata)

                    print(translate("セクション{0}のフレーム画像をメタデータ付きで保存しました").format(i_section))
                except Exception as e:
                    print(translate("セクション{0}最終フレーム画像保存時にエラー: {1}").format(i_section, e))

            # 全フレーム画像保存機能
            # 「全フレーム画像保存」または「最終セクションのみ全フレーム画像保存かつ最終セクション」が有効な場合
            # 最終セクションかどうかの判定をtotal_sectionsから正確に取得
            is_last_section = i_section == total_sections - 1
            
            # save_latent_frames と save_last_section_frames の値をcopy
            # ループ内の変数を変更してもグローバルな値は変わらないため
            # 注意：既にここに来る前に万が一の文字列→ブール変換処理が済んでいるはず
            
            # 値のコピーではなく、明示的に新しい変数に適切な値を設定
            # BooleanかStringかの型変換ミスを防ぐ
            is_save_all_frames = bool(save_latent_frames)
            is_save_last_frame_only = bool(save_last_section_frames)
            
            if is_save_all_frames:
                should_save_frames = True
            elif is_save_last_frame_only and is_last_section:
                should_save_frames = True
            else:
                should_save_frames = False
            
            if should_save_frames:
                try:
                    # source_pixelsは、このセクションで使用するピクセルデータ
                    source_pixels = None
                    
                    # i_section=0の場合、current_pixelsが定義される前に参照されるためエラーとなる
                    # history_pixelsを優先して使用するよう処理順序を変更
                    if history_pixels is not None:
                        source_pixels = history_pixels
                        print(translate("フレーム画像保存: history_pixelsを使用します"))
                    elif 'current_pixels' in locals() and current_pixels is not None:
                        source_pixels = current_pixels
                        print(translate("フレーム画像保存: current_pixelsを使用します"))
                    else:
                        print(translate("フレーム画像保存: 有効なピクセルデータがありません"))
                        return
                        
                    # フレーム数（1秒モードでは9フレーム、0.5秒モードでは5フレーム）
                    latent_frame_count = source_pixels.shape[2]
                    
                    # 保存モードに応じたメッセージを表示
                    # グローバル変数ではなく、ローカルのcopyを使用
                    if is_save_all_frames:
                        print(translate("全フレーム画像保存: セクション{0}の{1}フレームを保存します").format(i_section, latent_frame_count))
                    elif is_save_last_frame_only and is_last_section:
                        # 強調して最終セクションであることを表示
                        print(translate("最終セクションのみ全フレーム画像保存: セクション{0}/{1}の{2}フレームを保存します (最終セクション)").format(
                            i_section, total_sections-1, latent_frame_count))
                    else:
                        print(translate("フレーム画像保存: セクション{0}の{1}フレームを保存します").format(i_section, latent_frame_count))
                    
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
                        
                        # 画像の保存とメタデータの埋め込み
                        frame_path = os.path.join(frames_folder, f'frame_{frame_idx:03d}.png')
                        Image.fromarray(frame).save(frame_path)
                        embed_metadata_to_png(frame_path, frame_metadata)
                    
                    # 保存モードに応じたメッセージを表示
                    # グローバル変数ではなく、ローカルのcopyを使用
                    if is_save_all_frames:
                        print(translate("全フレーム画像保存: セクション{0}の{1}個のフレーム画像を保存しました: {2}").format(
                            i_section, latent_frame_count, frames_folder))
                    elif is_save_last_frame_only and is_last_section:
                        print(translate("最終セクションのみ全フレーム画像保存: セクション{0}/{1}の{2}個のフレーム画像を保存しました (最終セクション): {3}").format(
                            i_section, total_sections-1, latent_frame_count, frames_folder))
                    else:
                        print(translate("セクション{0}の{1}個のフレーム画像を保存しました: {2}").format(
                            i_section, latent_frame_count, frames_folder))
                except Exception as e:
                    print(translate("セクション{0}のフレーム画像保存中にエラー: {1}").format(i_section, e))
                    traceback.print_exc()

            if not high_vram:
                unload_complete_models()

            # MP4ファイル名はendframe_ichiの命名規則に合わせる
            # バッチ番号はファイル名に明示的に含めない
            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            # もしhistory_pixelsの値が不適切な範囲にある場合、範囲を修正
            if history_pixels.min() < -1.0 or history_pixels.max() > 1.0:
                history_pixels = torch.clamp(history_pixels, -1.0, 1.0)

            # MP4を保存
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(translate('Decoded. Current latent shape {0}; pixel shape {1}').format(real_history_latents.shape, history_pixels.shape))

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

                        if uploaded_tensor.shape[3] != real_history_latents.shape[3] or uploaded_tensor.shape[4] != real_history_latents.shape[4]:
                            print(translate("警告: テンソルサイズが異なります: アップロード={0}, 現在の生成={1}").format(uploaded_tensor.shape, real_history_latents.shape))
                            print(translate("テンソルサイズの不一致のため、前方結合をスキップします"))
                            stream.output_queue.push(('progress', (None, translate("テンソルサイズの不一致のため、前方結合をスキップしました"), make_progress_bar_html(85, translate('互換性エラー')))))
                        else:
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
                                print(translate("GPUメモリ確保状態: {memory:.2f}GB").format(memory=torch.cuda.memory_allocated()/1024**3))

                            # VAEをGPUに移動
                            if not high_vram and vae.device != torch.device('cuda'):
                                print(translate("VAEをGPUに移動: {0} → cuda").format(vae.device))
                                vae.to('cuda')

                            # 各チャンクを処理
                            # チャンクサイズを設定(各セクションと同等のサイズにする)
                            chunk_size = min(5, uploaded_frames)  # 最大チャンクサイズを5フレームに設定（メモリ使用量を減らすため）

                            # チャンク数を計算
                            num_chunks = (uploaded_frames + chunk_size - 1) // chunk_size

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
                                    print(translate("チャンク{0}処理前のGPUメモリ: {1:.2f}GB/{2:.2f}GB").format(chunk_idx+1, torch.cuda.memory_allocated()/1024**3, torch.cuda.get_device_properties(0).total_memory/1024**3))
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

                                    # 明示的にデバイスを合わせる
                                    if current_chunk.device != vae.device:
                                        print(translate("  - デバイスをVAEと同じに変更: {0} → {1}").format(current_chunk.device, vae.device))
                                        current_chunk = current_chunk.to(vae.device)

                                    # 型を明示的に合わせる
                                    if current_chunk.dtype != torch.float16:
                                        print(translate("  - データ型をfloat16に変更: {0} → torch.float16").format(current_chunk.dtype))
                                        current_chunk = current_chunk.to(dtype=torch.float16)

                                    # VAEデコード処理
                                    chunk_pixels = vae_decode(current_chunk, vae).cpu()
                                    print(translate("チャンク{0}のVAEデコード完了 (フレーム数: {1})").format(chunk_idx+1, chunk_frames))

                                    # メモリ使用量を出力
                                    if torch.cuda.is_available():
                                        print(translate("チャンク{0}デコード後のGPUメモリ: {1:.2f}GB").format(chunk_idx+1, torch.cuda.memory_allocated()/1024**3))

                                    # 結合する
                                    if combined_history_pixels is None:
                                        # 初回のチャンクの場合はそのまま設定
                                        combined_history_pixels = chunk_pixels
                                    else:
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
                                    print(translate("エラー情報:"))
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
                                print(translate("チャンク処理後のGPUメモリ確保状態: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))

                            # 全チャンクの処理が完了したら、最終的な結合動画を保存
                            if combined_history_pixels is not None:
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

                # 処理終了時に通知（アラーム設定が有効な場合のみ）
                # アラーム判定を行う（Gradioコンポーネントから正しく値を取得）
                should_play_alarm = False  # デフォルトはオフ
                
                # Gradioオブジェクトからの値取得
                if isinstance(alarm_on_completion, bool):
                    should_play_alarm = alarm_on_completion
                elif hasattr(alarm_on_completion, 'value') and isinstance(alarm_on_completion.value, bool):
                    should_play_alarm = alarm_on_completion.value
                else:
                    # UIからの値取得に失敗した場合は設定ファイルから取得
                    try:
                        from eichi_utils.settings_manager import load_app_settings_f1
                        app_settings = load_app_settings_f1()
                        if app_settings and "alarm_on_completion" in app_settings:
                            should_play_alarm = app_settings["alarm_on_completion"]
                    except:
                        # 設定ファイルからも取得できない場合はデフォルトでオフ
                        should_play_alarm = False
                
                if should_play_alarm:
                    if HAS_WINSOUND:
                        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                    else:
                        print(translate("処理が完了しました"))  # Linuxでの代替通知

                # メモリ解放を明示的に実行
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    print(translate("処理完了後のメモリクリア: {memory:.2f}GB/{total_memory:.2f}GB").format(memory=torch.cuda.memory_allocated()/1024**3, total_memory=torch.cuda.get_device_properties(0).total_memory/1024**3))

                # テンソルデータの保存処理
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
                                    print(translate("結合テンソル情報: 合計{0}フレーム ({1}+{2}), {3:.2f} MB").format(frames, tensor_to_save.shape[2], uploaded_tensor.shape[2], size))
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
                print(translate("全体の処理時間: {0}").format(time_str))

                # 完了メッセージの設定（結合有無によって変更）
                if combined_output_filename is not None:
                    # テンソル結合が成功した場合のメッセージ
                    combined_filename_only = os.path.basename(combined_output_filename)
                    completion_message = translate("すべてのセクション({sections}/{total_sections})が完了しました。テンソルデータとの後方結合も完了しました。結合ファイル名: {filename}\n全体の処理時間: {time}").format(sections=sections, total_sections=total_sections, filename=combined_filename_only, time=time_str)
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
                                print(translate("中間ファイル: {0}").format(file))
                            except Exception as e:
                                print(translate("ファイル削除時のエラー {0}: {1}").format(file, e))

                    if deleted_count > 0:
                        print(translate("{0}個の中間ファイルを削除しました。最終ファイルは保存されています: {1}").format(deleted_count, final_video_name))
                        final_message = translate("中間ファイルを削除しました。最終動画と結合動画は保存されています。")
                        stream.output_queue.push(('progress', (None, final_message, make_progress_bar_html(100, translate('処理完了')))))

                break
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
    """入力画像または画面に表示されている最後のキーフレーム画像のいずれかが有効かを確認する - SLIDER PRIORITIZED"""
    # 入力画像をチェック
    if input_image is not None:
        return True, ""

    # 現在の設定から表示すべきセクション数を計算
    total_display_sections = None
    if frame_size_radio is not None:
        try:
            # Try to get slider value from global components first
            seconds = None
            
            # Access the slider value directly from global components
            global current_ui_components
            if 'total_second_length' in current_ui_components:
                slider_component = current_ui_components['total_second_length']
                if hasattr(slider_component, 'value'):
                    seconds = slider_component.value
                    print(translate("🎯 validate_images using SLIDER value: {0}s").format(seconds))
            
            # Fallback to radio if slider not available
            if seconds is None and length_radio is not None:
                seconds = get_video_seconds(length_radio.value)
                print(translate("🔄 validate_images fallback to RADIO value: {0}s").format(seconds))
            
            # Default fallback
            if seconds is None:
                seconds = 1
                print(translate("⚠️ validate_images using DEFAULT value: {0}s").format(seconds))

            # フレームサイズ設定からlatent_window_sizeを計算
            latent_window_size = 4.5 if frame_size_radio.value == translate("0.5秒 (17フレーム)") else 9
            frame_count = latent_window_size * 4 - 3

            # セクション数を計算
            total_frames = int(seconds * 30)
            total_display_sections = int(max(round(total_frames / frame_count), 1))
            
        except Exception as e:
            print(translate("セクション数計算エラー: {0}").format(e))

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

        # 有効なセクションがあれば、最大の番号（最後のセクション）を探す
        if valid_sections:
            # 番号でソート
            valid_sections.sort(key=lambda x: x[0])
            # 最後のセクションを取得
            last_visible_section_num, last_visible_section_image = valid_sections[-1]

    # 最後のキーフレーム画像があればOK
    if last_visible_section_image is not None:
        return True, ""

    # どちらの画像もない場合はエラー
    error_html = f"""
    <div style="padding: 15px; border-radius: 10px; background-color: #ffebee; border: 1px solid #f44336; margin: 10px 0;">
        <h3 style="color: #d32f2f; margin: 0 0 10px 0;">{translate('画像が選択されていません')}</h3>
        <p>{translate('生成を開始する前に「Image」欄または表示されている最後のキーフレーム画像に画像をアップロードしてください。これは叡智の始発点となる重要な画像です。')}</p>
    </div>
    """
    error_bar = make_progress_bar_html(100, translate('画像がありません'))
    return False, error_html + error_bar

def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_random_seed, mp4_crf=16, all_padding_value=1.0, image_strength=1.0, frame_size_setting="1秒 (33フレーム)", keep_section_videos=False, lora_files=None, lora_files2=None, lora_files3=None, lora_scales_text="0.8,0.8,0.8", output_dir=None, save_section_frames=False, use_all_padding=False, use_lora=False, lora_mode=None, lora_dropdown1=None, lora_dropdown2=None, lora_dropdown3=None, save_tensor_data=False, section_settings=None, tensor_data_input=None, fp8_optimization=False, resolution=640, batch_count=1, frame_save_mode=translate("保存しない"), use_queue=False, prompt_queue_file=None, save_settings_on_start=False, alarm_on_completion=False):
    # 引数の型確認
    # 異常な型の修正 (boolなど)
    if section_settings is not None and not isinstance(section_settings, list):
        print(translate("section_settingsがリスト型ではありません：{0}. 初期化します。").format(type(section_settings).__name__))
        section_settings = [[None, None, ""] for _ in range(50)]
    # メイン生成処理
    global stream
    global batch_stopped
    global queue_enabled, queue_type, prompt_queue_file_path, image_queue_files

    # バッチ処理開始時に停止フラグをリセット
    batch_stopped = False


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
    batch_count = max(1, min(int(batch_count), 100))  # 1〜100の間に制限


    # Check if we're in queue processing mode and should track batch progress
    is_queue_processing = (current_processing_config_name is not None)
    if is_queue_processing:
        print(translate("📊 Queue processing detected - initializing batch progress tracking for {0} batches").format(batch_count))
        update_batch_progress(0, batch_count)  # Initialize progress


    print(translate("バッチ処理回数: {0}回").format(batch_count))

    # 解像度を安全な値に丸めてログ表示
    from diffusers_helper.bucket_tools import SAFE_RESOLUTIONS

    # 解像度値を表示
    print(translate("UIから受け取った解像度値: {0}（型: {1}）").format(resolution, type(resolution).__name__))

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

    # F1モードでは常に通常のみ
    mode_name = translate("通常モード")

    print(translate("==== 動画生成開始 ====="))
    print(translate("生成モード: {0}").format(mode_name))
    print(translate("動画長: {0}秒").format(total_second_length))
    
    # 自動保存機能
    if save_settings_on_start:
        try:
            from eichi_utils.settings_manager import save_app_settings_f1
            current_settings = {
                "resolution": resolution,
                "mp4_crf": mp4_crf,
                "steps": steps,
                "cfg": cfg,
                "use_teacache": use_teacache,
                "gpu_memory_preservation": gpu_memory_preservation,
                "gs": gs,
                "image_strength": image_strength,
                "keep_section_videos": keep_section_videos,
                "save_section_frames": save_section_frames,
                "save_tensor_data": save_tensor_data,
                "frame_save_mode": frame_save_mode,
                "save_settings_on_start": save_settings_on_start,
                "alarm_on_completion": alarm_on_completion
            }
            save_app_settings_f1(current_settings)
            print(translate("自動保存が完了しました"))
        except Exception as e:
            print(translate("自動保存中にエラーが発生しました: {0}").format(str(e)))
    print(translate("フレームサイズ: {0}").format(frame_size_setting))
    print(translate("生成セクション数: {0}回").format(total_latent_sections))
    print(translate("サンプリングステップ数: {0}").format(steps))
    print(translate("TeaCache使用: {0}").format(use_teacache))
    # TeaCache使用の直後にSEED値の情報を表示
    print(translate("使用SEED値: {0}").format(seed))
    print(translate("LoRA使用: {0}").format(use_lora))

    # FP8最適化設定のログ出力
    print(translate("FP8最適化: {0}").format(fp8_optimization))

    # オールパディング設定のログ出力（F1モードでは常に無効）
    print(translate("オールパディング: F1モードでは無効化されています"))

    # LoRA情報のログ出力
    if use_lora and has_lora_support:
        all_lora_files = []
        lora_paths = []
        
        # LoRAの読み込み方式に応じて処理を分岐
        if lora_mode == translate("ディレクトリから選択"):
            # ディレクトリから選択モードの場合
            print(translate("ディレクトリから選択モードでLoRAを処理"))
            lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
            
            # 特にlora_dropdown2の値が問題になることが多いので詳細ログ
            if isinstance(lora_dropdown2, int) and lora_dropdown2 == 0:
                
                # 特別処理: 数値の0はインデックスとして解釈されている可能性がある
                # 選択肢リストの0番目（なし）として扱う
                dropdown_direct_value = translate("なし")
                        
                # もし既に処理済みの文字列値が別にあればそちらを優先
                if isinstance(lora_dropdown2, str) and lora_dropdown2 != "0" and lora_dropdown2 != translate("なし"):
                    dropdown_direct_value = lora_dropdown2
            
            # 各ドロップダウンの値を処理
            for dropdown, label in zip([lora_dropdown1, lora_dropdown2, lora_dropdown3], ["LoRA1", "LoRA2", "LoRA3"]):
                if dropdown is not None and dropdown != translate("なし") and dropdown != 0:
                    # 選択あり
                    file_path = os.path.join(lora_dir, dropdown)
                    if os.path.exists(file_path):
                        lora_paths.append(file_path)
                        print(translate("{0}選択: {1}").format(label, dropdown))
                    else:
                        print(translate("選択された{0}ファイルが見つかりません: {1}").format(label, file_path))
        else:
            # ファイルアップロードモードの場合
            print(translate("ファイルアップロードモードでLoRAを処理"))
            
            # 1つ目のLoRAファイルを処理
            if lora_files is not None:
                if isinstance(lora_files, list):
                    all_lora_files.extend(lora_files)
                else:
                    all_lora_files.append(lora_files)
                    
            # 2つ目のLoRAファイルを処理
            if lora_files2 is not None:
                if isinstance(lora_files2, list):
                    all_lora_files.extend(lora_files2)
                else:
                    all_lora_files.append(lora_files2)
            
            # 3つ目のLoRAファイルを処理（F1版でも対応）
            if lora_files3 is not None:
                if isinstance(lora_files3, list):
                    all_lora_files.extend(lora_files3)
                else:
                    all_lora_files.append(lora_files3)
            
            # アップロードファイルからパスリストを生成
            for lora_file in all_lora_files:
                if hasattr(lora_file, 'name'):
                    lora_paths.append(lora_file.name)
        
        # スケール値を解析
        try:
            scales = [float(s.strip()) for s in lora_scales_text.split(',')]
        except:
            # 解析エラーの場合はデフォルト値を使用
            scales = [0.8] * len(lora_paths)
            
        # スケール値の数を調整
        if len(scales) < len(lora_paths):
            scales.extend([0.8] * (len(lora_paths) - len(scales)))
        elif len(scales) > len(lora_paths):
            scales = scales[:len(lora_paths)]
            
        # LoRAファイル情報を出力
        if len(lora_paths) == 1:
            # 単一ファイル
            print(translate("LoRAファイル: {0}").format(os.path.basename(lora_paths[0])))
            print(translate("LoRA適用強度: {0}").format(scales[0]))
        elif len(lora_paths) > 1:
            # 複数ファイル
            print(translate("LoRAファイル (複数):"))
            for i, path in enumerate(lora_paths):
                print(translate("   - {0} (スケール: {1})").format(os.path.basename(path), scales[i] if i < len(scales) else 0.8))
        else:
            # LoRAファイルなし
            print(translate("LoRA: 使用しない"))

    # セクションごとのキーフレーム画像の使用状況をログに出力
    valid_sections = []
    if section_settings is not None:
        # リストでない場合は空のリストとして扱う
        if not isinstance(section_settings, list):
            print(translate("section_settingsがリスト型ではありません。空のリストとして扱います。"))
            section_settings = []

        for i, sec_data in enumerate(section_settings):
            if sec_data and isinstance(sec_data, list) and len(sec_data) > 1 and sec_data[1] is not None:  # 画像が設定されている場合
                valid_sections.append(sec_data[0])

    if valid_sections:
        print(translate("使用するキーフレーム画像: セクション{0}").format(', '.join(map(str, valid_sections))))
    else:
        print(translate("キーフレーム画像: デフォルト設定のみ使用"))

    print("=============================")

    # バッチ処理の全体停止用フラグ
    batch_stopped = False

    # 元のシード値を保存（バッチ処理用）
    original_seed = seed
    
    # ランダムシード生成を文字列型も含めて判定
    use_random = False
    if isinstance(use_random_seed, bool):
        use_random = use_random_seed
    elif isinstance(use_random_seed, str):
        use_random = use_random_seed.lower() in ["true", "yes", "1", "on"]
    
    if use_random:
        # ランダムシード設定前の値を保存
        previous_seed = seed
        # 特定の範囲内で新しいシード値を生成
        seed = random.randint(0, 2**32 - 1)
        # ユーザーにわかりやすいメッセージを表示
        print(translate("ランダムシード機能が有効なため、指定されたSEED値 {0} の代わりに新しいSEED値 {1} を使用します。").format(previous_seed, seed))
        # UIのseed欄もランダム値で更新
        yield gr.skip(), None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update(value=seed)
        # ランダムシードの場合は最初の値を更新
        original_seed = seed
    else:
        print(translate("指定されたSEED値 {0} を使用します。").format(seed))
        yield gr.skip(), None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update()

    stream = AsyncStream()

    # stream作成後、バッチ処理前もう一度フラグを確認
    if batch_stopped:
        print(translate("バッチ処理が中断されました（バッチ開始前）"))
        yield (
            gr.skip(),
            gr.update(visible=False),
            translate("バッチ処理が中断されました"),
            '',
            gr.update(interactive=True),
            gr.update(interactive=False, value=translate("End Generation")),
            gr.update()
        )
        return

    # バッチ処理ループの開始
    if queue_enabled:
        if queue_type == "image":
            print(translate("バッチ処理情報: 合計{0}回").format(batch_count))
            print(translate("イメージキュー: 有効, 入力画像1枚 + 画像ファイル{0}枚").format(len(image_queue_files)))
            print(translate("処理順序: 1回目=入力画像, 2回目以降=入力フォルダの画像ファイル"))
            # バッチ処理を強調表示
            for i in range(batch_count):
                if i == 0:
                    img_src = "入力画像"
                else:
                    img_idx = i - 1
                    if img_idx < len(image_queue_files):
                        img_src = os.path.basename(image_queue_files[img_idx])
                    else:
                        img_src = "入力画像（キュー画像不足）"
                print(translate("   └ バッチ{0}: {1}").format(i+1, img_src))
        else:
            queue_lines_count = 0
            if prompt_queue_file_path and os.path.exists(prompt_queue_file_path):
                try:
                    with open(prompt_queue_file_path, 'r', encoding='utf-8') as f:
                        queue_lines = [line.strip() for line in f.readlines() if line.strip()]
                        queue_lines_count = len(queue_lines)
                        # 各プロンプトのプレビューを表示
                        for i in range(min(batch_count, queue_lines_count)):
                            prompt_preview = queue_lines[i][:50] + "..." if len(queue_lines[i]) > 50 else queue_lines[i]
                            print(translate("   └ バッチ{0}: {1}").format(i+1, prompt_preview))
                except:
                    pass
            print(translate("バッチ処理情報: 合計{0}回").format(batch_count))
            print(translate("プロンプトキュー: 有効, プロンプト行数={0}行").format(queue_lines_count))
    else:
        print(translate("バッチ処理情報: 合計{0}回").format(batch_count))
        print(translate("キュー機能: 無効"))
    for batch_index in range(batch_count):
        # 停止フラグが設定されている場合は全バッチ処理を中止
        if batch_stopped:
            print(translate("バッチ処理がユーザーによって中止されました"))
            yield (
                gr.skip(),
                gr.update(visible=False),
                translate("バッチ処理が中止されました。"),
                '',
                gr.update(interactive=True),
                gr.update(interactive=False, value=translate("End Generation")),
                gr.update()
            )
            break

        # ADDED: Update batch progress for queue processing
        if is_queue_processing:
            current_batch_num = batch_index + 1
            update_batch_progress(current_batch_num, batch_count)


        # 現在のバッチ番号を表示
        if batch_count > 1:
            batch_info = translate("バッチ処理: {0}/{1}").format(batch_index + 1, batch_count)
            print(f"{batch_info}")
            # UIにもバッチ情報を表示
            yield gr.skip(), gr.update(visible=False), batch_info, "", gr.update(interactive=False), gr.update(interactive=True), gr.update()


        # 今回処理用のプロンプトとイメージを取得（キュー機能対応）
        current_prompt = prompt
        current_image = input_image
        
        # イメージキューでカスタムプロンプトを使用しているかどうかを確認（ログ出力用）
        using_custom_prompt = False
        if queue_enabled and queue_type == "image" and batch_index > 0:
            if batch_index - 1 < len(image_queue_files):
                queue_img_path = image_queue_files[batch_index - 1]
                img_basename = os.path.splitext(queue_img_path)[0]
                txt_path = f"{img_basename}.txt"
                if os.path.exists(txt_path):
                    img_name = os.path.basename(queue_img_path)
                    using_custom_prompt = True
                    print(translate("セクション{0}はイメージキュー画像「{1}」の専用プロンプトを使用します").format("全て", img_name))

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
                                print(translate("プロンプトキュー実行中: バッチ {0}/{1}").format(batch_index+1, batch_count))
                                print(translate("  └ プロンプト: 「{0}...」").format(current_prompt[:50]))
                            else:
                                print(translate("プロンプトキュー実行中: バッチ {0}/{1} はプロンプト行数を超えているため元のプロンプトを使用").format(batch_index+1, batch_count))
                    except Exception as e:
                        print(translate("プロンプトキューファイル読み込みエラー: {0}").format(str(e)))

            elif queue_type == "image" and len(image_queue_files) > 0:
                # イメージキューの処理
                # 最初のバッチは入力画像を使用
                if batch_index == 0:
                    print(translate("イメージキュー実行中: バッチ {0}/{1} は入力画像を使用").format(batch_index+1, batch_count))
                elif batch_index > 0:
                    # 2回目以降はイメージキューの画像を順番に使用
                    image_index = batch_index - 1  # 0回目（入力画像）の分を引く

                    if image_index < len(image_queue_files):
                        current_image = image_queue_files[image_index]
                        image_filename = os.path.basename(current_image)
                        print(translate("イメージキュー実行中: バッチ {0}/{1} の画像「{2}」").format(batch_index+1, batch_count, image_filename))
                        print(translate("  └ 画像ファイルパス: {0}").format(current_image))
                        
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
                        print(translate("イメージキュー実行中: バッチ {0}/{1} は画像数を超えているため入力画像を使用").format(batch_index+1, batch_count))

        # バッチインデックスに応じてSEED値を設定
        current_seed = original_seed + batch_index
        if batch_count > 1:
            print(translate("初期SEED値: {0}").format(current_seed))
        # 現在のバッチ用のシードを設定
        seed = current_seed

        # もう一度停止フラグを確認 - worker処理実行前
        if batch_stopped:
            print(translate("バッチ処理が中断されました。worker関数の実行をキャンセルします。"))
            # 中断メッセージをUIに表示
            yield (gr.skip(),
                   gr.update(visible=False),
                   translate("バッチ処理が中断されました（{0}/{1}）").format(batch_index, batch_count),
                   '',
                   gr.update(interactive=True),
                   gr.update(interactive=False, value=translate("End Generation")),
                   gr.update())
            break

        # GPUメモリの設定値を出力し、正しい型に変換
        gpu_memory_value = float(gpu_memory_preservation) if gpu_memory_preservation is not None else 6.0
        print(translate('Using GPU memory preservation setting: {0} GB').format(gpu_memory_value))

        # 出力フォルダが空の場合はデフォルト値を使用
        if not output_dir or not output_dir.strip():
            output_dir = "outputs"
        print(translate('Output directory: {0}').format(output_dir))

        # Gradioオブジェクトから実際の値を取得
        if hasattr(frame_save_mode, 'value'):
            frame_save_mode_actual = frame_save_mode.value
        else:
            frame_save_mode_actual = frame_save_mode
            
        print(translate("現在のバッチ: {0}/{1}, 画像: {2}").format(
            batch_index + 1,
            batch_count,
            os.path.basename(current_image) if isinstance(current_image, str) else "入力画像"
        ))

        # キュー機能使用時の現在のプロンプトとイメージでワーカーを実行
        async_run(
            worker,
            current_image,  # キュー機能で選択された画像
            current_prompt,  # キュー機能で選択されたプロンプト
            n_prompt,
            seed,
            total_second_length,
            latent_window_size,
            steps,
            cfg,
            gs,
            rs,
            gpu_memory_value,
            use_teacache,
            mp4_crf,
            all_padding_value,
            image_strength,
            keep_section_videos,
            lora_files,
            lora_files2,
            lora_files3,  # 追加: lora_files3
            lora_scales_text,
            output_dir,
            save_section_frames,
            use_all_padding,
            use_lora,
            lora_mode,  # 追加: lora_mode
            lora_dropdown1,  # 追加: lora_dropdown1
            lora_dropdown2,  # 追加: lora_dropdown2
            lora_dropdown3,  # 追加: lora_dropdown3
            save_tensor_data,
            tensor_data_input,
            fp8_optimization,
            resolution,
            batch_index,
            frame_save_mode_actual
        )

        # 現在のバッチの出力ファイル名
        batch_output_filename = None

        # 現在のバッチの処理結果を取得
        while True:
            flag, data = stream.output_queue.next()

            if flag == 'file':
                batch_output_filename = data
                # より明確な更新方法を使用し、preview_imageを明示的にクリア
                yield (
                    batch_output_filename if batch_output_filename is not None else gr.skip(), 
                    gr.update(value=None, visible=False), 
                    gr.update(), 
                    gr.update(), 
                    gr.update(interactive=False), 
                    gr.update(interactive=True), 
                    gr.update(),
                )

            if flag == 'progress':
                preview, desc, html = data
                # バッチ処理中は現在のバッチ情報を追加
                if batch_count > 1:
                    batch_info = translate("バッチ処理: {0}/{1} - ").format(batch_index + 1, batch_count)
                    desc = batch_info + desc
                # preview_imageを明示的に設定
                yield gr.skip(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), gr.update()

            if flag == 'end':


                # ADDED: Log batch completion for queue processing
                if is_queue_processing:
                    print(translate("📊 Batch {0}/{1} completed for queue processing").format(batch_index + 1, batch_count))

                # このバッチの処理が終了
                if batch_index == batch_count - 1 or batch_stopped:
                    # 最終バッチの場合は処理完了を通知
                    completion_message = ""
                    if batch_stopped:
                        completion_message = translate("バッチ処理が中止されました（{0}/{1}）").format(batch_index + 1, batch_count)
                    else:
                        completion_message = translate("バッチ処理が完了しました（{0}/{1}）").format(batch_count, batch_count)


                    # ADDED: Reset batch progress when all batches complete (for queue processing)
                    if is_queue_processing:
                        print(translate("📊 All batches completed - resetting batch progress"))
                        # Don't reset here, let the queue manager handle it when config is fully done

                    yield (
                        batch_output_filename if batch_output_filename is not None else gr.skip(),
                        gr.update(value=None, visible=False),
                        completion_message,
                        '',
                        gr.update(interactive=True),
                        gr.update(interactive=False, value=translate("End Generation")),
                        gr.update()
                    )
                    # 最後のバッチが終わったので終了
                    print(translate("バッチシーケンス完了: 全 {0} バッチの処理を終了").format(batch_count))
                else:
                    # 次のバッチに進むメッセージを表示
                    next_batch_message = translate("バッチ処理: {0}/{1} 完了、次のバッチに進みます...").format(batch_index + 1, batch_count)
                    print(translate("バッチ {0}/{1} 完了 - 次のバッチに進みます").format(batch_index + 1, batch_count))
                    yield (
                        batch_output_filename if batch_output_filename is not None else gr.skip(),
                        gr.update(value=None, visible=False),
                        next_batch_message,
                        '',
                        gr.update(interactive=False),
                        gr.update(interactive=True),
                        gr.update()
                    )
                    # バッチループの内側で使用される変数を次のバッチ用に更新する
                    continue_next_batch = True
                break

        # 最終的な出力ファイル名を更新
        output_filename = batch_output_filename

        # バッチ処理が停止されている場合はループを抜ける
        if batch_stopped:
            print(translate("バッチ処理ループを中断します"))
            break
  

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
    gr.HTML('<h1>FramePack<span class="title-suffix">-<s>eichi</s> F1</span></h1>')

    # 一番上の行に「生成モード、セクションフレームサイズ、オールパディング、動画長」を配置
    with gr.Row():
        with gr.Column(scale=1):
            # 生成モードのラジオボタン（F1モードでは通常のみ）
            mode_radio = gr.Radio(choices=[MODE_TYPE_NORMAL], value=MODE_TYPE_NORMAL, label=translate("生成モード"), info=translate("F1モードでは通常のみ利用可能"))
        with gr.Column(scale=1):
            # フレームサイズ切替用のUIコントロール（名前を「セクションフレームサイズ」に変更）
            frame_size_radio = gr.Radio(
                choices=[translate("1秒 (33フレーム)"), translate("0.5秒 (17フレーム)")],
                value=translate("1秒 (33フレーム)"),
                label=translate("セクションフレームサイズ"),
                info=translate("1秒 = 高品質・通常速度 / 0.5秒 = よりなめらかな動き（実験的機能）")
            )
        with gr.Column(scale=1):
            # オールパディング設定 (F1モードでは無効化)
            use_all_padding = gr.Checkbox(
                label=translate("オールパディング"),
                value=False,
                info=translate("F1モードでは使用できません。無印モードでのみ有効です。"),
                elem_id="all_padding_checkbox",
                interactive=False  # F1モードでは非活性化
            )
            all_padding_value = gr.Slider(
                label=translate("パディング値"),
                minimum=0.2,
                maximum=3,
                value=1,
                step=0.1,
                info=translate("F1モードでは使用できません"),
                visible=False,
                interactive=False  # F1モードでは非活性化
            )

            # オールパディングのチェックボックス状態に応じてスライダーの表示/非表示を切り替える
            def toggle_all_padding_visibility(use_all_padding):
                return gr.update(visible=use_all_padding)

            use_all_padding.change(
                fn=toggle_all_padding_visibility,
                inputs=[use_all_padding],
                outputs=[all_padding_value]
            )
        with gr.Column(scale=1):
            # 設定から動的に選択肢を生成
            length_radio = gr.Radio(choices=get_video_modes(), value=translate("1秒"), label=translate("動画長"), info=translate("動画の長さを設定。F1モードでは右下の「動画の総長（秒）」で20秒より長い動画長を設定可能です"))

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources=['upload', 'clipboard'], type="filepath", label="Image", height=320)

            # テンソルデータ設定をグループ化して灰色のタイトルバーに変更
            with gr.Group():
                gr.Markdown(f"### " + translate("テンソルデータ設定"))

                # テンソルデータ使用有無のチェックボックス
                use_tensor_data = gr.Checkbox(label=translate("テンソルデータを使用する"), value=False, info=translate("チェックをオンにするとテンソルデータをアップロードできます"))

                # テンソルデータ設定コンポーネント（初期状態では非表示）
                with gr.Group(visible=False) as tensor_data_group:
                    tensor_data_input = gr.File(
                        label=translate("テンソルデータアップロード (.safetensors) - 生成動画の後方(末尾)に結合されます"),
                        file_types=[".safetensors"]
                    )

                    gr.Markdown(translate("※ テンソルデータをアップロードすると通常の動画生成後に、その動画の後方（末尾）に結合されます。\n結合した動画は「元のファイル名_combined.mp4」として保存されます。\n※ テンソルデータの保存機能を有効にすると、生成とアップロードのテンソルを結合したデータも保存されます。\n※ テンソルデータの結合は別ツール `python eichi_utils/tensor_combiner.py --ui` でもできます。"))

                # チェックボックスの状態によってテンソルデータ設定の表示/非表示を切り替える関数
                def toggle_tensor_data_settings(use_tensor):
                    return gr.update(visible=use_tensor)

                # チェックボックスの変更イベントに関数を紐づけ
                use_tensor_data.change(
                    fn=toggle_tensor_data_settings,
                    inputs=[use_tensor_data],
                    outputs=[tensor_data_group]
                )

                # キュー機能のトグル関数
                def toggle_queue_settings(use_queue_val):
                    # グローバル変数を使用
                    global queue_enabled, queue_type

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

                    print(translate("トグル関数: チェックボックスの型={0}, 値={1}").format(type(use_queue_val).__name__, use_queue_val))
                    print(translate("キュー設定の表示状態を変更: {0} (グローバル変数に保存: queue_enabled={1})").format(is_enabled, queue_enabled))

                    # キュータイプに応じて適切なグループを表示/非表示
                    if is_enabled:
                        if queue_type == "prompt":
                            return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]
                        else:  # image
                            return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)]
                    else:
                        # チェックがオフなら全て非表示
                        return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]

                # キュータイプの切り替え関数
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

                # Config Queue Systemの表示/非表示を制御する関数
                def toggle_config_queue(use_config_queue_val):
                    """Config Queue Systemの表示/非表示を切り替える"""
                    is_enabled = bool(use_config_queue_val)
                    print(translate("Config Queue System 表示設定: {0}").format(is_enabled))
                    return gr.update(visible=is_enabled)

                # ファイルアップロード処理関数
                def handle_file_upload(file_obj):
                    global prompt_queue_file_path

                    if file_obj is not None:
                        print(translate("ファイルアップロード検出: 型={0}").format(type(file_obj).__name__))

                        if hasattr(file_obj, 'name'):
                            prompt_queue_file_path = file_obj.name
                            print(translate("アップロードファイルパス保存: {0}").format(prompt_queue_file_path))
                        else:
                            prompt_queue_file_path = file_obj
                            print(translate("アップロードファイルデータ保存: {0}").format(file_obj))
                    else:
                        prompt_queue_file_path = None
                        print(translate("ファイルアップロード解除"))

                    return file_obj

                # 入力フォルダ名変更ハンドラ（フォルダ作成を行わない設計）
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

                # 入力フォルダを開くボタンハンドラ（設定保存とフォルダ作成を行う）
                def open_input_folder():
                    """入力フォルダを開く処理（保存も実行）"""
                    global input_folder_name_value

                    # 設定を保存
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

            # テンソルデータ設定の下に解像度スライダーとバッチ処理回数を追加
            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=2):
                        resolution = gr.Dropdown(
                            label=translate("解像度"),
                            choices=[512, 640, 768, 960, 1080],
                            value=saved_app_settings.get("resolution", 640) if saved_app_settings else 640,
                            info=translate("出力動画の基準解像度。640推奨。960/1080は高負荷・高メモリ消費"),
                            elem_classes="saveable-setting"
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

                # Config Queue System制御用のチェックボックス
                use_config_queue = gr.Checkbox(
                    label=translate("Config Queue System を使用（実験的機能）"),
                    value=False,
                    info=translate("チェックをオンにすると新しいConfig Queue Systemが利用できます。")
                )

                # Config Queue Systemをグループで囲んで表示制御
                with gr.Group(visible=False) as config_queue_group:
                    config_queue_components = create_enhanced_config_queue_ui()
                    queue_start_button = config_queue_components['enhanced_start_queue_btn']  

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
                    gr.Markdown(translate("※ 1回目はImage画像を使用し、2回目以降は入力フォルダの画像ファイルを名前順に使用します。\n※ 画像と同名のテキストファイル（例：image1.jpg → image1.txt）があれば、その内容を自動的にプロンプトとして使用します。\n※ バッチ回数が全画像数を超える場合、残りはImage画像で処理されます。\n※ バッチ処理回数が1でもキュー回数が優先されます。"))

                    # 入力フォルダ設定
                    with gr.Row():
                        input_folder_name = gr.Textbox(
                            label=translate("入力フォルダ名"),
                            value=input_folder_name_value,  # グローバル変数から値を取得
                            info=translate("画像ファイルを格納するフォルダ名")
                        )
                        open_input_folder_btn = gr.Button(value="📂 " + translate("保存及び入力フォルダを開く"), size="md")

                # チェックボックスの変更イベントに関数を紐づけ
                use_queue.change(
                    fn=toggle_queue_settings,
                    inputs=[use_queue],
                    outputs=[queue_type_selector, prompt_queue_group, image_queue_group]
                )

                # Config Queue Systemの表示切り替えイベント
                use_config_queue.change(
                    fn=toggle_config_queue,
                    inputs=[use_config_queue],
                    outputs=[config_queue_group]
                )

                # キュータイプの選択イベントに関数を紐づけ
                queue_type_selector.change(
                    fn=toggle_queue_type,
                    inputs=[queue_type_selector],
                    outputs=[prompt_queue_group, image_queue_group]
                )

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

            # 設定から最大キーフレーム数を取得
            max_keyframes = get_max_keyframes_count()

            # 現在の動画モードで必要なセクション数を取得する関数
            def get_current_sections_count():
                mode_value = length_radio.value
                if mode_value in VIDEO_MODE_SETTINGS:
                    # sections値をそのまま使用 - 注：これは0から始めた場合の最大値となる
                    return VIDEO_MODE_SETTINGS[mode_value]["sections"]
                return max_keyframes  # デフォルト値

            # 現在の必要セクション数を取得
            initial_sections_count = get_current_sections_count()
            # 簡略化セクション表示
            # セクションタイトルの関数は削除し、固定メッセージのみ表示

            # 埋め込みプロンプトおよびシードを複写するチェックボックス - 参照用に定義（表示はLoRA設定の下で行う）
            # グローバル変数として定義し、後で他の場所から参照できるようにする
            global copy_metadata
            copy_metadata = gr.Checkbox(
                label=translate("埋め込みプロンプトおよびシードを複写する"),
                value=False,
                info=translate("チェックをオンにすると、画像のメタデータからプロンプトとシードを自動的に取得します"),
                visible=False  # 元の位置では非表示
            )

            # F1モードではセクション設定は完全に削除
            # 隠しコンポーネント（互換性のため）
            section_image_inputs = []
            section_number_inputs = []
            section_prompt_inputs = []
            section_row_groups = []


            # メタデータ抽出関数を定義（後で登録する）
            def update_from_image_metadata(image_path, copy_enabled=False):
                """Imageアップロード時にメタデータを抽出してUIに反映する
                F1モードではキーフレームコピー機能を削除済みのため、単純化
                """
                # 複写機能が無効の場合は何もしない
                if not copy_enabled:
                    return [gr.update()] * 2

                if image_path is None:
                    return [gr.update()] * 2

                try:
                    # ファイルパスから直接メタデータを抽出
                    metadata = extract_metadata_from_png(image_path)

                    if not metadata:
                        print(translate("アップロードされた画像にメタデータが含まれていません"))
                        return [gr.update()] * 2

                    print(translate("画像からメタデータを抽出しました: {0}").format(metadata))

                    # プロンプトとSEEDをUIに反映
                    prompt_update = gr.update()
                    seed_update = gr.update()

                    if PROMPT_KEY in metadata and metadata[PROMPT_KEY]:
                        prompt_update = gr.update(value=metadata[PROMPT_KEY])

                    if SEED_KEY in metadata and metadata[SEED_KEY]:
                        # SEED値を整数に変換
                        try:
                            seed_value = int(metadata[SEED_KEY])
                            seed_update = gr.update(value=seed_value)
                        except (ValueError, TypeError):
                            print(translate("SEED値の変換エラー: {0}").format(metadata[SEED_KEY]))

                    return [prompt_update, seed_update]
                except Exception as e:
                    print(translate("メタデータ抽出処理中のエラー: {0}").format(e))
                    traceback.print_exc()
                    print(translate("メタデータ抽出エラー: {0}").format(e))
                    return [gr.update()] * 2

            # 注意: イベント登録は変数定義後に行うため、後で実行する
            # メタデータ抽出処理の登録は、promptとseed変数の定義後に移動します

            # LoRA設定グループを追加
            with gr.Group(visible=has_lora_support) as lora_settings_group:
                gr.Markdown(f"### " + translate("LoRA設定"))

                # LoRA使用有無のチェックボックス

                use_lora = gr.Checkbox(label=translate("LoRAを使用する"), value=False, info=translate("チェックをオンにするとLoRAを使用します（要16GB VRAM以上）"))

                def scan_lora_directory():
                    """./loraディレクトリからLoRAモデルファイルを検索する関数 - ENHANCED VERSION"""
                    lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora')
                    choices = []
                    
                    # ディレクトリが存在しない場合は作成
                    if not os.path.exists(lora_dir):
                        os.makedirs(lora_dir, exist_ok=True)
                        print(translate("LoRAディレクトリが存在しなかったため作成しました: {0}").format(lora_dir))
                    
                    # ディレクトリ内のファイルをリストアップ
                    try:
                        for filename in os.listdir(lora_dir):
                            if filename.endswith(('.safetensors', '.pt', '.bin')):
                                choices.append(filename)
                    except Exception as e:
                        print(translate("Error scanning LoRA directory: {0}").format(e))
                    
                    # 空の選択肢がある場合は"なし"を追加
                    choices = sorted(choices)
                    
                    # なしの選択肢を最初に追加
                    none_choice = translate("なし")
                    choices.insert(0, none_choice)
                    
                    # 重要: すべての選択肢が確実に文字列型であることを確認
                    for i, choice in enumerate(choices):
                        if not isinstance(choice, str):
                            choices[i] = str(choice)
                    
                    print(translate("🔍 Scanned LoRA directory: found {0} files").format(len(choices)-1))
                    return choices
                
                # LoRAの読み込み方式を選択するラジオボタン
                lora_mode = gr.Radio(
                    choices=[translate("ディレクトリから選択"), translate("ファイルアップロード")],
                    value=translate("ディレクトリから選択"),
                    label=translate("LoRA読み込み方式"),
                    visible=False  # 初期状態では非表示（toggle_lora_settingsで制御）
                )

                # ファイルアップロードグループ - 初期状態は非表示
                with gr.Group(visible=False) as lora_upload_group:
                    # メインのLoRAファイル
                    lora_files = gr.File(
                        label=translate("LoRAファイル (.safetensors, .pt, .bin)"),
                        file_types=[".safetensors", ".pt", ".bin"],
                        visible=True
                    )
                    # 追加のLoRAファイル1
                    lora_files2 = gr.File(
                        label=translate("LoRAファイル2 (.safetensors, .pt, .bin)"),
                        file_types=[".safetensors", ".pt", ".bin"],
                        visible=True
                    )
                    # 追加のLoRAファイル2（F1版でも3つ目を追加）
                    lora_files3 = gr.File(
                        label=translate("LoRAファイル3 (.safetensors, .pt, .bin)"),
                        file_types=[".safetensors", ".pt", ".bin"],
                        visible=True
                    )
                
                # ディレクトリ選択グループ - 初期状態は非表示
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

                # チェックボックスの状態によって他のLoRA設定の表示/非表示を切り替える関数
                def toggle_lora_settings(use_lora):
                    """
                    BASIC LORA TOGGLE: Original LoRA visibility control (simplified)
                    
                    This is the simplified version of the original inline function.
                    It only handles basic visibility without the complex config loading logic.
                    
                    PARAMETERS:
                    use_lora (bool): Whether LoRA is enabled
                    
                    RETURNS:
                    List of basic UI updates for visibility control:
                    [lora_mode_visible, lora_upload_group_visible, lora_dropdown_group_visible, lora_scales_visible]
                    """
                    if use_lora:
                        # LoRA使用時はデフォルトでディレクトリから選択モードを表示
                        choices = scan_lora_directory()
                        
                        # 選択肢の型チェックを追加
                        for i, choice in enumerate(choices):
                            if not isinstance(choice, str):
                                choices[i] = str(choice)
                        
                        # プリセットはディレクトリから選択モードの場合のみ表示
                        preset_visible = True  # デフォルトはディレクトリから選択なので表示
                        
                        # ドロップダウンが初期化時にも確実に更新されるようにする
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
                        
                        # 選択肢の型を明示的に確認＆変換
                        for i, choice in enumerate(choices):
                            if not isinstance(choice, str):
                                choices[i] = str(choice)
                        
                        # 最初の選択肢がちゃんと文字列になっているか再確認
                        first_choice = choices[0]
                        
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
                    
                    # すべての選択肢が確実に文字列型であることを確認
                    for i, choice in enumerate(choices):
                        if not isinstance(choice, str):
                            choices[i] = str(choice)
                    
                    # 各ドロップダウンを更新
                    return [
                        gr.update(choices=choices, value=choices[0]),  # lora_dropdown1
                        gr.update(choices=choices, value=choices[0]),  # lora_dropdown2
                        gr.update(choices=choices, value=choices[0]),  # lora_dropdown3
                    ]
                
                # 前回のLoRAモードを記憶するための変数
                previous_lora_mode = translate("ディレクトリから選択")  # デフォルトはディレクトリから選択
                
                # LoRAモードの変更を処理する関数
                def toggle_lora_mode_with_memory(mode_value):
                    # グローバル変数に選択を保存
                    global previous_lora_mode
                    previous_lora_mode = mode_value
                    
                    # 標準のtoggle_lora_mode関数を呼び出し
                    return toggle_lora_mode(mode_value)

                # チェックボックスの変更イベントにLoRA設定全体の表示/非表示を切り替える関数を紐づけ
                use_lora.change(
                    fn=toggle_lora_full_update,
                    inputs=[use_lora],
                    outputs=[lora_mode, lora_upload_group, lora_dropdown_group, lora_scales_text,
                             lora_dropdown1, lora_dropdown2, lora_dropdown3]
                )


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
                
                # UIロード時のLoRA初期化関数
                def lora_ready_init():
                    """LoRAドロップダウンの初期化を行う関数"""
                    
                    # 現在のuse_loraとlora_modeの値を取得
                    use_lora_value = getattr(use_lora, 'value', False)
                    lora_mode_value = getattr(lora_mode, 'value', translate("ディレクトリから選択"))
                    
                    # グローバル変数を更新
                    global previous_lora_mode
                    previous_lora_mode = lora_mode_value
                    
                    if use_lora_value:
                        # LoRAが有効な場合
                        if lora_mode_value == translate("ディレクトリから選択"):
                            # ディレクトリから選択モードの場合はドロップダウンを初期化
                            choices = scan_lora_directory()
                            return [
                                gr.update(choices=choices, value=choices[0]),  # lora_dropdown1
                                gr.update(choices=choices, value=choices[0]),  # lora_dropdown2
                                gr.update(choices=choices, value=choices[0])   # lora_dropdown3
                            ]
                        else:
                            # ファイルアップロードモードの場合はドロップダウンを更新しない
                            return [gr.update(), gr.update(), gr.update()]
                    
                    # LoRAが無効な場合は何も更新しない
                    return [gr.update(), gr.update(), gr.update()]
                
                # 初期化用の非表示ボタン
                lora_init_btn = gr.Button(visible=False, elem_id="lora_init_btn_f1")
                lora_init_btn.click(
                    fn=lora_ready_init,
                    inputs=[],
                    outputs=[lora_dropdown1, lora_dropdown2, lora_dropdown3]
                )
                
                # UIロード後に自動的に初期化するJavaScriptを追加
                js_init_code = """
                function initLoraDropdowns() {
                    // UIロード後、少し待ってからボタンをクリック
                    setTimeout(function() {
                        // 非表示ボタンを探して自動クリック
                        var initBtn = document.getElementById('lora_init_btn_f1');
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
            
            # LoRAプリセット用変数を初期化
            lora_preset_group = None
            
            # LoRAプリセット機能（LoRAが有効な場合のみ）
            if has_lora_support:
                from eichi_utils.lora_preset_manager import save_lora_preset, load_lora_preset
                
                # LoRAプリセット機能（初期状態では非表示）
                with gr.Group(visible=False) as lora_preset_group:
                    # シンプルな1行レイアウト
                    with gr.Row():
                        # プリセット選択ボタン（1-5）
                        preset_buttons = []
                        for i in range(1, 6):
                            preset_buttons.append(
                                gr.Button(
                                    translate("設定{0}").format(i),
                                    variant="secondary",
                                    scale=1
                                )
                            )
                        
                        # Load/Save選択（ラベルなし、横並び）
                        with gr.Row(scale=1):
                            load_btn = gr.Button(translate("Load"), variant="primary", scale=1)
                            save_btn = gr.Button(translate("Save"), variant="secondary", scale=1)
                        # 内部的に使うRadio（非表示）
                        lora_preset_mode = gr.Radio(
                            choices=[translate("Load"), translate("Save")],
                            value=translate("Load"),
                            visible=False
                        )
                    
                    # プリセット状態表示
                    lora_preset_status = gr.Textbox(
                        label=translate("プリセット状態"),
                        value="",
                        interactive=False,
                        lines=1
                    )
                
                # LoRAタイプとプリセット表示の組み合わせを制御する関数
                def toggle_lora_and_preset(use_lora_val, lora_mode_val):
                    # LoRAが有効かつディレクトリから選択モードの場合のみプリセットを表示
                    preset_visible = use_lora_val and lora_mode_val == translate("ディレクトリから選択")
                    return gr.update(visible=preset_visible)
                
                # LoRAプリセット機能のハンドラー関数
                def handle_lora_preset_button(button_index, mode, lora1, lora2, lora3, scales):
                    """LoRAプリセットボタンのクリックを処理する"""
                    if mode == translate("Load"):  # Load
                        # ロードモード
                        loaded_values = load_lora_preset(button_index)
                        if loaded_values:
                            return (
                                gr.update(value=loaded_values[0]),  # lora_dropdown1
                                gr.update(value=loaded_values[1]),  # lora_dropdown2
                                gr.update(value=loaded_values[2]),  # lora_dropdown3
                                gr.update(value=loaded_values[3]),  # lora_scales_text
                                translate("設定{0}を読み込みました").format(button_index + 1)  # status
                            )
                        else:
                            return (
                                gr.update(), gr.update(), gr.update(), gr.update(),
                                translate("設定{0}の読み込みに失敗しました").format(button_index + 1)
                            )
                    else:
                        # セーブモード
                        success, message = save_lora_preset(button_index, lora1, lora2, lora3, scales)
                        return (
                            gr.update(), gr.update(), gr.update(), gr.update(),
                            message
                        )
                
                # Load/Saveボタンのイベントハンドラー
                def set_load_mode():
                    return (
                        gr.update(value=translate("Load")),
                        gr.update(variant="primary"),
                        gr.update(variant="secondary")
                    )
                
                def set_save_mode():
                    return (
                        gr.update(value=translate("Save")),
                        gr.update(variant="secondary"),
                        gr.update(variant="primary")
                    )
                
                # イベントの設定
                # プリセットボタンのイベント
                for i, btn in enumerate(preset_buttons):
                    btn.click(
                        fn=lambda mode, lora1, lora2, lora3, scales, idx=i: handle_lora_preset_button(
                            idx, mode, lora1, lora2, lora3, scales
                        ),
                        inputs=[lora_preset_mode, lora_dropdown1, lora_dropdown2, lora_dropdown3, lora_scales_text],
                        outputs=[lora_dropdown1, lora_dropdown2, lora_dropdown3, lora_scales_text, lora_preset_status]
                    )
                
                # Load/Saveボタンのイベント
                load_btn.click(
                    set_load_mode,
                    outputs=[lora_preset_mode, load_btn, save_btn]
                )
                
                save_btn.click(
                    set_save_mode,
                    outputs=[lora_preset_mode, load_btn, save_btn]
                )
                
                # LoRA使用状態とモードの変更でプリセット表示を更新
                use_lora.change(
                    toggle_lora_and_preset,
                    inputs=[use_lora, lora_mode],
                    outputs=[lora_preset_group]
                )
                
                lora_mode.change(
                    toggle_lora_and_preset,
                    inputs=[use_lora, lora_mode],
                    outputs=[lora_preset_group]
                )
            else:
                # LoRAサポートがない場合はダミー
                lora_preset_group = gr.Group(visible=False)

            # FP8最適化設定は開始・終了ボタンの下に移動済み

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
                        sorted_choices, default_value = fix_prompt_preset_dropdown_initialization()
                        preset_dropdown = gr.Dropdown(
                            label=translate("プリセット"),
                            choices=sorted_choices,
                            value=default_value,
                            type="value"
                        )
                        # sorted_choices = [(name, name) for name in sorted(default_presets) + sorted(user_presets)]
                        # preset_dropdown = gr.Dropdown(label=translate("プリセット"), choices=sorted_choices, value=default_preset, type="value")

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

                # collect_section_settings関数は未使用のため削除

                # シンプルな互換性のためのダミーステートを作成
                section_settings = gr.State([[None, None, ""] for _ in range(max_keyframes)])
                section_inputs = []

                # update_section_settings関数は未使用のため削除

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
                    {translate('<strong>計算詳細</strong>: フレームサイズ={0}, 総フレーム数={1}, セクションあたり={2}フレーム, 必要セクション数={3}').format(frame_size, total_frames, frame_count, total_sections)}
                    <br>
                    {translate('動画モード {0} とフレームサイズ {1} で必要なセクション数: <strong>{2}</strong>').format(length, frame_size, total_sections)}
                    </div>"""

                    # セクション計算ログ
                    print(translate("計算結果: モード=通常, フレームサイズ={0}, latent_window_size={1}, 総フレーム数={2}, 必要セクション数={3}").format(frame_size, latent_window_size, total_frames, total_sections))

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

                # セクション表示機能をシンプル化
                def update_section_visibility(mode, length, frame_size=None):
                    """F1モードではシンプル化された関数"""
                    # 秒数だけ計算して返す
                    seconds = get_video_seconds(length)
                    print(translate("F1モード：シンプル設定（不要な機能を削除済み）"))

                    # 最低限の返値（入力に対応するだけの空更新）
                    return [gr.update()] * 2 + [] + [gr.update(value=seconds)] + []

                # 注意: この関数のイベント登録は、total_second_lengthのUIコンポーネント定義後に行うため、
                # ここでは関数の定義のみ行い、実際のイベント登録はUIコンポーネント定義後に行います。

                # 動画長変更イベントでもセクション数計算を更新
                length_radio.change(
                    fn=update_section_calculation,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_calc_display]
                )

                # F1モードではセクションタイトルは不要

                # モード変更時にも計算を更新
                mode_radio.change(
                    fn=update_section_calculation,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_calc_display]
                )

                # F1モードではセクションタイトルは不要

                # モード変更時の処理もtotal_second_lengthコンポーネント定義後に行います

                # 動画長変更時のセクション表示更新もtotal_second_lengthコンポーネント定義後に行います

                # F1モードでは終端フレームとループモード関連の機能をすべて削除

                # キーフレーム処理関数とZipファイルアップロード処理関数は未使用のため削除


        with gr.Column():
            result_video = gr.Video(
                label=translate("Finished Frames"),
                key="result_video",
                autoplay=True,
                show_share_button=False,
                height=512,
                loop=True,
                format="mp4",
                interactive=False,
            )
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)

            # フレームサイズ切替用のUIコントロールは上部に移動したため削除

            # 計算結果を表示するエリア
            section_calc_display = gr.HTML("", label="")

            use_teacache = gr.Checkbox(
                label=translate('Use TeaCache'), 
                value=saved_app_settings.get("use_teacache", True) if saved_app_settings else True, 
                info=translate('Faster speed, but often makes hands and fingers slightly worse.'),
                elem_classes="saveable-setting"
            )

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

            # チェックボックスの変更時に再読み込みを行う
            def check_metadata_on_checkbox_change(copy_enabled, image_path):
                if not copy_enabled or image_path is None:
                    return [gr.update()] * 2
                # チェックボックスオン時に、画像があれば再度メタデータを読み込む
                return update_from_image_metadata(image_path, copy_enabled)

            # update_section_metadata_on_checkbox_change関数は未使用のため削除

            copy_metadata.change(
                fn=check_metadata_on_checkbox_change,
                inputs=[copy_metadata, input_image],
                outputs=[prompt, seed]
            )


            def set_random_seed(is_checked):
                if is_checked:
                    return random.randint(0, 2**32 - 1)
                else:
                    return gr.update()
            use_random_seed.change(fn=set_random_seed, inputs=use_random_seed, outputs=seed)

            total_second_length = gr.Slider(label=translate("Total Video Length (Seconds)"), minimum=1, maximum=120, value=1, step=1)
            latent_window_size = gr.Slider(label=translate("Latent Window Size"), minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
            steps = gr.Slider(
                label=translate("Steps"), 
                minimum=1, 
                maximum=100, 
                value=saved_app_settings.get("steps", 25) if saved_app_settings else 25, 
                step=1, 
                info=translate('Changing this value is not recommended.'),
                elem_classes="saveable-setting"
            )

            cfg = gr.Slider(
                label=translate("CFG Scale"), 
                minimum=1.0, 
                maximum=32.0, 
                value=saved_app_settings.get("cfg", 1.0) if saved_app_settings else 1.0, 
                step=0.01, 
                visible=False,  # Should not change
                elem_classes="saveable-setting"
            )
            gs = gr.Slider(
                label=translate("Distilled CFG Scale"), 
                minimum=1.0, 
                maximum=32.0, 
                value=saved_app_settings.get("gs", 10) if saved_app_settings else 10, 
                step=0.01, 
                info=translate('Changing this value is not recommended.'),
                elem_classes="saveable-setting"
            )
            rs = gr.Slider(label=translate("CFG Re-Scale"), minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

            available_cuda_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
            default_gpu_memory_preservation_gb = 6 if available_cuda_memory_gb >= 20 else (8 if available_cuda_memory_gb > 16 else 10)
            gpu_memory_preservation = gr.Slider(label=translate("GPU Memory to Preserve (GB) (smaller = more VRAM usage)"), minimum=6, maximum=128, value=saved_app_settings.get("gpu_memory_preservation", default_gpu_memory_preservation_gb) if saved_app_settings else default_gpu_memory_preservation_gb, step=0.1, info=translate("空けておくGPUメモリ量を指定。小さい値=より多くのVRAMを使用可能=高速、大きい値=より少ないVRAMを使用=安全"), elem_classes="saveable-setting")

            # MP4圧縮設定スライダーを追加
            mp4_crf = gr.Slider(
                label=translate("MP4 Compression"), 
                minimum=0, 
                maximum=100, 
                value=saved_app_settings.get("mp4_crf", 16) if saved_app_settings else 16, 
                step=1, 
                info=translate("数値が小さいほど高品質になります。0は無圧縮。黒画面が出る場合は16に設定してください。"),
                elem_classes="saveable-setting"
            )

            # セクションごとの動画保存チェックボックスを追加（デフォルトOFF）
            keep_section_videos = gr.Checkbox(label=translate("完了時にセクションごとの動画を残す - チェックがない場合は最終動画のみ保存されます（デフォルトOFF）"), value=saved_app_settings.get("keep_section_videos", False) if saved_app_settings else False, elem_classes="saveable-setting")

            # テンソルデータ保存チェックボックス违加
            save_tensor_data = gr.Checkbox(
                label=translate("完了時にテンソルデータ(.safetensors)も保存 - このデータを別の動画の後に結合可能"),
                value=saved_app_settings.get("save_tensor_data", False) if saved_app_settings else False,
                info=translate("チェックすると、生成されたテンソルデータを保存します。アップロードされたテンソルがあれば、結合したテンソルデータも保存されます。"),
                elem_classes="saveable-setting"
            )

            # セクションごとの静止画保存チェックボックスを追加（デフォルトOFF）
            save_section_frames = gr.Checkbox(label=translate("Save Section Frames"), value=saved_app_settings.get("save_section_frames", False) if saved_app_settings else False, info=translate("各セクションの最終フレームを静止画として保存します（デフォルトOFF）"), elem_classes="saveable-setting")
            
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
                    # value=saved_app_settings.get("frame_save_mode", translate("保存しない")) if saved_app_settings else translate("保存しない"),
                    value=translate(saved_app_settings.get("frame_save_mode", "保存しない") if saved_app_settings else "保存しない"),
                    info=translate("フレーム画像の保存方法を選択します。過去セクション分も含めて保存します。全セクションか最終セクションのみか選択できます。"),
                    elem_classes="saveable-setting"
                )

            # UIコンポーネント定義後のイベント登録
            # F1モードではセクション機能を削除済み - シンプル化したイベントハンドラ
            mode_radio.change(
                fn=update_section_visibility,
                inputs=[mode_radio, length_radio, frame_size_radio],
                outputs=[input_image, input_image, total_second_length]
            )

            # フレームサイズ変更時の処理（シンプル化）
            frame_size_radio.change(
                fn=update_section_visibility,
                inputs=[mode_radio, length_radio, frame_size_radio],
                outputs=[input_image, input_image, total_second_length]
            )

            # 動画長変更時の処理（シンプル化）
            length_radio.change(
                fn=update_section_visibility,
                inputs=[mode_radio, length_radio, frame_size_radio],
                outputs=[input_image, input_image, total_second_length]
            )


            # Image影響度調整スライダー
            with gr.Group():
                gr.Markdown("### " + translate("Image影響度調整"))
                image_strength = gr.Slider(
                    label=translate("Image影響度"),
                    minimum=1.00,
                    maximum=1.02,
                    value=saved_app_settings.get("image_strength", 1.00) if saved_app_settings else 1.00,
                    step=0.001,
                    info=translate("開始フレーム(Image)が動画に与える影響の強さを調整します。1.00が通常の動作（100%）です。値を大きくすると始点の影響が強まり、変化が少なくなります。100%-102%の範囲で0.1%刻みの微調整が可能です。"),
                    elem_classes="saveable-setting"
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
            
            # アプリケーション設定管理UI
            with gr.Group():
                gr.Markdown(f"### " + translate("アプリケーション設定"))
                with gr.Row():
                    with gr.Column(scale=1):
                        save_current_settings_btn = gr.Button(value=translate("💾 現在の設定を保存"), size="sm")
                    with gr.Column(scale=1):
                        reset_settings_btn = gr.Button(value=translate("🔄 設定をリセット"), size="sm")
                
                # 自動保存設定
                save_settings_default_value = saved_app_settings.get("save_settings_on_start", False) if saved_app_settings else False
                save_settings_on_start = gr.Checkbox(
                    label=translate("生成開始時に自動保存"),
                    value=save_settings_default_value,
                    info=translate("チェックをオンにすると、生成開始時に現在の設定が自動的に保存されます。設定は再起動時に反映されます。"),
                    elem_classes="saveable-setting",
                    interactive=True
                )
                
                # 完了時のアラーム設定
                alarm_default_value = saved_app_settings.get("alarm_on_completion", True) if saved_app_settings else True
                alarm_on_completion = gr.Checkbox(
                    label=translate("完了時にアラームを鳴らす(Windows)"),
                    value=alarm_default_value,
                    info=translate("チェックをオンにすると、生成完了時にアラーム音を鳴らします（Windows）"),
                    elem_classes="saveable-setting",
                    interactive=True
                )
                
                # ログ設定
                gr.Markdown("### " + translate("ログ設定"))
                
                # 設定からログ設定を読み込む
                all_settings = load_settings()
                log_settings = all_settings.get('log_settings', {'log_enabled': False, 'log_folder': 'logs'})
                
                # ログ有効/無効設定
                log_enabled = gr.Checkbox(
                    label=translate("コンソールログを出力する"),
                    value=log_settings.get('log_enabled', False),
                    info=translate("チェックをオンにすると、コンソール出力をログファイルにも保存します"),
                    elem_classes="saveable-setting",
                    interactive=True
                )
                
                # ログ出力先設定
                log_folder = gr.Textbox(
                    label=translate("ログ出力先"),
                    value=log_settings.get('log_folder', 'logs'),
                    info=translate("ログファイルの保存先フォルダを指定します"),
                    elem_classes="saveable-setting",
                    interactive=True
                )
                
                # ログフォルダを開くボタン
                open_log_folder_btn = gr.Button(value=translate("📂 ログフォルダを開く"), size="sm")
                
                # ログフォルダを開くボタンのクリックイベント
                open_log_folder_btn.click(fn=open_log_folder)
                
                # 設定状態の表示
                settings_status = gr.Markdown("")
            
            # アプリケーション設定の保存機能
            def save_app_settings_handler(
                # Basic settings
                resolution_val,
                mp4_crf_val,
                steps_val,
                cfg_val,
                # Performance settings
                use_teacache_val,
                gpu_memory_preservation_val,
                # Detail settings
                gs_val,
                # F1 specific settings
                image_strength_val,
                # Save settings
                keep_section_videos_val,
                save_section_frames_val,
                save_tensor_data_val,
                frame_save_mode_val,
                # Auto-save settings
                save_settings_on_start_val,
                alarm_on_completion_val,
                # Log settings
                log_enabled_val,
                log_folder_val,
                # ADD NEW CONFIG QUEUE SETTING
                add_timestamp_to_config_val
            ):
                """現在の設定を保存"""
                from eichi_utils.settings_manager import save_app_settings_f1
                
                # アプリ設定用の設定辞書を作成
                current_settings = {
                    # 基本設定
                    "resolution": resolution_val,
                    "mp4_crf": mp4_crf_val,
                    "steps": steps_val,
                    "cfg": cfg_val,
                    # パフォーマンス設定
                    "use_teacache": use_teacache_val,
                    "gpu_memory_preservation": gpu_memory_preservation_val,
                    # 詳細設定
                    "gs": gs_val,
                    # F1独自設定
                    "image_strength": image_strength_val,
                    # 保存設定
                    "keep_section_videos": keep_section_videos_val,
                    "save_section_frames": save_section_frames_val,
                    "save_tensor_data": save_tensor_data_val,
                    "frame_save_mode": frame_save_mode_val,
                    # 自動保存・アラーム設定
                    "save_settings_on_start": save_settings_on_start_val,
                    "alarm_on_completion": alarm_on_completion_val,
                    # CONFIG QUEUE設定 - NEW
                    "add_timestamp_to_config": bool(add_timestamp_to_config_val)
                }
                
                # アプリ設定を保存
                try:
                    app_success = save_app_settings_f1(current_settings)
                except Exception as e:
                    return translate("設定の保存に失敗しました: {0}").format(str(e))
                
                # ログ設定も保存 - 値の型を確認
                # log_enabledはbooleanに確実に変換
                is_log_enabled = False
                if isinstance(log_enabled_val, bool):
                    is_log_enabled = log_enabled_val
                elif hasattr(log_enabled_val, 'value'):
                    is_log_enabled = bool(log_enabled_val.value)
                
                # log_folderは文字列に確実に変換
                log_folder_path = "logs"
                if log_folder_val and isinstance(log_folder_val, str):
                    log_folder_path = log_folder_val
                elif hasattr(log_folder_val, 'value') and log_folder_val.value:
                    log_folder_path = str(log_folder_val.value)
                
                log_settings = {
                    "log_enabled": is_log_enabled,
                    "log_folder": log_folder_path
                }
                
                # 全体設定を取得し、ログ設定を更新
                all_settings = load_settings()
                all_settings['log_settings'] = log_settings
                log_success = save_settings(all_settings)
                
                # ログ設定を適用（設定保存後、すぐに新しいログ設定を反映）
                if log_success:
                    # 一旦ログを無効化
                    disable_logging()
                    # 新しい設定でログを再開（有効な場合）
                    apply_log_settings(log_settings, source_name="endframe_ichi_f1")
                    print(translate("ログ設定を更新しました: 有効={0}, フォルダ={1}").format(
                        log_enabled_val, log_folder_val))
                
                if app_success and log_success:
                    return translate("設定を保存しました")
                else:
                    return translate("設定の一部保存に失敗しました")

            def reset_app_settings_handler():
                """設定をデフォルトに戻す"""
                from eichi_utils.settings_manager import get_default_app_settings_f1
                from locales import i18n
                
                # 現在の言語設定を取得して、その言語用のデフォルト設定を取得
                current_lang = i18n.lang
                
                # 言語設定を考慮したデフォルト設定を取得
                default_settings = get_default_app_settings_f1(current_lang)
                updates = []
                
                # 各UIコンポーネントのデフォルト値を設定（F1の順序に合わせる）
                updates.append(gr.update(value=default_settings.get("resolution", 640)))  # 1
                updates.append(gr.update(value=default_settings.get("mp4_crf", 16)))  # 2
                updates.append(gr.update(value=default_settings.get("steps", 25)))  # 3
                updates.append(gr.update(value=default_settings.get("cfg", 1.0)))  # 4
                updates.append(gr.update(value=default_settings.get("use_teacache", True)))  # 5
                updates.append(gr.update(value=default_settings.get("gpu_memory_preservation", 6)))  # 6
                updates.append(gr.update(value=default_settings.get("gs", 10)))  # 7
                # F1独自
                updates.append(gr.update(value=default_settings.get("image_strength", 1.0)))  # 8
                updates.append(gr.update(value=default_settings.get("keep_section_videos", False)))  # 9
                updates.append(gr.update(value=default_settings.get("save_section_frames", False)))  # 10
                updates.append(gr.update(value=default_settings.get("save_tensor_data", False)))  # 11
                updates.append(gr.update(value=default_settings.get("frame_save_mode", translate("保存しない"))))  # 12
                updates.append(gr.update(value=default_settings.get("save_settings_on_start", False)))  # 13
                updates.append(gr.update(value=default_settings.get("alarm_on_completion", True)))  # 14
                
                # ログ設定 (15番目め16番目の要素)
                # ログ設定は固定値を使用 - 絶対に文字列とbooleanを使用
                updates.append(gr.update(value=False))  # log_enabled (15)
                updates.append(gr.update(value="logs"))  # log_folder (16)
                
                # ログ設定をアプリケーションに適用
                default_log_settings = {
                    "log_enabled": False,
                    "log_folder": "logs"
                }

                # CONFIG QUEUE設定 - NEW (17番目の要素)
                updates.append(gr.update(value=default_settings.get("add_timestamp_to_config", True)))  # 17
                
                # 設定ファイルを更新
                all_settings = load_settings()
                all_settings['log_settings'] = default_log_settings
                save_settings(all_settings)
                
                # ログ設定を適用 (既存のログファイルを閉じて、設定に従って再設定)
                disable_logging()  # 既存のログを閉じる
                
                # 設定状態メッセージ (18番目の要素)
                updates.append(translate("設定をデフォルトに戻しました"))
                
                return updates

    # 実行前のバリデーション関数
    def validate_and_process(*args):
        """入力画像または最後のキーフレーム画像のいずれかが有効かどうかを確認し、問題がなければ処理を実行する"""
        # グローバル変数の宣言
        global batch_stopped, queue_enabled, queue_type, prompt_queue_file_path, image_queue_files

        input_img = args[0]  # 入力の最初が入力画像

        # UIのセットアップとips配列 (実際のips配列の順序):
        # [0]input_image, [1]prompt, [2]n_prompt, [3]seed, [4]total_second_length, [5]latent_window_size,
        # [6]steps, [7]cfg, [8]gs, [9]rs, [10]gpu_memory_preservation, [11]use_teacache, [12]use_random_seed,
        # [13]mp4_crf, [14]all_padding_value, [15]image_strength, [16]frame_size_radio, [17]keep_section_videos,
        # [18]lora_files, [19]lora_files2, [20]lora_files3, [21]lora_scales_text, [22]output_dir, [23]save_section_frames,
        # [24]use_all_padding, [25]use_lora, [26]lora_mode, [27]lora_dropdown1, [28]lora_dropdown2, [29]lora_dropdown3,
        # [30]save_tensor_data, [31]section_settings, [32]tensor_data_input, [33]fp8_optimization, [34]resolution,
        # [35]batch_count, [36]frame_save_mode, [37]use_queue, [38]prompt_queue_file, [39]save_settings_on_start, [40]alarm_on_completion
        
        # 各引数を明示的に取得 - コメントに基づいて正確なインデックスを使用
        output_dir = args[22] if len(args) > 22 else None
        save_section_frames = args[23] if len(args) > 23 else False
        use_all_padding = args[24] if len(args) > 24 else False
        use_lora = args[25] if len(args) > 25 else False
        lora_mode = args[26] if len(args) > 26 else translate("ディレクトリから選択")
        lora_dropdown1 = args[27] if len(args) > 27 else None
        lora_dropdown2 = args[28] if len(args) > 28 else None
        lora_dropdown3 = args[29] if len(args) > 29 else None
        save_tensor_data = args[30] if len(args) > 30 else False
        # F1版ではsection_settingsは常に固定値を使用（無印版の部分は不要）
        # F1版用のsection_settings - 一貫性のために配列を作成
        # section_settingsが存在するかチェックする（args[31]）
        section_settings = [[None, None, ""] for _ in range(50)]
        if len(args) > 31 and args[31] is not None:
            # すでに配列なら使用、そうでなければ初期化した配列を使用
            if isinstance(args[31], list):
                section_settings = args[31]
        tensor_data_input = args[32] if len(args) > 32 else None
        fp8_optimization = args[33] if len(args) > 33 else True
        resolution_value = args[34] if len(args) > 34 else 640
        batch_count = args[35] if len(args) > 35 else 1
        frame_save_mode = args[36] if len(args) > 36 else translate("保存しない")
        # 新しいキュー関連の引数を取得
        use_queue_ui = args[37] if len(args) > 37 else False
        prompt_queue_file_ui = args[38] if len(args) > 38 else None
        
        # 自動保存・アラーム設定の引数を取得
        save_settings_on_start_ui = args[39] if len(args) > 39 else False
        alarm_on_completion_ui = args[40] if len(args) > 40 else False
        
        # 値の取得処理
        actual_save_settings_value = save_settings_on_start_ui
        if hasattr(save_settings_on_start_ui, 'value'):
            actual_save_settings_value = save_settings_on_start_ui.value
        
        # アラーム設定値を取得
        actual_alarm_value = False  # デフォルトはオフ
        
        # Gradioのチェックボックスから値を適切に取得
        if isinstance(alarm_on_completion_ui, bool):
            # booleanの場合はそのまま使用
            actual_alarm_value = alarm_on_completion_ui
        elif hasattr(alarm_on_completion_ui, 'value'):
            # Gradioオブジェクトの場合はvalue属性を取得
            if isinstance(alarm_on_completion_ui.value, bool):
                actual_alarm_value = alarm_on_completion_ui.value

        # キュー設定の出力
        print(translate("キュータイプ: {0}").format(queue_type))

        # キュー機能の状態を更新（UIチェックボックスからの値を直接反映）
        queue_enabled = use_queue_ui

        # section_settings型チェック - エラー修正
        if len(args) > 31 and args[31] is not None and not isinstance(args[31], list):
            print(translate("section_settingsが正しい型ではありません: {0}. 初期化します。").format(type(args[31]).__name__))
            section_settings = [[None, None, ""] for _ in range(50)]

        # バッチ数の上限を設定
        batch_count = max(1, min(int(batch_count), 100))  # 1〜100の範囲に制限

        # イメージキューの場合は、事前に画像ファイルリストを更新
        if queue_enabled and queue_type == "image":
            # 入力フォルダから画像ファイルリストを更新
            get_image_queue_files()
            image_queue_count = len(image_queue_files)
            print(translate("イメージキュー使用: 入力フォルダの画像 {0} 個を使用します").format(image_queue_count))

            # バッチ数を画像数+1（入力画像を含む）に合わせる
            if image_queue_count > 0:
                # 入力画像を使う1回 + 画像ファイル分のバッチ数
                total_needed_batches = 1 + image_queue_count

                # 設定されたバッチ数より必要数が多い場合は調整
                if total_needed_batches > batch_count:
                    print(translate("画像キュー数+1に合わせてバッチ数を自動調整: {0} → {1}").format(batch_count, total_needed_batches))
                    batch_count = total_needed_batches

        # プロンプトキューの場合はファイルの内容を確認
        if queue_enabled and queue_type == "prompt":
            # グローバル変数からファイルパスを取得
            if prompt_queue_file_path is not None:
                queue_file_path = prompt_queue_file_path
                print(translate("プロンプトキューファイル: {0}").format(queue_file_path))

                # ファイルパスが有効かチェック
                if os.path.exists(queue_file_path):
                    print(translate("プロンプトキューファイルの内容を読み込みます: {0}").format(queue_file_path))
                    try:
                        with open(queue_file_path, 'r', encoding='utf-8') as f:
                            lines = [line.strip() for line in f.readlines() if line.strip()]
                            queue_prompts_count = len(lines)
                            print(translate("有効なプロンプト行数: {0}").format(queue_prompts_count))

                            if queue_prompts_count > 0:
                                # サンプルとして最初の数行を表示
                                sample_lines = lines[:min(3, queue_prompts_count)]
                                print(translate("プロンプトサンプル: {0}").format(sample_lines))

                                # バッチ数をプロンプト数に合わせる
                                if queue_prompts_count > batch_count:
                                    print(translate("プロンプト数に合わせてバッチ数を自動調整: {0} → {1}").format(batch_count, queue_prompts_count))
                                    batch_count = queue_prompts_count
                            else:
                                print(translate("プロンプトキューファイルに有効なプロンプトがありません"))
                    except Exception as e:
                        print(translate("プロンプトキューファイル読み込みエラー: {0}").format(str(e)))
                else:
                    print(translate("プロンプトキューファイルが存在しないか無効です: {0}").format(queue_file_path))
            else:
                print(translate("プロンプトキュー無効: ファイルが正しくアップロードされていません"))
        
        # Gradioのラジオボタンオブジェクトが直接渡されているか、文字列値が渡されているかを確認
        if hasattr(frame_save_mode, 'value'):
            # Gradioオブジェクトの場合は値を取得
            frame_save_mode_value = frame_save_mode.value
        else:
            # 文字列などの通常の値の場合はそのまま使用
            frame_save_mode_value = frame_save_mode
        
        # フレーム保存モードはworker関数内で処理されるため、ここでの設定は不要
        # frame_save_mode は worker関数に直接渡される
        # バッチ回数を有効な範囲に制限
        batch_count = max(1, min(int(batch_count), 100))

        # F1モードでは固定のダミーセクション設定を使用
        section_settings = [[None, None, ""] for _ in range(50)]

        # 現在の動画長設定とフレームサイズ設定を渡す
        is_valid, error_message = validate_images(input_img, section_settings, length_radio, frame_size_radio)

        if not is_valid:
            # 画像が無い場合はエラーメッセージを表示して終了
            yield None, gr.update(visible=False), translate("エラー: 画像が選択されていません"), error_message, gr.update(interactive=True), gr.update(interactive=False), gr.update()
            return

        # 画像がある場合は通常の処理を実行
        # 元のパラメータを使用
        new_args = list(args)
        
        # 引数を正しいインデックスで設定 (LoRA関連パラメータ追加に伴い調整)
        if len(new_args) > 25:
            new_args[25] = use_lora  # use_loraを確実に正しい値に
        if len(new_args) > 26:
            new_args[26] = lora_mode  # lora_modeを設定
        if len(new_args) > 27:
            new_args[27] = lora_dropdown1  # lora_dropdown1を設定
        if len(new_args) > 28:
            new_args[28] = lora_dropdown2  # lora_dropdown2を設定
        if len(new_args) > 29:
            new_args[29] = lora_dropdown3  # lora_dropdown3を設定
        # ===========================================================
        # 重要: save_tensor_dataは正確にインデックス30に設定すること
        # 後続のコードでこのインデックスが上書きされないよう注意
        # ===========================================================
        if len(new_args) > 30:
            new_args[30] = save_tensor_data  # save_tensor_dataを確実に正しい値に
        
        # F1モードでは固定のセクション設定を使用
        if len(new_args) > 31:
            new_args[31] = section_settings
        
        # その他の引数も必要に応じて設定
        if len(new_args) <= 37:  # 引数の最大インデックスに合わせて調整
            # 不足している場合は拡張
            new_args.extend([None] * (37 - len(new_args)))
            if len(new_args) <= 31:
                if len(new_args) <= 30:
                    if len(new_args) <= 29:
                        # resolutionもない場合
                        new_args.append(resolution_value)  # resolutionを追加
                    new_args.append(batch_count)  # batch_countを追加
        else:
            # 既に存在する場合は更新
            # =============================================================================
            # 重要: save_tensor_data(index 30)は3507行で既に設定済みのため、上書きしないこと
            # 以前はここでnew_args[30] = batch_countとなっており、テンソルデータが常に保存される
            # バグが発生していた。インデックスを間違えないよう注意すること。
            # =============================================================================
            new_args[34] = resolution_value  # resolution
            new_args[35] = batch_count  # batch_count
            # save_tensor_dataは上部で既に設定済み (new_args[30])
            new_args[36] = frame_save_mode  # frame_save_mode
            new_args[37] = use_queue_ui  # use_queue
            new_args[38] = prompt_queue_file_ui  # prompt_queue_file
            new_args[39] = actual_save_settings_value  # save_settings_on_start
            new_args[40] = actual_alarm_value  # alarm_on_completion

        # process関数に渡す前に重要な値を確認
        # 注意: ここではインデックス25と書かれていますが、これは誤りです
        # 正しくはnew_args[30]がsave_tensor_dataの値です
        
        # new_argsの引数を出力（特にsection_settings）
        # section_settingsは配列であることを確認
        section_settings_index = 31  # section_settingsのインデックス
        if len(new_args) > section_settings_index:
            if not isinstance(new_args[section_settings_index], list):
                print(translate("section_settingsがリストではありません。修正します。"))
                new_args[section_settings_index] = [[None, None, ""] for _ in range(50)]

        # process関数のジェネレータを返す
        yield from process(*new_args)

    # 設定保存ボタンのクリックイベント
    save_current_settings_btn.click(
        fn=save_app_settings_handler,
        inputs=[
            resolution,
            mp4_crf,
            steps,
            cfg,
            use_teacache,
            gpu_memory_preservation,
            gs,
            image_strength,
            keep_section_videos,
            save_section_frames,
            save_tensor_data,
            frame_save_mode,
            save_settings_on_start,
            alarm_on_completion,
            # ログ設定を追加
            log_enabled,
            log_folder,
            # ADD CONFIG QUEUE SETTING
            config_queue_components['add_timestamp_to_config']  # NEW INPUT
        ],
        outputs=[settings_status]
    )

    # 設定リセットボタンのクリックイベント
    reset_settings_btn.click(
        fn=reset_app_settings_handler,
        inputs=[],
        outputs=[
            resolution,           # 1
            mp4_crf,              # 2
            steps,                # 3
            cfg,                  # 4
            use_teacache,         # 5
            gpu_memory_preservation, # 6
            gs,                   # 7
            image_strength,       # 8
            keep_section_videos,  # 9
            save_section_frames,  # 10
            save_tensor_data,     # 11
            frame_save_mode,      # 12
            save_settings_on_start, # 13
            alarm_on_completion,  # 14
            log_enabled,          # 15
            log_folder,           # 16
            config_queue_components['add_timestamp_to_config'], # 17 - NEW OUTPUT
            settings_status       # 18
        ]
    )

    # 実行ボタンのイベント
    # ===================================================================================================
    # 重要: ips配列の引数の順序と、validate_and_process/process/worker関数の引数の順序を正確に一致させる
    # インデックスを変更する場合は、全ての関連箇所（validate_and_process内の処理）も合わせて変更すること
    # 特に重要: [30]save_tensor_dataのインデックスは変更しないこと。変更すると誤作動の原因となります。
    # 5/13修正: save_tensor_data(インデックス30)はバッチカウントに上書きされる問題を修正しました。
    # ===================================================================================================
    # 注意: 以下が実際のips配列の順序です
    #  [0]input_image, [1]prompt, [2]n_prompt, [3]seed, [4]total_second_length, [5]latent_window_size,
    #  [6]steps, [7]cfg, [8]gs, [9]rs, [10]gpu_memory_preservation, [11]use_teacache, [12]use_random_seed,
    #  [13]mp4_crf, [14]all_padding_value, [15]image_strength, [16]frame_size_radio, [17]keep_section_videos,
    #  [18]lora_files, [19]lora_files2, [20]lora_files3, [21]lora_scales_text, [22]output_dir, [23]save_section_frames,
    #  [24]use_all_padding, [25]use_lora, [26]lora_mode, [27]lora_dropdown1, [28]lora_dropdown2, [29]lora_dropdown3,
    #  [30]save_tensor_data, [31]section_settings, [32]tensor_data_input, [33]fp8_optimization, [34]resolution,
    #  [35]batch_count, [36]frame_save_mode, [37]use_queue, [38]prompt_queue_file, [39]save_settings_on_start, [40]alarm_on_completion
    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_random_seed, mp4_crf, all_padding_value, image_strength, frame_size_radio, keep_section_videos, lora_files, lora_files2, lora_files3, lora_scales_text, output_dir, save_section_frames, use_all_padding, use_lora, lora_mode, lora_dropdown1, lora_dropdown2, lora_dropdown3, save_tensor_data, section_settings, tensor_data_input, fp8_optimization, resolution, batch_count, frame_save_mode, use_queue, prompt_queue_file, save_settings_on_start, alarm_on_completion]

    start_button.click(fn=validate_and_process_with_queue_check, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, queue_start_button, seed])
    end_button.click(fn=end_process_enhanced, outputs=[end_button,queue_start_button])

    # F1モードではセクション機能とキーフレームコピー機能を削除済み

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

    # F1モードではキーフレームコピー機能を削除済み
    
    # =============================================================================
    # SETUP CONFIG QUEUE EVENT HANDLERS - MOVED INSIDE BLOCKS CONTEXT
    # =============================================================================
    
    # Setup config queue event handlers
    ui_components = {
        'input_image': input_image,
        'prompt': prompt,
        'use_lora': use_lora,
        'lora_mode': lora_mode,
        'lora_dropdown1': lora_dropdown1,
        'lora_dropdown2': lora_dropdown2,
        'lora_dropdown3': lora_dropdown3,
        'lora_files': lora_files,
        'lora_files2': lora_files2,
        'lora_files3': lora_files3,
        'lora_scales_text': lora_scales_text,
        'progress_desc': progress_desc,
        'progress_bar': progress_bar,
        'preview_image': preview_image,
        'result_video': result_video
    }

    setup_enhanced_config_queue_events(config_queue_components, ui_components)

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

# F1モードではキーフレームコピー機能を削除済み

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
        print("======================================================")
        print(translate("エラー: FramePack-eichiは既に起動しています。"))
        print(translate("同時に複数のインスタンスを実行することはできません。"))
        print(translate("現在実行中のアプリケーションを先に終了してください。"))
        print("======================================================")
        input(translate("続行するには何かキーを押してください..."))
    else:
        # その他のOSErrorの場合は元のエラーを表示
        print(translate("エラーが発生しました: {e}").format(e=e))
        input(translate("続行するには何かキーを押してください..."))

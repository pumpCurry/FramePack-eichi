import os
import sys
from datetime import datetime

sys.path.append(
    os.path.abspath(
        os.path.realpath(
            os.path.join(os.path.dirname(__file__), "./submodules/FramePack")
        )
    )
)

# Windows環境で loop再生時に [WinError 10054] の warning が出るのを回避する設定
import asyncio

if sys.platform in ("win32", "cygwin"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.append(
    os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))
)

# VAEキャッシュ機能のインポート
from eichi_utils.vae_cache import vae_decode_cache

# グローバル変数の設定
vae_cache_enabled = False  # VAEキャッシュのチェックボックス状態を保持
current_prompt = None  # キューから読み込まれた現在のプロンプト

import random
import time
import traceback  # デバッグログ出力用

import argparse

import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from eichi_utils.combine_mode import (
    COMBINE_MODE,
    COMBINE_MODE_OPTIONS,
    COMBINE_MODE_OPTIONS_KEYS,
    COMBINE_MODE_DEFAULT,
    get_combine_mode,
)

from eichi_utils.png_metadata import (
    embed_metadata_to_png,
    extract_metadata_from_png,
    extract_metadata_from_numpy_array,
    PROMPT_KEY,
    SEED_KEY,
    SECTION_PROMPT_KEY,
    SECTION_NUMBER_KEY,
)

parser = argparse.ArgumentParser()
parser.add_argument("--share", action="store_true")
parser.add_argument("--server", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--inbrowser", action="store_true")
parser.add_argument("--lang", type=str, default="ja", help="Language: ja, zh-tw, en")
args = parser.parse_args()

# Load translations from JSON files
from locales.i18n_extended import set_lang, translate

set_lang(args.lang)

try:
    import winsound

    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False
import traceback

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.abspath(
        os.path.realpath(os.path.join(os.path.dirname(__file__), "../hf_download"))
    )
    print(translate("HF_HOMEを設定: {0}").format(os.environ["HF_HOME"]))
else:
    print(translate("既存のHF_HOMEを使用: {0}").format(os.environ["HF_HOME"]))

temp_dir = "./temp_for_zip_section_info"

# LoRAサポートの確認
has_lora_support = False
try:
    import lora_utils

    has_lora_support = True
    print(translate("LoRAサポートが有効です"))
except ImportError:
    print(
        translate(
            "LoRAサポートが無効です（lora_utilsモジュールがインストールされていません）"
        )
    )

# 設定モジュールをインポート（ローカルモジュール）
import os.path
from eichi_utils.video_mode_settings import (
    VIDEO_MODE_SETTINGS,
    get_video_modes,
    get_video_seconds,
    get_important_keyframes,
    get_copy_targets,
    get_max_keyframes_count,
    get_total_sections,
    generate_keyframe_guide_html,
    handle_mode_length_change,
    process_keyframe_change,
    MODE_TYPE_NORMAL,
    MODE_TYPE_LOOP,
)

# 設定管理モジュールをインポート
import cv2
from eichi_utils.settings_manager import (
    get_settings_file_path,
    get_output_folder_path,
    initialize_settings,
    load_settings,
    save_settings,
    open_output_folder,
)

# プリセット管理モジュールをインポート
from eichi_utils.preset_manager import (
    initialize_presets,
    load_presets,
    get_default_startup_prompt,
    save_preset,
    delete_preset,
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

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import (
    encode_prompt_conds,
    vae_encode,
    vae_decode_fake,
    vae_decode,
)
from diffusers_helper.utils import (
    save_bcthw_as_mp4,
    crop_or_pad_yield_mask,
    soft_append_bcthw,
    resize_and_center_crop,
    generate_timestamp,
)
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    cpu,
    gpu,
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device,
    DynamicSwapInstaller,
    unload_complete_models,
    load_model_as_complete,
)
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

from eichi_utils.transformer_manager import TransformerManager
from eichi_utils.text_encoder_manager import TextEncoderManager

from eichi_utils.tensor_processing import (
    process_tensor_chunks,
    print_tensor_info,
    ensure_tensor_properties,
    output_latent_to_image,
    fix_tensor_size,
    reorder_tensor,
)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 100

print(translate("Free VRAM {0} GB").format(free_mem_gb))
print(translate("High-VRAM Mode: {0}").format(high_vram))

# モデルを並列ダウンロードしておく
from eichi_utils.model_downloader import ModelDownloader

ModelDownloader().download_original()

# グローバルなモデル状態管理インスタンスを作成
# 通常モードではuse_f1_model=Falseを指定（デフォルト値なので省略可）
transformer_manager = TransformerManager(
    device=gpu, high_vram_mode=high_vram, use_f1_model=True
)
text_encoder_manager = TextEncoderManager(device=gpu, high_vram_mode=high_vram)

try:
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer_2"
    )
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder="vae",
        torch_dtype=torch.float16,
    ).cpu()

    # text_encoderとtext_encoder_2の初期化
    if not text_encoder_manager.ensure_text_encoder_state():
        raise Exception(translate("text_encoderとtext_encoder_2の初期化に失敗しました"))
    text_encoder, text_encoder_2 = text_encoder_manager.get_text_encoders()

    # transformerの初期化
    transformer_manager.ensure_download_models()
    transformer = (
        transformer_manager.get_transformer()
    )  # 仮想デバイス上のtransformerを取得

    # 他のモデルの読み込み
    feature_extractor = SiglipImageProcessor.from_pretrained(
        "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        "lllyasviel/flux_redux_bfl",
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    ).cpu()
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
if not high_vram and not vae_settings.get("custom_vae_settings", False):
    vae.enable_slicing()
    vae.enable_tiling()

vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)

vae.requires_grad_(False)
image_encoder.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(
        transformer, device=gpu
    )  # クラスを操作するので仮想デバイス上のtransformerでもOK
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
    input_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), input_folder_name_value
    )

    # 入力ディレクトリが存在するかチェック（ボタン押下時のみ作成するため、ここでは作成しない）
    if not os.path.exists(input_dir):
        print(
            translate(
                "入力ディレクトリが存在しません: {0}（保存及び入力フォルダを開くボタンを押すと作成されます）"
            ).format(input_dir)
        )
        return []

    # 画像ファイル（png, jpg, jpeg）のみをリスト
    image_files = []
    for file in sorted(os.listdir(input_dir)):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_files.append(os.path.join(input_dir, file))

    print(
        translate("入力ディレクトリから画像ファイル{0}個を読み込みました").format(
            len(image_files)
        )
    )
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
    open_output_folder,
)

# フォルダ構造を先に定義
webui_folder = os.path.dirname(os.path.abspath(__file__))

# 設定保存用フォルダの設定
settings_folder = os.path.join(webui_folder, "settings")
os.makedirs(settings_folder, exist_ok=True)

# 設定ファイル初期化
initialize_settings()

# ベースパスを定義
base_path = os.path.dirname(os.path.abspath(__file__))

# 設定から出力フォルダを取得
app_settings = load_settings()
output_folder_name = app_settings.get("output_folder", "outputs")
print(translate("設定から出力フォルダを読み込み: {0}").format(output_folder_name))

# 入力フォルダ名も設定から取得
input_folder_name_value = app_settings.get("input_folder", "inputs")
print(translate("設定から入力フォルダを読み込み: {0}").format(input_folder_name_value))

# 出力フォルダのフルパスを生成
outputs_folder = get_output_folder_path(output_folder_name)
os.makedirs(outputs_folder, exist_ok=True)

# 入力フォルダも存在確認して作成
input_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), input_folder_name_value
)
os.makedirs(input_dir, exist_ok=True)


def is_combine_mode(combine_mode, check_mode):
    if COMBINE_MODE_OPTIONS[combine_mode] == check_mode:
        return True
    else:
        return False


def remove_first_frame_from_tensor_latent(tensor_latent, trim_size):
    """テンソルデータの先頭フレームを削除する"""
    # 削除するフレームサイズを計算
    tensor_latent_size = tensor_latent.shape[2]
    edited_latent = tensor_latent
    if tensor_latent_size > trim_size:
        # テンソルデータの先頭フレームを削除
        if trim_size > 0:
            edited_latent = tensor_latent[:, :, trim_size:, :, :].clone()
            print(
                translate(
                    "アップロードされたテンソルデータの先頭フレームを削除しました。削除数: {0}/{1}"
                ).format(trim_size, tensor_latent_size)
            )
    else:
        print(
            translate(
                "警告: テンソルデータのフレーム数よりも、先頭フレーム削除数が大きく指定されているため、先頭フレーム削除は実施しません。"
            )
        )

    print(
        translate("テンソルデータ読み込み成功: shape={0}, dtype={1}").format(
            tensor_latent.shape, tensor_latent.dtype
        )
    )
    print(
        translate(
            "テンソルデータ読み込み成功（先頭フレーム削除後）: shape={0}, dtype={1}"
        ).format(edited_latent.shape, edited_latent.dtype)
    )
    stream.output_queue.push(
        (
            "progress",
            (
                None,
                translate("Tensor data loaded successfully!"),
                make_progress_bar_html(
                    10, translate("Tensor data loaded successfully!")
                ),
            ),
        )
    )
    return edited_latent


@torch.no_grad()
def worker(
    input_image,
    prompt,
    seed,
    steps,
    cfg,
    gs,
    rs,
    gpu_memory_preservation,
    use_teacache,
    mp4_crf=16,
    end_frame_strength=1.0,
    keep_section_videos=False,
    lora_files=None,
    lora_files2=None,
    lora_files3=None,
    lora_scales_text="0.8,0.8,0.8",
    output_dir=None,
    save_intermediate_frames=False,
    use_lora=False,
    lora_mode=None,
    lora_dropdown1=None,
    lora_dropdown2=None,
    lora_dropdown3=None,
    save_tensor_data=False,
    tensor_data_input=None,
    trim_start_latent_size=0,
    generation_latent_size=0,
    combine_mode=COMBINE_MODE_DEFAULT,
    fp8_optimization=False,
    batch_index=None,
    use_vae_cache=False,
):
    # グローバル変数を使用
    global vae_cache_enabled, current_prompt
    # パラメータ経由の値とグローバル変数の値を確認
    print(
        f"worker関数でのVAEキャッシュ設定: パラメータ={use_vae_cache}, グローバル変数={vae_cache_enabled}"
    )

    # キュー処理
    current_prompt = prompt
    current_image = input_image

    # グローバル変数の値を優先
    use_vae_cache = vae_cache_enabled or use_vae_cache

    # 入力画像のキーフレーム画像が存在するか確認
    print(translate("[DEBUG] worker内 input_imageの型: {0}").format(type(input_image)))
    if isinstance(input_image, str):
        print(
            translate("[DEBUG] input_imageはファイルパスです: {0}").format(input_image)
        )
        has_any_image = input_image is not None
    else:
        print(translate("[DEBUG] input_imageはファイルパス以外です"))
        has_any_image = input_image is not None

    if not has_any_image:
        # UIに直接エラーメッセージを表示
        stream.output_queue.push(
            (
                "progress",
                (
                    None,
                    translate(
                        "❗️ 画像が選択されていません\n生成を開始する前に「Image」欄または表示されている最後のキーフレーム画像に画像をアップロードしてください。これはあまねく叡智の始発点となる重要な画像です。"
                    ),
                    make_progress_bar_html(0, translate("エラー")),
                ),
            )
        )
        # 処理を終了
        stream.output_queue.push(("end", None))
        return

    # 入力画像チェック
    if input_image is None:
        print(translate("[ERROR] 入力画像は必須です"))
        raise Exception(translate("入力画像は必須です"))

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
            settings["output_folder"] = output_dir
            if save_settings(settings):
                output_folder_name = output_dir
                print(
                    translate("出力フォルダ設定を保存しました: {0}").format(output_dir)
                )
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

    # 現在のバッチ番号が指定されていれば使用する
    batch_suffix = f"_batch{batch_index + 1}" if batch_index is not None else ""
    job_id = generate_timestamp() + batch_suffix

    stream.output_queue.push(
        ("progress", (None, "", make_progress_bar_html(0, "Starting ...")))
    )

    try:
        # Clean GPU
        if not high_vram:
            # モデルをCPUにアンロード
            unload_complete_models(image_encoder, vae)

        # Text encoding

        stream.output_queue.push(
            (
                "progress",
                (None, "", make_progress_bar_html(0, translate("Text encoding ..."))),
            )
        )

        if not high_vram:
            fake_diffusers_current_device(
                text_encoder, gpu
            )  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        # プロンプトキューから選択されたプロンプトを使用
        # フラグが設定されていなくてもcurrent_promptを使うことで、
        # バッチ処理中で既にプロンプトが上書きされていた場合でも対応
        llama_vec, clip_l_pooler = encode_prompt_conds(
            current_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
        )

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = (
                torch.zeros_like(llama_vec),
                torch.zeros_like(clip_l_pooler),
            )
        else:
            # n_prompt
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                "", text_encoder, text_encoder_2, tokenizer, tokenizer_2
            )

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
            llama_vec_n, length=512
        )

        # これ以降の処理は text_encoder, text_encoder_2 は不要なので、メモリ解放してしまって構わない
        if not high_vram:
            text_encoder, text_encoder_2 = None, None
            text_encoder_manager.dispose_text_encoders()

        # テンソルデータのアップロードがあれば読み込み
        uploaded_tensor_latents = None
        if tensor_data_input is not None:
            try:
                tensor_path = tensor_data_input.name
                print(
                    translate("テンソルデータを読み込み: {0}").format(
                        os.path.basename(tensor_path)
                    )
                )
                stream.output_queue.push(
                    (
                        "progress",
                        (
                            None,
                            "",
                            make_progress_bar_html(
                                0, translate("Loading tensor data ...")
                            ),
                        ),
                    )
                )

                # safetensorsからテンソルを読み込み
                tensor_dict = sf.load_file(tensor_path)

                # テンソルに含まれているキーとシェイプを確認
                print(translate("テンソルデータの内容:"))
                for key, tensor in tensor_dict.items():
                    print(f"  - {key}: shape={tensor.shape}, dtype={tensor.dtype}")

                uploaded_tensor_edit_latents = None
                # history_latentsと呼ばれるキーが存在するか確認
                if "history_latents" in tensor_dict:
                    # テンソルデータからlatentデータを取得
                    uploaded_tensor_latents = tensor_dict["history_latents"]

                    preview_tensor = uploaded_tensor_latents
                    preview_tensor = vae_decode_fake(preview_tensor)

                    preview_tensor = (
                        (preview_tensor * 255.0)
                        .detach()
                        .cpu()
                        .numpy()
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                    preview_tensor = einops.rearrange(
                        preview_tensor, "b c t h w -> (b h) (t w) c"
                    )

                    desc = "テンソルデータを解析中です ..."
                    # テンソルデータのコマ画像
                    stream.output_queue.push(
                        (
                            "progress",
                            (
                                preview_tensor,
                                desc,
                                "",
                            ),
                        )
                    )

                    # テンソルデータの先頭フレームを削除する
                    uploaded_tensor_edit_latents = (
                        remove_first_frame_from_tensor_latent(
                            uploaded_tensor_latents, trim_start_latent_size
                        )
                    )

                    # input imageサイズ調整用
                    adjust_height = uploaded_tensor_latents.shape[3] * 8
                    adjust_width = uploaded_tensor_latents.shape[4] * 8
                else:
                    print(
                        translate(
                            "異常: テンソルデータに 'history_latents' キーが見つかりません"
                        )
                    )
            except Exception as e:
                print(translate("テンソルデータ読み込みエラー: {0}").format(e))
                import traceback

                traceback.print_exc()
                raise Exception("テンソルデータ読み込みエラー")

        stream.output_queue.push(
            (
                "progress",
                (
                    None,
                    "",
                    make_progress_bar_html(0, translate("Image processing ...")),
                ),
            )
        )

        def preprocess_image(img_path_or_array, resolution=640):
            """Pathまたは画像配列を処理して適切なサイズに変換する"""
            print(
                translate("[DEBUG] preprocess_image: img_path_or_array型 = {0}").format(
                    type(img_path_or_array)
                )
            )

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
                img = np.array(Image.open(img_path_or_array).convert("RGB"))
            else:
                # NumPy配列の場合はそのまま使う
                img = img_path_or_array

            # H, W, C = img.shape
            # 解像度パラメータを使用してサイズを決定
            # height, width = find_nearest_bucket(H, W, resolution=resolution)

            # 入力画像の解像度をテンソルファイルのサイズに合わせる
            img_np = resize_and_center_crop(
                img, target_width=adjust_width, target_height=adjust_height
            )
            img_pt = torch.from_numpy(img_np).float() / 127.5 - 1
            img_pt = img_pt.permute(2, 0, 1)[None, :, None]
            return img_np, img_pt, adjust_height, adjust_width

        input_image_np, input_image_pt, height, width = preprocess_image(current_image)
        Image.fromarray(input_image_np).save(
            os.path.join(outputs_folder, f"{job_id}.png")
        )
        # 入力画像にメタデータを埋め込んで保存
        initial_image_path = os.path.join(outputs_folder, f"{job_id}.png")
        Image.fromarray(input_image_np).save(initial_image_path)

        # メタデータの埋め込み
        # print(translate("\n[DEBUG] 入力画像へのメタデータ埋め込み開始: {0}").format(initial_image_path))
        # print(f"[DEBUG] prompt: {prompt}")
        # print(f"[DEBUG] seed: {seed}")
        metadata = {PROMPT_KEY: prompt, SEED_KEY: seed}
        # print(translate("[DEBUG] 埋め込むメタデータ: {0}").format(metadata))
        embed_metadata_to_png(initial_image_path, metadata)

        # VAE encoding

        stream.output_queue.push(
            (
                "progress",
                (None, "", make_progress_bar_html(0, translate("VAE encoding ..."))),
            )
        )

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        # UI上でテンソルデータの情報を表示
        tensor_info = translate(
            "テンソルデータ ({0}フレーム) を検出しました。動画生成後に結合します。"
        ).format(uploaded_tensor_latents.shape[2])
        stream.output_queue.push(
            (
                "progress",
                (
                    None,
                    tensor_info,
                    make_progress_bar_html(10, translate("テンソルデータを結合")),
                ),
            )
        )

        # 常に入力画像から通常のエンコーディングを行う
        input_image_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision

        stream.output_queue.push(
            (
                "progress",
                (
                    None,
                    "",
                    make_progress_bar_html(0, translate("CLIP Vision encoding ...")),
                ),
            )
        )

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(
            input_image_np, feature_extractor, image_encoder
        )
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(
            transformer.dtype
        )

        # Sampling

        stream.output_queue.push(
            (
                "progress",
                (None, "", make_progress_bar_html(0, translate("Start sampling ..."))),
            )
        )

        rnd = torch.Generator("cpu").manual_seed(seed)
        fix_uploaded_tensor_pixels = None

        # -------- LoRA 設定 START ---------

        # UI設定のuse_loraフラグ値を保存
        original_use_lora = use_lora
        print(f"[DEBUG] UI設定のuse_loraフラグの値: {original_use_lora}")

        # LoRAの環境変数設定（PYTORCH_CUDA_ALLOC_CONF）
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            old_env = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            print(
                translate(
                    "CUDA環境変数設定: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (元の値: {0})"
                ).format(old_env)
            )

        # 次回のtransformer設定を更新
        current_lora_paths = []
        current_lora_scales = []

        # ディレクトリモードのLoRA選択がある場合は強制的に有効にする
        if lora_mode == translate("ディレクトリから選択") and has_lora_support:
            # ディレクトリからドロップダウンで選択されたLoRAが1つでもあるか確認
            has_selected_lora = False
            for dropdown in [lora_dropdown1, lora_dropdown2, lora_dropdown3]:
                dropdown_value = (
                    dropdown.value if hasattr(dropdown, "value") else dropdown
                )

                # 通常の値が0や0.0などの数値の場合の特別処理（GradioのUIの問題によるもの）
                if (
                    dropdown_value == 0
                    or dropdown_value == "0"
                    or dropdown_value == 0.0
                ):
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
                print(
                    translate(
                        "[INFO] workerプロセス: ディレクトリでLoRAが選択されているため、LoRA使用を有効にしました"
                    )
                )

        # ファイルアップロードモードでLoRAファイルが選択されている場合も強制的に有効化
        elif not use_lora and has_lora_support:
            if (
                (lora_files is not None and hasattr(lora_files, "name"))
                or (lora_files2 is not None and hasattr(lora_files2, "name"))
                or (lora_files3 is not None and hasattr(lora_files3, "name"))
            ):
                use_lora = True
                print(
                    translate(
                        "[INFO] workerプロセス: LoRAファイルが選択されているため、LoRA使用を有効にしました"
                    )
                )

        if use_lora and has_lora_support:
            # LoRAの読み込み方式によって処理を分岐
            lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora")

            # UI状態からモードを検出 (lora_modeがgr.Radioオブジェクトの場合、valueプロパティを使用)
            # 重要: worker関数内では、直接引数として渡されたlora_modeの値を尊重する
            lora_mode_value = lora_mode
            if hasattr(lora_mode, "value"):
                try:
                    # valueプロパティが存在する場合は、それを使用
                    temp_value = lora_mode.value
                    if temp_value and isinstance(temp_value, str):
                        lora_mode_value = temp_value
                except:
                    # エラーが発生した場合は引数の値を直接使用
                    pass
            print(
                translate("[DEBUG] lora_mode_value 型: {0}, 値: {1}").format(
                    type(lora_mode_value).__name__, lora_mode_value
                )
            )

            # ドロップダウン値のデバッグ情報を出力
            print(translate("[DEBUG] worker内のLoRAドロップダウン値（元の値）:"))
            print(
                translate("  - lora_dropdown1: {0}, 型: {1}").format(
                    lora_dropdown1, type(lora_dropdown1).__name__
                )
            )
            print(
                translate("  - lora_dropdown2: {0}, 型: {1}").format(
                    lora_dropdown2, type(lora_dropdown2).__name__
                )
            )
            print(
                translate("  - lora_dropdown3: {0}, 型: {1}").format(
                    lora_dropdown3, type(lora_dropdown3).__name__
                )
            )
            if lora_mode_value and lora_mode_value == translate("ディレクトリから選択"):
                # ディレクトリから選択モード
                print(translate("[INFO] LoRA読み込み方式: ディレクトリから選択"))

                # ドロップダウンから選択されたファイルを処理
                # 渡されたままの値を保存
                # lora_dropdown2が数値0の場合、UI上で選択されたはずの値がおかしくなっている可能性あり
                if lora_dropdown2 == 0:
                    print(
                        translate(
                            "[DEBUG] lora_dropdown2が数値0になっています。特別処理を実行します"
                        )
                    )

                    # loraディレクトリ内の実際のファイルを確認（デバッグ用）
                    lora_dir = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "lora"
                    )
                    if os.path.exists(lora_dir):
                        lora_file_listing = []
                        for filename in os.listdir(lora_dir):
                            if filename.endswith((".safetensors", ".pt", ".bin")):
                                lora_file_listing.append(filename)
                        print(
                            translate(
                                "[DEBUG] ディレクトリ内LoRAファイル数: {0}"
                            ).format(len(lora_file_listing))
                        )

                original_dropdowns = {
                    "LoRA1": lora_dropdown1,
                    "LoRA2": lora_dropdown2,
                    "LoRA3": lora_dropdown3,
                }

                print(translate("[DEBUG] ドロップダウン詳細オリジナル値:"))
                print(
                    f"  lora_dropdown1 = {lora_dropdown1}, 型: {type(lora_dropdown1).__name__}"
                )
                print(
                    f"  lora_dropdown2 = {lora_dropdown2}, 型: {type(lora_dropdown2).__name__}"
                )
                print(
                    f"  lora_dropdown3 = {lora_dropdown3}, 型: {type(lora_dropdown3).__name__}"
                )

                # 渡されたドロップダウン値をそのまま使用する（Gradioオブジェクトを避けるため）
                # これは引数として渡された値をそのまま使うアプローチで、Gradioの複雑な内部構造と型の変換を回避

                # ドロップダウンの値の問題を特別に処理
                # 問題が起きやすい2番目の値に詳細ログを出力
                print(
                    translate("[DEBUG] 詳細ログ: LoRA2の値={0!r}, 型={1}").format(
                        lora_dropdown2, type(lora_dropdown2).__name__
                    )
                )

                # Gradioのバグ対応: 特に2番目の値が数値0になりやすい
                if lora_dropdown2 == 0:
                    print(
                        translate(
                            "[DEBUG] LoRA2の値が0になっているため、詳細な状態を確認"
                        )
                    )

                    # loraディレクトリの内容を確認
                    lora_dir = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "lora"
                    )
                    if os.path.exists(lora_dir):
                        print(translate("[DEBUG] LoRAディレクトリ内容:"))
                        directory_files = []
                        for filename in os.listdir(lora_dir):
                            if filename.endswith((".safetensors", ".pt", ".bin")):
                                directory_files.append(filename)

                        # 最初の何個かのファイルを表示
                        for i, file in enumerate(directory_files[:5]):
                            print(f"  {i + 1}. {file}")

                dropdown_direct_values = {
                    "dropdown1": original_dropdowns["LoRA1"],
                    "dropdown2": original_dropdowns["LoRA2"],
                    "dropdown3": original_dropdowns["LoRA3"],
                }

                # 各ドロップダウンを処理
                for (
                    dropdown_name,
                    dropdown_direct_value,
                ) in dropdown_direct_values.items():
                    # ドロップダウンの値を直接使用
                    print(
                        translate(
                            "[DEBUG] ドロップダウン{0}処理直接使用: 値={1}, 型={2}, 数値として表示={3}"
                        ).format(
                            dropdown_name,
                            repr(dropdown_direct_value),
                            type(dropdown_direct_value).__name__,
                            "0"
                            if dropdown_direct_value == 0
                            or dropdown_direct_value == 0.0
                            else "非0またはNone",
                        )
                    )

                    # 特に第2ドロップダウンの処理を強化（問題が最も頻繁に発生している場所）
                    if dropdown_name == "dropdown2" and dropdown_direct_value == 0:
                        print(
                            translate(
                                "[DEBUG] dropdown2の特別処理: 数値0が検出されました。元の値: {0}"
                            ).format(lora_dropdown2)
                        )
                        if (
                            isinstance(lora_dropdown2, str)
                            and lora_dropdown2 != "0"
                            and lora_dropdown2 != translate("なし")
                        ):
                            # 元の引数の値が文字列で、有効な値なら使用
                            dropdown_direct_value = lora_dropdown2
                            print(
                                translate(
                                    "[DEBUG] dropdown2の値を元の引数から復元: {0}"
                                ).format(dropdown_direct_value)
                            )

                    # 処理用の変数にコピー
                    dropdown_value = dropdown_direct_value

                    # 通常の値が0や0.0などの数値の場合の特別処理（GradioのUIの問題によるもの）
                    # 先に数値0かどうかをチェック（文字列変換前）
                    if (
                        dropdown_value == 0
                        or dropdown_value == 0.0
                        or dropdown_value == "0"
                    ):
                        # 数値の0を"なし"として扱う
                        print(
                            translate(
                                "[DEBUG] {name}の値が数値0として検出されました。'なし'として扱います"
                            ).format(name=dropdown_name)
                        )
                        dropdown_value = translate("なし")
                    # この段階で文字列変換を強制的に行う（Gradioの型が入り乱れる問題に対処）
                    elif dropdown_value is not None and not isinstance(
                        dropdown_value, str
                    ):
                        print(
                            translate(
                                "[DEBUG] {name}の前処理: 非文字列値が検出されたため文字列変換を実施: 値={1}, 型={2}"
                            ).format(
                                dropdown_name,
                                dropdown_value,
                                type(dropdown_value).__name__,
                            )
                        )
                        dropdown_value = str(dropdown_value)

                    # 最終的な型チェック - 万一文字列になっていない場合の保険
                    if dropdown_value is not None and not isinstance(
                        dropdown_value, str
                    ):
                        print(
                            translate(
                                "[DEBUG] {name}の値のタイプが依然として非文字列です: {type}"
                            ).format(
                                name=dropdown_name, type=type(dropdown_value).__name__
                            )
                        )
                        dropdown_value = str(dropdown_value)

                    # コード実行前の最終状態を記録
                    print(
                        translate(
                            "[DEBUG] {name}の最終値 (loading前): 値={value!r}, 型={type}, 'なし'と比較={is_none}"
                        ).format(
                            name=dropdown_name,
                            value=dropdown_value,
                            type=type(dropdown_value).__name__,
                            is_none="True"
                            if dropdown_value == translate("なし")
                            else "False",
                        )
                    )

                    if dropdown_value and dropdown_value != translate("なし"):
                        lora_path = os.path.join(lora_dir, dropdown_value)
                        print(
                            translate("[DEBUG] {name}のロード試行: パス={path}").format(
                                name=dropdown_name, path=lora_path
                            )
                        )
                        if os.path.exists(lora_path):
                            current_lora_paths.append(lora_path)
                            print(
                                translate("[INFO] {name}を選択: {path}").format(
                                    name=dropdown_name, path=lora_path
                                )
                            )
                        else:
                            # パスを修正して再試行（単なるファイル名の場合）
                            if os.path.dirname(
                                lora_path
                            ) == lora_dir and not os.path.isabs(dropdown_value):
                                # すでに正しく構築されているので再試行不要
                                print(
                                    translate(
                                        "[WARN] 選択された{name}が見つかりません: {file}"
                                    ).format(name=dropdown_name, file=dropdown_value)
                                )
                            else:
                                # 直接ファイル名だけで試行
                                lora_path_retry = os.path.join(
                                    lora_dir, os.path.basename(str(dropdown_value))
                                )
                                print(
                                    translate("[DEBUG] {name}を再試行: {path}").format(
                                        name=dropdown_name, path=lora_path_retry
                                    )
                                )
                                if os.path.exists(lora_path_retry):
                                    current_lora_paths.append(lora_path_retry)
                                    print(
                                        translate(
                                            "[INFO] {name}を選択 (パス修正後): {path}"
                                        ).format(
                                            name=dropdown_name, path=lora_path_retry
                                        )
                                    )
                                else:
                                    print(
                                        translate(
                                            "[WARN] 選択された{name}が見つかりません: {file}"
                                        ).format(
                                            name=dropdown_name, file=dropdown_value
                                        )
                                    )
            else:
                # ファイルアップロードモード
                print(translate("[INFO] LoRA読み込み方式: ファイルアップロード"))

                # 1つ目のLoRAファイルを処理
                if lora_files is not None:
                    # デバッグ情報を出力
                    print(
                        f"[DEBUG] lora_filesの型: {type(lora_files)}, 値: {lora_files}"
                    )

                    if isinstance(lora_files, list):
                        # 複数のLoRAファイル（将来のGradioバージョン用）
                        # 各ファイルがnameプロパティを持っているか確認
                        for file in lora_files:
                            if hasattr(file, "name") and file.name:
                                current_lora_paths.append(file.name)
                            else:
                                print(
                                    translate(
                                        "[WARN] LoRAファイル1のリスト内に無効なファイルがあります"
                                    )
                                )
                    else:
                        # 単一のLoRAファイル
                        # nameプロパティがあるか確認
                        if hasattr(lora_files, "name") and lora_files.name:
                            current_lora_paths.append(lora_files.name)
                        else:
                            print(
                                translate(
                                    "[WARN] 1つ目のLoRAファイルは無効か選択されていません"
                                )
                            )

                # 2つ目のLoRAファイルがあれば追加
                if lora_files2 is not None:
                    # デバッグ情報を出力
                    print(
                        f"[DEBUG] lora_files2の型: {type(lora_files2)}, 値: {lora_files2}"
                    )

                    if isinstance(lora_files2, list):
                        # 複数のLoRAファイル（将来のGradioバージョン用）
                        # 各ファイルがnameプロパティを持っているか確認
                        for file in lora_files2:
                            if hasattr(file, "name") and file.name:
                                current_lora_paths.append(file.name)
                            else:
                                print(
                                    translate(
                                        "[WARN] LoRAファイル2のリスト内に無効なファイルがあります"
                                    )
                                )
                    else:
                        # 単一のLoRAファイル
                        # nameプロパティがあるか確認
                        if hasattr(lora_files2, "name") and lora_files2.name:
                            current_lora_paths.append(lora_files2.name)
                        else:
                            print(
                                translate(
                                    "[WARN] 2つ目のLoRAファイルは無効か選択されていません"
                                )
                            )

                # 3つ目のLoRAファイルがあれば追加
                if lora_files3 is not None:
                    # デバッグ情報を出力
                    print(
                        f"[DEBUG] lora_files3の型: {type(lora_files3)}, 値: {lora_files3}"
                    )

                    if isinstance(lora_files3, list):
                        # 複数のLoRAファイル（将来のGradioバージョン用）
                        # 各ファイルがnameプロパティを持っているか確認
                        for file in lora_files3:
                            if hasattr(file, "name") and file.name:
                                current_lora_paths.append(file.name)
                            else:
                                print(
                                    translate(
                                        "[WARN] LoRAファイル3のリスト内に無効なファイルがあります"
                                    )
                                )
                    else:
                        # 単一のLoRAファイル
                        # nameプロパティがあるか確認
                        if hasattr(lora_files3, "name") and lora_files3.name:
                            current_lora_paths.append(lora_files3.name)
                        else:
                            print(
                                translate(
                                    "[WARN] 3つ目のLoRAファイルは無効か選択されていません"
                                )
                            )

            # スケール値をテキストから解析
            if current_lora_paths:  # LoRAパスがある場合のみ解析
                try:
                    scales_text = lora_scales_text.strip()
                    if scales_text:
                        # カンマ区切りのスケール値を解析
                        scales = [
                            float(scale.strip()) for scale in scales_text.split(",")
                        ]
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
                    current_lora_scales.extend(
                        [0.8] * (len(current_lora_paths) - len(current_lora_scales))
                    )
                elif len(current_lora_scales) > len(current_lora_paths):
                    # 余分は切り捨て
                    current_lora_scales = current_lora_scales[: len(current_lora_paths)]

        # UIでLoRA使用が有効になっていた場合、ファイル選択に関わらず強制的に有効化
        if original_use_lora:
            use_lora = True
            print(
                translate(
                    "[INFO] UIでLoRA使用が有効化されているため、LoRA使用を有効にします"
                )
            )

        print(f"[DEBUG] 最終的なuse_loraフラグ: {use_lora}")

        # LoRA設定を更新（リロードは行わない）
        transformer_manager.set_next_settings(
            lora_paths=current_lora_paths,
            lora_scales=current_lora_scales,
            fp8_enabled=fp8_optimization,  # fp8_enabledパラメータを追加
            high_vram_mode=high_vram,
            force_dict_split=True,  # 常に辞書分割処理を行う
        )

        # -------- LoRA 設定 END ---------

        # -------- FP8 設定 START ---------
        # FP8設定（既にLoRA設定に含めたので不要）
        # この行は削除しても問題ありません
        # -------- FP8 設定 END ---------

        print(translate("\ntransformer状態チェック..."))
        try:
            # transformerの状態を確認し、必要に応じてリロード
            if not transformer_manager.ensure_transformer_state():
                raise Exception(translate("transformer状態の確認に失敗しました"))

            # 最新のtransformerインスタンスを取得
            transformer = transformer_manager.get_transformer()
            print(translate("transformer状態チェック完了"))
        except Exception as e:
            print(translate("transformer状態チェックエラー: {0}").format(e))
            import traceback

            traceback.print_exc()
            raise e

        if stream.input_queue.top() == "end":
            stream.output_queue.push(("end", None))
            return

        combined_output_filename = None

        # テンソルデータを結合
        try:
            if not high_vram:
                load_model_as_complete(vae, target_device=gpu)

            # latentが小さすぎる場合に補間する
            fix_uploaded_tensor_latent = fix_tensor_size(uploaded_tensor_edit_latents)

            # テンソルデータフレームをpixels化
            uploaded_tensor_edit_pixels, _ = process_tensor_chunks(
                tensor=uploaded_tensor_edit_latents,
                frames=uploaded_tensor_edit_latents.shape[2],
                use_vae_cache=use_vae_cache,
                job_id=job_id,
                outputs_folder=outputs_folder,
                mp4_crf=mp4_crf,
                stream=stream,
                vae=vae,
            )

            if is_combine_mode(combine_mode, COMBINE_MODE.FIRST):
                # テンソルデータの先頭のフレーム
                uploaded_tensor_one_latent = fix_uploaded_tensor_latent[
                    :, :, 0, :, :
                ].clone()
            else:
                # テンソルデータの末尾のフレーム
                uploaded_tensor_one_latent = fix_uploaded_tensor_latent[
                    :, :, -1, :, :
                ].clone()
                # 2次元目を反転
                # fix_uploaded_tensor_latent = reorder_tensor(
                #     fix_uploaded_tensor_latent, True
                # )

            # 開始、終了画像
            first_image_latent = input_image_latent
            last_image_latent = uploaded_tensor_one_latent.unsqueeze(2)

            # 開始、終了画像を出力
            output_latent_to_image(
                first_image_latent,
                os.path.join(outputs_folder, f"{job_id}_generation_start.png"),
                vae,
                use_vae_cache,
            )
            output_latent_to_image(
                last_image_latent,
                os.path.join(outputs_folder, f"{job_id}_generation_end.png"),
                vae,
                use_vae_cache,
            )

            if not high_vram:
                unload_complete_models()
                # GPUメモリ保存値を明示的に浮動小数点に変換
                preserved_memory = (
                    float(gpu_memory_preservation)
                    if gpu_memory_preservation is not None
                    else 6.0
                )
                print(
                    translate(
                        "Setting transformer memory preservation to: {0} GB"
                    ).format(preserved_memory)
                )
                move_model_to_device_with_memory_preservation(
                    transformer, target_device=gpu, preserved_memory_gb=preserved_memory
                )

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            # アップロード（編集後）されたフレーム数
            uploaded_frames = uploaded_tensor_edit_latents.shape[2]

            print(
                translate("新規に動画を生成します。結合箇所: {direction}").format(
                    direction=get_combine_mode(combine_mode),
                )
            )

            # UI上で進捗状況を更新
            stream.output_queue.push(
                (
                    "progress",
                    (
                        None,
                        translate(
                            "新規に動画を生成します。結合箇所: {direction}"
                        ).format(
                            direction=get_combine_mode(combine_mode),
                        ),
                        make_progress_bar_html(80, translate("新規動画生成")),
                    ),
                )
            )

            def callback_generation(d):
                preview = d["denoised"]
                preview = vae_decode_fake(preview)

                preview = (
                    (preview * 255.0)
                    .detach()
                    .cpu()
                    .numpy()
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                preview = einops.rearrange(preview, "b c t h w -> (b h) (t w) c")

                if stream.input_queue.top() == "end":
                    stream.output_queue.push(("end", None))
                    raise KeyboardInterrupt("User ends the task.")

                current_step = d["i"] + 1
                percentage = int(100.0 * current_step / steps)
                hint = translate("Sampling {0}/{1}").format(current_step, steps)
                desc = "新規動画を生成中です ..."
                stream.output_queue.push(
                    (
                        "progress",
                        (
                            preview,
                            desc,
                            make_progress_bar_html(percentage, hint),
                        ),
                    )
                )
                return

            # EndFrame影響度設定（良い効果がなさそうなのでコメント）
            # last_image_latent = last_image_latent * end_frame_strength

            effective_window_size = generation_latent_size

            # 補間frames
            generation_num_frames = int(effective_window_size * 4 - 3)

            # indexとclean_latent
            indices = torch.arange(
                0, sum([1, 16, 2, 1, effective_window_size])
            ).unsqueeze(0)
            (
                clean_latent_indices_start,
                clean_latent_4x_indices,
                clean_latent_2x_indices,
                clean_latent_1x_indices,
                latent_indices,
            ) = indices.split([1, 16, 2, 1, effective_window_size], dim=1)
            clean_latent_indices = torch.cat(
                [clean_latent_indices_start, clean_latent_1x_indices], dim=1
            )
            clean_latents_4x, clean_latents_2x, clean_latents_1x = (
                fix_uploaded_tensor_latent[:, :, -sum([16, 2, 1]) :, :, :].split(
                    [16, 2, 1], dim=2
                )
            )
            first_image_latent = first_image_latent.to(clean_latents_4x.device)
            last_image_latent = last_image_latent.to(clean_latents_4x.device)
            clean_latents = torch.cat([first_image_latent, last_image_latent], dim=2)

            # フレームを生成
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler="unipc",
                width=width,
                height=height,
                frames=generation_num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,  # メインプロンプトを使用
                prompt_embeds_mask=llama_attention_mask,  # メインプロンプトのマスクを使用
                prompt_poolers=clip_l_pooler,  # メインプロンプトを使用
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
                callback=callback_generation,
            )

            # 生成データの配置
            device = uploaded_tensor_edit_latents.device
            generated_latents = generated_latents.to(device)

            if not high_vram:
                # 減圧時に使用するGPUメモリ値も明示的に浮動小数点に設定
                preserved_memory_offload = 8.0  # こちらは固定値のまま
                print(
                    translate(
                        "Offloading transformer with memory preservation: {0} GB"
                    ).format(preserved_memory_offload)
                )
                offload_model_from_device_for_memory_preservation(
                    transformer,
                    target_device=gpu,
                    preserved_memory_gb=preserved_memory_offload,
                )
                load_model_as_complete(vae, target_device=gpu)

            reverse = False
            # if is_combine_mode(combine_mode, COMBINE_MODE.LAST):
            #     reverse = True

            # 生成データをデコード
            generated_pixels, _ = process_tensor_chunks(
                tensor=generated_latents,
                frames=generated_latents.shape[2],
                use_vae_cache=use_vae_cache,
                job_id=job_id,
                outputs_folder=outputs_folder,
                mp4_crf=mp4_crf,
                stream=stream,
                vae=vae,
                reverse=reverse,
            )

            # 生成データが逆順なら戻す
            # generated_pixels = reorder_tensor(
            #     generated_pixels,
            #     is_combine_mode(combine_mode, COMBINE_MODE.LAST),
            # )

            # 生成データをmp4にして出力
            generation_output_filename = os.path.join(
                outputs_folder, f"{job_id}_generation.mp4"
            )
            save_bcthw_as_mp4(
                generated_pixels,
                generation_output_filename,
                fps=30,
                crf=mp4_crf,
            )
            print(
                translate(
                    "生成データの動画を保存しました: {generation_output_filename}"
                ).format(generation_output_filename=generation_output_filename)
            )

            generated_frames = generated_pixels.shape[2]

            ############################################
            # 生成データとテンソルデータを結合する
            ############################################

            print(
                translate(
                    "テンソルデータと生成データを結合します。結合箇所: {direction}, アップロードされたフレーム数 = {uploaded_frames}, 生成動画のフレーム数 = {generated_frames}"
                ).format(
                    direction=get_combine_mode(combine_mode),
                    uploaded_frames=uploaded_frames,
                    generated_frames=generated_frames,
                )
            )
            # UI上で進捗状況を更新
            stream.output_queue.push(
                (
                    "progress",
                    (
                        None,
                        translate(
                            "テンソルデータと生成データを結合します。結合箇所: {direction}, アップロードされたフレーム数 = {uploaded_frames}, 生成動画のフレーム数 = {generated_frames}"
                        ).format(
                            direction=get_combine_mode(combine_mode),
                            uploaded_frames=uploaded_frames,
                            generated_frames=generated_frames,
                        ),
                        make_progress_bar_html(90, translate("テンソルデータ結合準備")),
                    ),
                )
            )

            # 結合時のoverlapサイズ（設定するとしても小さめにする）
            overlapped_frames = 0
            # if overlapped_frames > generated_pixels.shape[2]:
            #     overlapped_frames = generated_pixels.shape[2]
            # if overlapped_frames > uploaded_tensor_edit_pixels.shape[2]:
            #     overlapped_frames = uploaded_tensor_edit_pixels.shape[2]

            # 結合
            if is_combine_mode(combine_mode, COMBINE_MODE.FIRST):
                combined_pixels = soft_append_bcthw(
                    generated_pixels.cpu(),
                    uploaded_tensor_edit_pixels.cpu(),
                    overlapped_frames,
                )
            else:
                combined_pixels = soft_append_bcthw(
                    uploaded_tensor_edit_pixels.cpu(),
                    generated_pixels.cpu(),
                    overlapped_frames,
                )
            del uploaded_tensor_edit_pixels
            del generated_pixels

            print(translate("新規生成データとテンソルデータのフレームを結合しました。"))

            print("combined_pixels: ", combined_pixels.shape)
            # 全チャンクの処理が完了したら、最終的な結合動画を保存
            if combined_pixels is not None:
                # 結合された最終結果の情報を出力
                print(translate("[DEBUG] 最終結合結果:"))
                print(translate("  - 形状: {0}").format(combined_pixels.shape))
                print(translate("  - 型: {0}").format(combined_pixels.dtype))
                print(translate("  - デバイス: {0}").format(combined_pixels.device))
                stream.output_queue.push(
                    (
                        "progress",
                        (
                            None,
                            translate("結合した動画をMP4に変換中..."),
                            make_progress_bar_html(95, translate("最終MP4変換処理")),
                        ),
                    )
                )

                # 最終的な結合ファイル名
                combined_output_filename = os.path.join(
                    outputs_folder, f"{job_id}_combined.mp4"
                )

                # MP4として保存
                save_bcthw_as_mp4(
                    combined_pixels,
                    combined_output_filename,
                    fps=30,
                    crf=mp4_crf,
                )
                print(
                    translate("最終結果を保存しました: {0}").format(
                        combined_output_filename
                    )
                )
                print(
                    translate("結合動画の保存場所: {0}").format(
                        os.path.abspath(combined_output_filename)
                    )
                )  # 中間ファイルの削除処理
                print(translate("中間ファイルの削除を開始します..."))
                deleted_files = []
                try:
                    import re

                    interim_pattern = re.compile(rf"{job_id}_combined_interim_\d+\.mp4")
                    deleted_count = 0

                    for filename in os.listdir(outputs_folder):
                        if interim_pattern.match(filename):
                            interim_path = os.path.join(outputs_folder, filename)
                            try:
                                os.remove(interim_path)
                                deleted_files.append(filename)
                                deleted_count += 1
                                print(
                                    translate(
                                        "  - 中間ファイルを削除しました: {0}"
                                    ).format(filename)
                                )
                            except Exception as e:
                                print(
                                    translate(
                                        "  - ファイル削除エラー ({0}): {1}"
                                    ).format(filename, str(e))
                                )

                    if deleted_count > 0:
                        print(
                            translate("中間ファイル {0} 個を削除しました").format(
                                deleted_count
                            )
                        )
                        files_str = ", ".join(deleted_files)
                        stream.output_queue.push(
                            (
                                "progress",
                                (
                                    None,
                                    translate("中間ファイルを削除しました: {0}").format(
                                        files_str
                                    ),
                                    make_progress_bar_html(
                                        97, translate("クリーンアップ完了")
                                    ),
                                ),
                            )
                        )
                    else:
                        print(translate("削除対象の中間ファイルは見つかりませんでした"))
                except Exception as e:
                    print(
                        translate(
                            "中間ファイル削除中にエラーが発生しました: {0}"
                        ).format(e)
                    )
                    import traceback

                    traceback.print_exc()

                # 結合した動画をUIに反映するため、出力フラグを立てる
                stream.output_queue.push(("file", combined_output_filename))

                # 結合後の全フレーム数を計算して表示
                combined_frames = combined_pixels.shape[2]
                combined_size_mb = (
                    combined_pixels.element_size() * combined_pixels.nelement()
                ) / (1024 * 1024)
                print(
                    translate(
                        "結合完了情報: テンソルデータ({0}フレーム) + 新規動画({1}フレーム) = 合計{2}フレーム"
                    ).format(uploaded_frames, generated_frames, combined_frames)
                )
                print(
                    translate("結合動画の再生時間: {0:.2f}秒").format(
                        combined_frames / 30
                    )
                )
                print(
                    translate("データサイズ: {0:.2f} MB（制限無し）").format(
                        combined_size_mb
                    )
                )

                # UI上で完了メッセージを表示
                stream.output_queue.push(
                    (
                        "progress",
                        (
                            None,
                            translate(
                                "テンソルデータ({0}フレーム)と動画({1}フレーム)の結合が完了しました。\n合計フレーム数: {2}フレーム ({3:.2f}秒) - サイズ制限なし"
                            ).format(
                                uploaded_frames,
                                generated_frames,
                                combined_frames,
                                combined_frames / 30,
                            ),
                            make_progress_bar_html(100, translate("結合完了")),
                        ),
                    )
                )
            else:
                print(translate("テンソルデータの結合に失敗しました。"))
                stream.output_queue.push(
                    (
                        "progress",
                        (
                            None,
                            translate("テンソルデータの結合に失敗しました。"),
                            make_progress_bar_html(100, translate("エラー")),
                        ),
                    )
                )

            # 結合した動画をUIに反映するため、出力フラグを立てる
            stream.output_queue.push(("file", combined_output_filename))

            # 結合後の全フレーム数を計算して表示
            combined_frames = combined_pixels.shape[2]
            combined_size_mb = (
                combined_pixels.element_size() * combined_pixels.nelement()
            ) / (1024 * 1024)
            print(
                translate(
                    "結合完了情報: テンソルデータ({0}フレーム) + 新規動画({1}フレーム) = 合計{2}フレーム"
                ).format(uploaded_frames, generated_frames, combined_frames)
            )
            print(
                translate("結合動画の再生時間: {0:.2f}秒").format(combined_frames / 30)
            )
            print(
                translate("データサイズ: {0:.2f} MB（制限無し）").format(
                    combined_size_mb
                )
            )

            # UI上で完了メッセージを表示
            stream.output_queue.push(
                (
                    "progress",
                    (
                        None,
                        translate(
                            "テンソルデータ({0}フレーム)と動画({1}フレーム)の結合が完了しました。\n合計フレーム数: {2}フレーム ({3:.2f}秒)"
                        ).format(
                            uploaded_frames,
                            generated_frames,
                            combined_frames,
                            combined_frames / 30,
                        ),
                        make_progress_bar_html(100, translate("結合完了")),
                    ),
                )
            )
        except Exception as e:
            print(
                translate("テンソルデータ結合中にエラーが発生しました: {0}").format(e)
            )
            import traceback

            traceback.print_exc()
            stream.output_queue.push(
                (
                    "progress",
                    (
                        None,
                        translate(
                            "エラー: テンソルデータ結合に失敗しました - {0}"
                        ).format(str(e)),
                        make_progress_bar_html(100, translate("エラー")),
                    ),
                )
            )

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
            print(
                translate(
                    "[MEMORY] 処理完了後のメモリクリア: {memory:.2f}GB/{total_memory:.2f}GB"
                ).format(
                    memory=torch.cuda.memory_allocated() / 1024**3,
                    total_memory=torch.cuda.get_device_properties(0).total_memory
                    / 1024**3,
                )
            )

        # テンソルデータの保存処理
        print(
            translate("[DEBUG] テンソルデータ保存フラグの値: {0}").format(
                save_tensor_data
            )
        )
        if save_tensor_data:
            try:
                # 結合
                combined_latents = torch.cat(
                    [generated_latents, uploaded_tensor_edit_latents], dim=2
                )

                # 結果のテンソルを保存するファイルパス
                tensor_file_path = os.path.join(outputs_folder, f"{job_id}.safetensors")

                # 保存するデータを準備
                print(translate("=== テンソルデータ保存処理開始 ==="))
                print(
                    translate("保存対象フレーム数: {frames}").format(
                        frames=combined_latents.shape[2]
                    )
                )

                # サイズ制限を完全に撤廃し、全フレームを保存
                combined_latents = combined_latents.cpu()

                # テンソルデータの保存サイズの概算
                tensor_size_mb = (
                    combined_latents.element_size() * combined_latents.nelement()
                ) / (1024 * 1024)

                print(
                    translate(
                        "テンソルデータを保存中... shape: {shape}, フレーム数: {frames}, サイズ: {size:.2f} MB"
                    ).format(
                        shape=combined_latents.shape,
                        frames=combined_latents.shape[2],
                        size=tensor_size_mb,
                    )
                )
                stream.output_queue.push(
                    (
                        "progress",
                        (
                            None,
                            translate(
                                "テンソルデータを保存中... ({frames}フレーム)"
                            ).format(frames=combined_latents.shape[2]),
                            make_progress_bar_html(
                                95, translate("テンソルデータの保存")
                            ),
                        ),
                    )
                )

                # メタデータの準備（フレーム数も含める）
                metadata = torch.tensor(
                    [height, width, combined_latents.shape[2]], dtype=torch.int32
                )

                # safetensors形式で保存
                tensor_dict = {
                    "history_latents": combined_latents,
                    "metadata": metadata,
                }
                sf.save_file(tensor_dict, tensor_file_path)

                print(
                    translate("テンソルデータを保存しました: {path}").format(
                        path=tensor_file_path
                    )
                )
                print(
                    translate(
                        "保存済みテンソルデータ情報: {frames}フレーム, {size:.2f} MB"
                    ).format(frames=combined_latents.shape[2], size=tensor_size_mb)
                )
                print(translate("=== テンソルデータ保存処理完了 ==="))
                stream.output_queue.push(
                    (
                        "progress",
                        (
                            None,
                            translate(
                                "テンソルデータが保存されました: {path} ({frames}フレーム, {size:.2f} MB)"
                            ).format(
                                path=os.path.basename(tensor_file_path),
                                frames=combined_latents.shape[2],
                                size=tensor_size_mb,
                            ),
                            make_progress_bar_html(100, translate("処理完了")),
                        ),
                    )
                )
            except Exception as e:
                print(translate("テンソルデータ保存エラー: {0}").format(e))
                import traceback

                traceback.print_exc()
                stream.output_queue.push(
                    (
                        "progress",
                        (
                            None,
                            translate("テンソルデータの保存中にエラーが発生しました。"),
                            make_progress_bar_html(100, translate("処理完了")),
                        ),
                    )
                )

        # 全体の処理時間を計算
        process_end_time = time.time()
        total_process_time = process_end_time - process_start_time
        hours, remainder = divmod(total_process_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = ""
        if hours > 0:
            time_str = translate("{0}時間 {1}分 {2}秒").format(
                int(hours), int(minutes), f"{seconds:.1f}"
            )
        elif minutes > 0:
            time_str = translate("{0}分 {1}秒").format(int(minutes), f"{seconds:.1f}")
        else:
            time_str = translate("{0:.1f}秒").format(seconds)
        print(translate("\n全体の処理時間: {0}").format(time_str))

        # 完了メッセージの設定
        # テンソル結合が成功した場合のメッセージ
        print(combined_output_filename)
        combined_filename_only = os.path.basename(combined_output_filename)
        completion_message = translate(
            "テンソルデータとの結合が完了しました。結合ファイル名: {filename}\n全体の処理時間: {time}"
        ).format(filename=combined_filename_only, time=time_str)
        # 最終的な出力ファイルを結合したものに変更
        output_filename = combined_output_filename

        stream.output_queue.push(
            (
                "progress",
                (
                    None,
                    completion_message,
                    make_progress_bar_html(100, translate("処理完了")),
                ),
            )
        )

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
                if (
                    file.startswith(job_id_part)
                    and file.endswith(".mp4")
                    and file != final_video_name
                    and "combined" not in file
                ):  # combinedファイルは保護
                    file_path = os.path.join(outputs_folder, file)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        print(translate("[削除] 中間ファイル: {0}").format(file))
                    except Exception as e:
                        print(
                            translate(
                                "[エラー] ファイル削除時のエラー {0}: {1}"
                            ).format(file, e)
                        )

            if deleted_count > 0:
                print(
                    translate(
                        "[済] {0}個の中間ファイルを削除しました。最終ファイルは保存されています: {1}"
                    ).format(deleted_count, final_video_name)
                )
                final_message = translate(
                    "中間ファイルを削除しました。最終動画と結合動画は保存されています。"
                )
                stream.output_queue.push(
                    (
                        "progress",
                        (
                            None,
                            final_message,
                            make_progress_bar_html(100, translate("処理完了")),
                        ),
                    )
                )

        if not high_vram:
            unload_complete_models()
    except:
        import traceback

        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(("end", None))
    return


# 画像のバリデーション関数
def validate_images(input_image, length_radio=None, frame_size_radio=None):
    """入力画像が有効かを確認する"""
    # 入力画像をチェック
    if input_image is not None:
        return True, ""

    # 画像がない場合はエラー
    error_html = f"""
    <div style="padding: 15px; border-radius: 10px; background-color: #ffebee; border: 1px solid #f44336; margin: 10px 0;">
        <h3 style="color: #d32f2f; margin: 0 0 10px 0;">{translate("❗️ 画像が選択されていません")}</h3>
        <p>{translate("生成を開始する前に「Image」欄に画像をアップロードしてください。これはあまねく叡智の始発点となる重要な画像です。")}</p>
    </div>
    """
    error_bar = make_progress_bar_html(100, translate("画像がありません"))
    return False, error_html + error_bar


def process(
    input_image,
    prompt,
    seed,
    steps,
    cfg,
    gs,
    rs,
    gpu_memory_preservation,
    use_teacache,
    mp4_crf=16,
    end_frame_strength=1.0,
    keep_section_videos=False,
    lora_files=None,
    lora_files2=None,
    lora_files3=None,
    lora_scales_text="0.8,0.8,0.8",
    output_dir=None,
    save_intermediate_frames=False,
    use_lora=False,
    lora_mode=None,
    lora_dropdown1=None,
    lora_dropdown2=None,
    lora_dropdown3=None,
    save_tensor_data=False,
    tensor_data_input=None,
    trim_start_latent_size=0,
    generation_latent_size=0,
    combine_mode=COMBINE_MODE_DEFAULT,
    fp8_optimization=False,
    batch_count=1,
    use_vae_cache=False,
):
    # プロセス関数の最初でVAEキャッシュ設定を確認
    print(
        f"process関数開始時のVAEキャッシュ設定: {use_vae_cache}, 型: {type(use_vae_cache)}"
    )
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

    # バッチ処理回数を確認し、詳細を出力
    # 型チェックしてから変換（数値でない場合はデフォルト値の1を使用）
    try:
        batch_count_val = int(batch_count)
        batch_count = max(1, min(batch_count_val, 100))  # 1〜100の間に制限
    except (ValueError, TypeError):
        print(
            translate(
                "[WARN] バッチ処理回数が無効です。デフォルト値の1を使用します: {0}"
            ).format(batch_count)
        )
        batch_count = 1  # デフォルト値

    print(translate("\u25c6 バッチ処理回数: {0}回").format(batch_count))

    # TODO:解像度は一旦無視
    # 解像度を安全な値に丸めてログ表示
    from diffusers_helper.bucket_tools import SAFE_RESOLUTIONS

    # 解像度値を表示
    resolution = 640

    # 解像度設定を出力
    print(translate("解像度を設定: {0}").format(resolution))

    # 動画生成の設定情報をログに出力
    # 4.5の場合は5として計算するための特別処理

    print(translate("\n==== 動画生成開始 ====="))
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
    print(
        f"VAEキャッシュ詳細状態: use_vae_cache={use_vae_cache}, type={type(use_vae_cache)}"
    )

    # LoRA情報のログ出力とLoRAモード判定
    # LoRAモードがディレクトリ選択で、ドロップダウンに値が選択されている場合は使用フラグを上書き
    if lora_mode == translate("ディレクトリから選択") and has_lora_support:
        # ディレクトリからドロップダウンで選択されたLoRAが1つでもあるか確認
        has_selected_lora = False
        for dropdown in [lora_dropdown1, lora_dropdown2, lora_dropdown3]:
            dropdown_value = dropdown.value if hasattr(dropdown, "value") else dropdown
            if dropdown_value and dropdown_value != translate("なし"):
                has_selected_lora = True
                break

        # LoRA選択があれば強制的に有効にする
        if has_selected_lora:
            use_lora = True
            print(
                translate(
                    "[INFO] ディレクトリでLoRAが選択されているため、LoRA使用を有効にしました"
                )
            )

    if use_lora and has_lora_support:
        all_lora_files = []
        lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora")

        # UI状態からモードを検出 (lora_modeがgr.Radioオブジェクトの場合、valueプロパティを使用)
        lora_mode_value = lora_mode.value if hasattr(lora_mode, "value") else lora_mode
        if lora_mode_value and lora_mode_value == translate("ディレクトリから選択"):
            # ディレクトリから選択モード
            print(translate("[INFO] LoRA読み込み方式: ディレクトリから選択"))

            # ドロップダウンから選択されたファイルを処理 (ログ出力用)
            selected_lora_names = []

            # 各ドロップダウンを確認
            for dropdown, dropdown_name in [
                (lora_dropdown1, "LoRA1"),
                (lora_dropdown2, "LoRA2"),
                (lora_dropdown3, "LoRA3"),
            ]:
                # ドロップダウンの値を取得（gr.Dropdownオブジェクトの場合はvalueプロパティを使用）
                dropdown_value = (
                    dropdown.value if hasattr(dropdown, "value") else dropdown
                )

                # 通常の値が0や0.0などの数値の場合の特別処理（GradioのUIの問題によるもの）
                if (
                    dropdown_value == 0
                    or dropdown_value == "0"
                    or dropdown_value == 0.0
                ):
                    # 数値の0を"なし"として扱う
                    print(
                        translate(
                            "[DEBUG] 情報表示: {name}の値が数値0として検出されました。'なし'として扱います"
                        ).format(name=dropdown_name)
                    )
                    dropdown_value = translate("なし")

                # 型チェックと文字列変換を追加
                if not isinstance(dropdown_value, str) and dropdown_value is not None:
                    print(
                        translate(
                            "[DEBUG] 情報表示: {name}の値のタイプ変換が必要: {type}"
                        ).format(name=dropdown_name, type=type(dropdown_value).__name__)
                    )
                    dropdown_value = str(dropdown_value)

                if dropdown_value and dropdown_value != translate("なし"):
                    lora_path = os.path.join(lora_dir, dropdown_value)
                    # よりわかりやすい表記に
                    model_name = f"LoRA{dropdown_name[-1]}: {dropdown_value}"
                    selected_lora_names.append(model_name)

            # 選択されたLoRAモデルの情報出力を明確に
            if selected_lora_names:
                print(
                    translate("[INFO] 選択されたLoRAモデル: {0}").format(
                        ", ".join(selected_lora_names)
                    )
                )
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
                            print(
                                translate(
                                    "[WARN] LoRAファイル1のリスト内に無効なファイルがあります"
                                )
                            )
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
                            print(
                                translate(
                                    "[WARN] LoRAファイル2のリスト内に無効なファイルがあります"
                                )
                            )
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
                            print(
                                translate(
                                    "[WARN] LoRAファイル3のリスト内に無効なファイルがあります"
                                )
                            )
                elif lora_files3 is not None:
                    all_lora_files.append(lora_files3)

        # スケール値を解析
        try:
            scales = [float(s.strip()) for s in lora_scales_text.split(",")]
        except:
            # 解析エラーの場合はデフォルト値を使用
            scales = [0.8] * len(all_lora_files)

        # スケール値の数を調整
        if len(scales) < len(all_lora_files):
            scales.extend([0.8] * (len(all_lora_files) - len(scales)))
        elif len(scales) > len(all_lora_files):
            scales = scales[: len(all_lora_files)]

        # LoRAファイル情報を出力
        if len(all_lora_files) == 1:
            # 単一ファイル
            print(
                translate("\u25c6 LoRAファイル: {0}").format(
                    os.path.basename(all_lora_files[0].name)
                )
            )
            print(translate("\u25c6 LoRA適用強度: {0}").format(scales[0]))
        elif len(all_lora_files) > 1:
            # 複数ファイル
            print(translate("\u25c6 LoRAファイル (複数):"))
            for i, file in enumerate(all_lora_files):
                print(f"   - {os.path.basename(file.name)} (スケール: {scales[i]})")
        else:
            # LoRAファイルなし
            print(translate("\u25c6 LoRA: 使用しない"))

    print("=============================\n")

    # バッチ処理の全体停止用フラグ
    batch_stopped = False

    # 元のシード値を保存（バッチ処理用）
    original_seed = seed

    # ランダムシード状態をデバッグ表示
    print(
        f"[DEBUG] use_random_seed: {use_random_seed}, タイプ: {type(use_random_seed)}"
    )

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
        print(
            translate(
                "\n[INFO] ランダムシード機能が有効なため、指定されたSEED値 {0} の代わりに新しいSEED値 {1} を使用します。"
            ).format(previous_seed, seed)
        )
        # UIのseed欄もランダム値で更新
        yield (
            None,
            None,
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(value=seed),
        )
        # ランダムシードの場合は最初の値を更新
        original_seed = seed
    else:
        print(translate("[INFO] 指定されたSEED値 {0} を使用します。").format(seed))
        yield (
            None,
            None,
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(),
        )

    stream = AsyncStream()

    # stream作成後、バッチ処理前もう一度フラグを確認
    if batch_stopped:
        print(translate("\nバッチ処理が中断されました（バッチ開始前）"))
        yield (
            None,
            gr.update(visible=False),
            translate("バッチ処理が中断されました"),
            "",
            gr.update(interactive=True),
            gr.update(interactive=False, value=translate("End Generation")),
            gr.update(),
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
                "",
                gr.update(interactive=True),
                gr.update(interactive=False, value=translate("End Generation")),
                gr.update(),
            )
            break

        # 現在のバッチ番号を表示
        if batch_count > 1:
            batch_info = translate("バッチ処理: {0}/{1}").format(
                batch_index + 1, batch_count
            )

            print(f"\n{batch_info}")
            # UIにもバッチ情報を表示
            yield (
                None,
                gr.update(visible=False),
                batch_info,
                "",
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(),
            )

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
                    print(
                        translate(
                            "[INFO] バッチ {0}/{1} の処理を開始: SEED値を {2} に設定しました。"
                        ).format(batch_index + 1, batch_count, seed)
                    )
            else:
                # ランダムシード使用時は各バッチで新しい値を生成
                if batch_index > 0:  # 最初のバッチ以外は新しい値を生成
                    prev_seed = seed
                    seed = random.randint(0, 2**32 - 1)
                    print(
                        translate(
                            "[INFO] バッチ {0}/{1} の処理を開始: 新しいランダムSEED値 {2} を生成しました。"
                        ).format(batch_index + 1, batch_count, seed)
                    )

        # 常に現在のシード値を表示（バッチ数に関わらず）
        print(translate("現在のSEED値: {0}").format(seed))

        # もう一度停止フラグを確認 - worker処理実行前
        if batch_stopped:
            print(
                translate(
                    "バッチ処理が中断されました。worker関数の実行をキャンセルします。"
                )
            )
            # 中断メッセージをUIに表示
            yield (
                None,
                gr.update(visible=False),
                translate("バッチ処理が中断されました（{0}/{1}）").format(
                    batch_index, batch_count
                ),
                "",
                gr.update(interactive=True),
                gr.update(interactive=False, value=translate("End Generation")),
                gr.update(),
            )
            break

        # GPUメモリの設定値をデバッグ出力し、正しい型に変換
        gpu_memory_value = (
            float(gpu_memory_preservation)
            if gpu_memory_preservation is not None
            else 6.0
        )
        print(
            translate("Using GPU memory preservation setting: {0} GB").format(
                gpu_memory_value
            )
        )

        # 出力フォルダが空の場合はデフォルト値を使用
        if not output_dir or not output_dir.strip():
            output_dir = "outputs"
        print(translate("Output directory: {0}").format(output_dir))

        # 先に入力データの状態をログ出力（デバッグ用）
        if input_image is not None:
            if isinstance(input_image, str):
                print(
                    translate("[DEBUG] input_image path: {0}, type: {1}").format(
                        input_image, type(input_image)
                    )
                )
            else:
                print(
                    translate("[DEBUG] input_image shape: {0}, type: {1}").format(
                        input_image.shape, type(input_image)
                    )
                )

        # バッチ処理の各回で実行
        # worker関数の定義と引数の順序を完全に一致させる
        print(
            translate("[DEBUG] async_run直前のsave_tensor_data: {0}").format(
                save_tensor_data
            )
        )
        print(translate("[DEBUG] async_run直前のLoRA関連パラメータ:"))
        print(
            translate("  - lora_mode: {0}, 型: {1}").format(
                lora_mode, type(lora_mode).__name__
            )
        )
        print(
            translate("  - lora_dropdown1: {0!r}, 型: {1}").format(
                lora_dropdown1, type(lora_dropdown1).__name__
            )
        )
        print(
            translate("  - lora_dropdown2: {0!r}, 型: {1}").format(
                lora_dropdown2, type(lora_dropdown2).__name__
            )
        )
        print(
            translate("  - lora_dropdown3: {0!r}, 型: {1}").format(
                lora_dropdown3, type(lora_dropdown3).__name__
            )
        )
        print(
            translate("  - use_lora: {0}, 型: {1}").format(
                use_lora, type(use_lora).__name__
            )
        )

        # 特に2番目のドロップダウン値が正しく扱われていない場合のデバッグ情報を出力
        # この段階では値を変更せず、情報収集のみ行う
        if lora_mode == translate("ディレクトリから選択") and lora_dropdown2 == 0:
            print(translate("[DEBUG] lora_dropdown2が数値0になっています"))

            # ディレクトリ情報を出力（デバッグ用）
            lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora")
            if os.path.exists(lora_dir):
                print(translate("[DEBUG] LoRAディレクトリ内ファイル:"))
                for filename in os.listdir(lora_dir):
                    if filename.endswith((".safetensors", ".pt", ".bin")):
                        print(f"  - {filename}")

        # イメージキューが有効な場合、バッチに応じた入力画像を設定
        current_input_image = input_image  # デフォルトでは元の入力画像

        # 入力画像の情報をログに出力
        if isinstance(current_input_image, str):
            print(translate("処理用入力画像: {0}").format(current_input_image))

        async_run(
            worker,
            current_input_image,  # イメージキューで変更された可能性がある入力画像
            prompt,
            seed,
            steps,
            cfg,
            gs,
            rs,
            gpu_memory_value,  # gpu_memory_preservation
            use_teacache,
            mp4_crf,
            end_frame_strength,
            keep_section_videos,
            lora_files,
            lora_files2,
            lora_files3,  # 追加：3つ目のLoRAファイル
            lora_scales_text,
            output_dir,
            save_intermediate_frames,
            use_lora,
            lora_mode,  # 追加：LoRAモード設定
            lora_dropdown1,  # 追加：LoRAドロップダウン1
            lora_dropdown2,  # 追加：LoRAドロップダウン2
            lora_dropdown3,  # 追加：LoRAドロップダウン3
            save_tensor_data,  # テンソルデータ保存フラグ - 確実に正しい位置に配置
            tensor_data_input,
            trim_start_latent_size,
            generation_latent_size,
            combine_mode,
            fp8_optimization,
            batch_index,
            use_vae_cache,  # VAEキャッシュ設定
        )

        # 現在のバッチの出力ファイル名
        batch_output_filename = None

        # 現在のバッチの処理結果を取得
        while True:
            flag, data = stream.output_queue.next()

            if flag == "file":
                batch_output_filename = data
                # より明確な更新方法を使用し、preview_imageを明示的にクリア
                yield (
                    batch_output_filename,
                    gr.update(value=None, visible=False),
                    gr.update(),
                    gr.update(),
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    gr.update(),
                )

            if flag == "progress":
                preview, desc, html = data
                # バッチ処理中は現在のバッチ情報を追加
                # 単一バッチでも常にSEED値情報を表示する
                batch_info = ""
                if batch_count > 1:
                    batch_info = translate("バッチ処理: {0}/{1} - ").format(
                        batch_index + 1, batch_count
                    )

                # 現在のSEED値を常に表示
                current_seed_info = translate("現在のSEED値: {0}").format(seed)
                if batch_info:
                    desc = batch_info + desc

                # プロンプトの末尾に現在のシード値の情報を追加（バッチ処理数に関わらず）
                if current_seed_info not in desc:
                    desc = desc + "\n\n" + current_seed_info
                # preview_imageを明示的に設定
                yield (
                    gr.update(),
                    gr.update(visible=True, value=preview),
                    desc,
                    html,
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    gr.update(),
                )

            if flag == "end":
                # このバッチの処理が終了
                if batch_index == batch_count - 1 or batch_stopped:
                    # 最終バッチの場合は処理完了を通知
                    completion_message = ""
                    if batch_stopped:
                        completion_message = translate(
                            "バッチ処理が中止されました（{0}/{1}）"
                        ).format(batch_index + 1, batch_count)
                    else:
                        completion_message = translate(
                            "バッチ処理が完了しました（{0}/{1}）"
                        ).format(batch_count, batch_count)
                    yield (
                        batch_output_filename,
                        gr.update(value=None, visible=False),
                        completion_message,
                        "",
                        gr.update(interactive=True),
                        gr.update(interactive=False, value=translate("End Generation")),
                        gr.update(),
                    )
                else:
                    # 次のバッチに進むメッセージを表示
                    next_batch_message = translate(
                        "バッチ処理: {0}/{1} 完了、次のバッチに進みます..."
                    ).format(batch_index + 1, batch_count)
                    yield (
                        batch_output_filename,
                        gr.update(value=None, visible=False),
                        next_batch_message,
                        "",
                        gr.update(interactive=False),
                        gr.update(interactive=True),
                        gr.update(),
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
    stream.input_queue.push("end")

    # ボタンの名前を一時的に変更することでユーザーに停止処理が進行中であることを表示
    return gr.update(value=translate("停止処理中..."))


# 既存のQuick Prompts（初期化時にプリセットに変換されるので、互換性のために残す）
quick_prompts = [
    "A character doing some simple body movements.",
    "A character uses expressive hand gestures and body language.",
    "A character walks leisurely with relaxed movements.",
    "A character performs dynamic movements with energy and flowing motion.",
    "A character moves in unexpected ways, with surprising transitions poses.",
]
quick_prompts = [[x] for x in quick_prompts]

css = get_app_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.HTML(
        '<h1>FramePack<span class="title-suffix">-eichi</span> tensor tool -F1</h1>'
    )

    with gr.Tab("Generation"):
        gr.Markdown(
            translate(
                "**eichi等で生成した（history_latentsを持つ）テンソルデータを起点とした生成ツールです。テンソルデータの後ろに生成した動画を継ぎ足します。**"
            )
        )

        # 入力・出力
        with gr.Row():
            # 入力
            with gr.Column():
                # テンソルデータ
                with gr.Group():
                    gr.Markdown(
                        "### " + translate("入力：テンソルデータ"),
                        elem_classes="markdown-title",
                    )
                    with gr.Row():
                        # テンソルデータファイルのアップロード
                        tensor_data_input = gr.File(
                            label=translate(
                                "テンソルデータアップロード (.safetensors) - eichiにて生成したファイル。生成動画の前方（先頭）または後方（末尾）に結合されます"
                            ),
                            file_types=[".safetensors"],
                            type="filepath",
                        )

                    with gr.Row():
                        # テンソルデータ読み込み状態
                        preview_tensor_desc = gr.Markdown(
                            "", elem_classes="markdown-desc"
                        )

                # 生成動画
                with gr.Group():
                    gr.Markdown(
                        "### " + translate("入力：動画生成で使用する画像、プロンプト"),
                        elem_classes="markdown-title",
                    )
                    with gr.Row():
                        # 生成フレーム数
                        generation_latent_size = gr.Slider(
                            label=translate("生成フレーム数"),
                            minimum=1,
                            maximum=12,
                            value=9,
                            step=1,
                            interactive=True,
                            info=translate(
                                "新規で生成するフレーム数。6-9推奨。\n設定値と時間の目安：3（0.3秒）、6（0.7秒）、9（1.1秒）、12（1.5秒）"
                            ),
                        )
                    with gr.Row():
                        # 初期フレーム用のState変数
                        input_image_state = gr.State(
                            None
                        )  # 開始フレーム画像のパスを保持

                        # 入力画像
                        input_image = gr.Image(
                            label=translate(
                                "Image - テンソルデータの前方（先頭）または後方（末尾）の動画を生成するための方向性を示す画像"
                            ),
                            sources=["upload", "clipboard"],
                            type="filepath",
                            height=320,
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
                            outputs=[input_image_state],
                        )

                        # メタデータ抽出関数を定義（後で登録する）
                        def update_from_image_metadata(image_path, copy_enabled=False):
                            """Imageアップロード時にメタデータを抽出してUIに反映する
                            copy_enabled: メタデータの複写が有効化されているかどうか
                            """
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
                                    print(
                                        translate(
                                            "アップロードされた画像にメタデータが含まれていません"
                                        )
                                    )
                                    return [gr.update()] * 2

                                # print(translate("[DEBUG] メタデータサイズ: {0}, 内容: {1}").format(len(metadata), metadata))
                                print(
                                    translate(
                                        "画像からメタデータを抽出しました: {0}"
                                    ).format(metadata)
                                )

                                # プロンプトとSEEDをUIに反映
                                prompt_update = gr.update()
                                seed_update = gr.update()

                                if PROMPT_KEY in metadata and metadata[PROMPT_KEY]:
                                    prompt_update = gr.update(
                                        value=metadata[PROMPT_KEY]
                                    )
                                    # print(translate("[DEBUG] プロンプトを更新: {0}").format(metadata[PROMPT_KEY]))
                                    print(
                                        translate(
                                            "プロンプトを画像から取得: {0}"
                                        ).format(metadata[PROMPT_KEY])
                                    )

                                if SEED_KEY in metadata and metadata[SEED_KEY]:
                                    # SEED値を整数に変換
                                    try:
                                        seed_value = int(metadata[SEED_KEY])
                                        seed_update = gr.update(value=seed_value)
                                        # print(translate("[DEBUG] SEED値を更新: {0}").format(seed_value))
                                        print(
                                            translate(
                                                "SEED値を画像から取得: {0}"
                                            ).format(seed_value)
                                        )
                                    except (ValueError, TypeError):
                                        # print(translate("[DEBUG] SEED値の変換エラー: {0}").format(metadata[SEED_KEY]))
                                        print(
                                            translate("SEED値の変換エラー: {0}").format(
                                                metadata[SEED_KEY]
                                            )
                                        )

                                # print(translate("[DEBUG] 更新結果: prompt_update={0}, seed_update={1}").format(prompt_update, seed_update))
                                return [prompt_update, seed_update]
                            except Exception as e:
                                # print(translate("[ERROR] メタデータ抽出処理中のエラー: {0}").format(e))
                                # traceback.print_exc()
                                print(translate("メタデータ抽出エラー: {0}").format(e))
                                return [gr.update()] * 2

                    # プロンプト
                    # metadata
                    with gr.Row(variant="compact"):
                        # 埋め込みプロンプトおよびシードを複写するチェックボックスの定義
                        # 参照先として必要だが、表示はLoRA設定の下で行うため非表示に設定
                        global copy_metadata
                        copy_metadata = gr.Checkbox(
                            label=translate("埋め込みプロンプトおよびシードを複写する"),
                            value=False,
                            info=translate(
                                "チェックをオンにすると、画像のメタデータからプロンプトとシードを自動的に取得します"
                            ),
                            visible=False,  # 元の位置では非表示
                        )

                        # 埋め込みプロンプトおよびシードを複写するチェックボックス（LoRA設定の下に表示）
                        copy_metadata_visible = gr.Checkbox(
                            label=translate("埋め込みプロンプトおよびシードを複写する"),
                            value=False,
                            info=translate(
                                "チェックをオンにすると、画像のメタデータからプロンプトとシードを自動的に取得します"
                            ),
                        )

                        # 表示用チェックボックスと実際の処理用チェックボックスを同期
                        copy_metadata_visible.change(
                            fn=lambda x: x,
                            inputs=[copy_metadata_visible],
                            outputs=[copy_metadata],
                        )

                        # 元のチェックボックスが変更されたときも表示用を同期
                        copy_metadata.change(
                            fn=lambda x: x,
                            inputs=[copy_metadata],
                            outputs=[copy_metadata_visible],
                            queue=False,  # 高速化のためキューをスキップ
                        )

                    # プロンプト
                    with gr.Row():
                        prompt = gr.Textbox(
                            label=translate(
                                "Prompt - 生成する動画の動きを指示するプロンプト"
                            ),
                            value=get_default_startup_prompt(),
                            lines=6,
                        )

                # 結合設定
                with gr.Group():
                    gr.Markdown(
                        "### "
                        + translate("生成動画をテンソルデータの前後どちらに結合するか"),
                        elem_classes="markdown-title",
                    )
                    with gr.Row():
                        combine_mode = gr.Radio(
                            choices=COMBINE_MODE_OPTIONS_KEYS,
                            value=COMBINE_MODE_DEFAULT,
                            label=translate("生成動画の結合箇所"),
                            interactive=True,
                        )

                    with gr.Row():
                        # テンソルデータの先頭の削除フレーム数
                        trim_start_latent_size = gr.Slider(
                            label=translate("テンソルデータの先頭の削除フレーム数"),
                            minimum=0,
                            maximum=5,
                            value=0,
                            step=1,
                            interactive=False,  # 後方のみにしているので操作不可
                            visible=True,
                            info=translate(
                                "テンソルデータの先頭から削除するフレーム数。用途：生成動画をテンソル動画の先頭に結合するケースで、テンソルデータの先頭部分にノイズがあるとわかっている際に、設定してください。出力動画の品質を確認して長さを調整してください。"
                            ),
                        )

                # バッチ設定
                with gr.Group():
                    gr.Markdown(
                        "### " + translate("バッチ設定"),
                        elem_classes="markdown-title",
                    )
                    with gr.Row():
                        # バッチ処理回数
                        batch_count = gr.Slider(
                            label=translate("バッチ処理回数"),
                            minimum=1,
                            maximum=100,
                            value=1,
                            step=1,
                            info=translate(
                                "同じ設定で連続生成する回数。SEEDは各回で+1されます"
                            ),
                        )

                # 開始・終了
                with gr.Row():
                    # 開始・終了ボタン
                    start_button = gr.Button(
                        value=translate("Start Generation"),
                        variant="primary",
                        interactive=False,
                    )
                    end_button = gr.Button(
                        value=translate("End Generation"), interactive=False
                    )

                # 詳細設定
                with gr.Group():
                    gr.Markdown(
                        "### " + translate("詳細設定"),
                        elem_classes="markdown-title",
                    )
                    with gr.Accordion(
                        "",
                        open=False,
                        elem_classes="section-accordion",
                    ):
                        # Use Random Seedの初期値
                        use_random_seed_default = True
                        seed_default = (
                            random.randint(0, 2**32 - 1)
                            if use_random_seed_default
                            else 1
                        )

                        with gr.Row(variant="compact"):
                            use_random_seed = gr.Checkbox(
                                label=translate("Use Random Seed"),
                                value=use_random_seed_default,
                            )
                            seed = gr.Number(
                                label=translate("Seed"), value=seed_default, precision=0
                            )

                        def set_random_seed(is_checked):
                            if is_checked:
                                return random.randint(0, 2**32 - 1)
                            else:
                                return gr.update()

                        use_random_seed.change(
                            fn=set_random_seed, inputs=use_random_seed, outputs=seed
                        )

                        steps = gr.Slider(
                            label=translate("Steps"),
                            minimum=1,
                            maximum=100,
                            value=25,
                            step=1,
                            info=translate("Changing this value is not recommended."),
                        )

                        cfg = gr.Slider(
                            label=translate("CFG Scale"),
                            minimum=1.0,
                            maximum=32.0,
                            value=1.0,
                            step=0.01,
                            visible=False,
                        )  # Should not change

                        gs = gr.Slider(
                            label=translate("Distilled CFG Scale"),
                            minimum=1.0,
                            maximum=32.0,
                            value=10.0,
                            step=0.01,
                            info=translate("Changing this value is not recommended."),
                        )

                        rs = gr.Slider(
                            label=translate("CFG Re-Scale"),
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.01,
                            visible=False,
                        )  # Should not change

                        # EndFrame影響度
                        end_frame_strength = gr.Slider(
                            label=translate("EndFrame影響度"),
                            minimum=0.01,
                            maximum=1.00,
                            value=1.00,
                            step=0.01,
                            info=translate(
                                "最終フレームが動画全体に与える影響の強さを調整します。値を小さくすると最終フレームの影響が弱まり、最初のフレームに早く移行します。1.00が通常の動作です。"
                            ),
                        )

                # LoRA設定
                with gr.Group():
                    with gr.Row():
                        # LoRA設定
                        with gr.Group(visible=has_lora_support) as lora_settings_group:
                            gr.Markdown(
                                "### " + translate("LoRA設定"),
                                elem_classes="markdown-title",
                            )

                            # LoRA使用有無のチェックボックス
                            use_lora = gr.Checkbox(
                                label=translate("LoRAを使用する"),
                                value=False,
                                info=translate(
                                    "チェックをオンにするとLoRAを使用します（要16GB VRAM以上）"
                                ),
                            )

                            # LoRA読み込み方式を選択するラジオボタン
                            lora_mode = gr.Radio(
                                choices=[
                                    translate("ディレクトリから選択"),
                                    translate("ファイルアップロード"),
                                ],
                                value=translate("ディレクトリから選択"),
                                label=translate("LoRA読み込み方式"),
                                visible=False,  # 初期状態では非表示
                            )

                            # ファイルアップロード方式のコンポーネント（グループ化）
                            with gr.Group(visible=False) as lora_upload_group:
                                # メインのLoRAファイル
                                lora_files = gr.File(
                                    label=translate(
                                        "LoRAファイル (.safetensors, .pt, .bin)"
                                    ),
                                    file_types=[".safetensors", ".pt", ".bin"],
                                )
                                # 追加のLoRAファイル
                                lora_files2 = gr.File(
                                    label=translate(
                                        "LoRAファイル2 (.safetensors, .pt, .bin)"
                                    ),
                                    file_types=[".safetensors", ".pt", ".bin"],
                                )
                                # さらに追加のLoRAファイル
                                lora_files3 = gr.File(
                                    label=translate(
                                        "LoRAファイル3 (.safetensors, .pt, .bin)"
                                    ),
                                    file_types=[".safetensors", ".pt", ".bin"],
                                )

                            # ディレクトリから選択方式のコンポーネント（グループ化）
                            with gr.Group(visible=False) as lora_dropdown_group:
                                # ディレクトリからスキャンされたモデルのドロップダウン
                                lora_dropdown1 = gr.Dropdown(
                                    label=translate("LoRAモデル選択 1"),
                                    choices=[],
                                    value=None,
                                    allow_custom_value=False,
                                )
                                lora_dropdown2 = gr.Dropdown(
                                    label=translate("LoRAモデル選択 2"),
                                    choices=[],
                                    value=None,
                                    allow_custom_value=False,
                                )
                                lora_dropdown3 = gr.Dropdown(
                                    label=translate("LoRAモデル選択 3"),
                                    choices=[],
                                    value=None,
                                    allow_custom_value=False,
                                )
                                # スキャンボタン
                                lora_scan_button = gr.Button(
                                    translate("LoRAディレクトリを再スキャン"),
                                    variant="secondary",
                                )

                            # スケール値の入力フィールド（両方の方式で共通）
                            lora_scales_text = gr.Textbox(
                                label=translate("LoRA適用強度 (カンマ区切り)"),
                                value="0.8,0.8,0.8",
                                info=translate(
                                    "各LoRAのスケール値をカンマ区切りで入力 (例: 0.8,0.5,0.3)"
                                ),
                                visible=False,
                            )
                            lora_blocks_type = gr.Dropdown(
                                label=translate("LoRAブロック選択"),
                                choices=[
                                    "all",
                                    "single_blocks",
                                    "double_blocks",
                                    "db0-9",
                                    "db10-19",
                                    "sb0-9",
                                    "sb10-19",
                                    "important",
                                ],
                                value="all",
                                info=translate(
                                    "選択するブロックタイプ（all=すべて、その他=メモリ節約）"
                                ),
                                visible=False,
                            )

                            # LoRAディレクトリからモデルを検索する関数
                            def scan_lora_directory():
                                """./loraディレクトリからLoRAモデルファイルを検索する関数"""
                                lora_dir = os.path.join(
                                    os.path.dirname(os.path.abspath(__file__)), "lora"
                                )
                                choices = []

                                # ディレクトリが存在しない場合は作成
                                if not os.path.exists(lora_dir):
                                    os.makedirs(lora_dir, exist_ok=True)
                                    print(
                                        translate(
                                            "[INFO] LoRAディレクトリが存在しなかったため作成しました: {0}"
                                        ).format(lora_dir)
                                    )

                                # ディレクトリ内のファイルをリストアップ
                                for filename in os.listdir(lora_dir):
                                    if filename.endswith(
                                        (".safetensors", ".pt", ".bin")
                                    ):
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
                                        print(
                                            translate(
                                                "[DEBUG] 選択肢の型変換が必要: インデックス {0}, 型 {1}, 値 {2}"
                                            ).format(i, type(choice).__name__, choice)
                                        )
                                        # 明示的に文字列に変換
                                        choices[i] = str(choice)

                                # ファイル内容のデバッグ出力を追加
                                print(
                                    translate(
                                        "[INFO] LoRAディレクトリから{0}個のモデルを検出しました"
                                    ).format(len(choices) - 1)
                                )
                                print(
                                    translate(
                                        "[DEBUG] 'なし'の値: {0!r}, 型: {1}"
                                    ).format(choices[0], type(choices[0]).__name__)
                                )

                                # 一貫性があるか確認: "なし"がちゃんとに文字列型になっているか確認
                                if not isinstance(choices[0], str):
                                    print(
                                        translate(
                                            "[重要警告] 'なし'の選択肢が文字列型ではありません！型: {0}"
                                        ).format(type(choices[0]).__name__)
                                    )

                                # 数値の0に変換されないようにする
                                if choices[0] == 0 or choices[0] == 0.0:
                                    print(
                                        translate(
                                            "[重要警告] 'なし'の選択肢が数値0になっています。修正します。"
                                        )
                                    )
                                    choices[0] = none_choice

                                # 戻り値のオブジェクトと型を表示
                                print(
                                    translate(
                                        "[DEBUG] scan_lora_directory戻り値: 型={0}, 最初の要素={1!r}"
                                    ).format(
                                        type(choices).__name__,
                                        choices[0] if choices else "なし",
                                    )
                                )

                                return choices

                            # チェックボックスの状態によって他のLoRA設定の表示/非表示を切り替える関数
                            def toggle_lora_settings(use_lora):
                                if use_lora:
                                    # LoRA使用時はデフォルトでディレクトリから選択モードを表示
                                    choices = scan_lora_directory()
                                    print(
                                        translate(
                                            "[DEBUG] toggle_lora_settings - 選択肢リスト: {0}"
                                        ).format(choices)
                                    )

                                    # 選択肢がある場合は確実に文字列型に変換
                                    # 型チェックを追加
                                    for i, choice in enumerate(choices):
                                        if not isinstance(choice, str):
                                            print(
                                                translate(
                                                    "[DEBUG] toggle_lora_settings - 選択肢を文字列に変換: インデックス {0}, 元の値 {1}, 型 {2}"
                                                ).format(
                                                    i, choice, type(choice).__name__
                                                )
                                            )
                                            choices[i] = str(choice)

                                    # ドロップダウンが初期化時にも確実に更新されるようにする
                                    # LoRAを有効にしたときにドロップダウンの選択肢も適切に更新
                                    # まずモードを表示してからフラグを返す
                                    return [
                                        gr.update(visible=True),  # lora_mode
                                        gr.update(
                                            visible=False
                                        ),  # lora_upload_group - デフォルトでは非表示
                                        gr.update(
                                            visible=True
                                        ),  # lora_dropdown_group - デフォルトで表示
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
                                    print(
                                        translate(
                                            "[DEBUG] toggle_lora_mode - 選択肢リスト: {0}"
                                        ).format(choices)
                                    )

                                    # 選択肢の型を明示的に確認＆変換
                                    for i, choice in enumerate(choices):
                                        if not isinstance(choice, str):
                                            print(
                                                translate(
                                                    "[DEBUG] toggle_lora_mode - 選択肢を文字列に変換: インデックス {0}, 元の値 {1}, 型 {2}"
                                                ).format(
                                                    i, choice, type(choice).__name__
                                                )
                                            )
                                            choices[i] = str(choice)

                                    # 最初の選択肢がちゃんと文字列になっているか再確認
                                    first_choice = choices[0]
                                    print(
                                        translate(
                                            "[DEBUG] toggle_lora_mode - 変換後の最初の選択肢: {0}, 型: {1}"
                                        ).format(
                                            first_choice, type(first_choice).__name__
                                        )
                                    )

                                    # 選択肢が確実に更新されるようにする
                                    return [
                                        gr.update(visible=False),  # lora_upload_group
                                        gr.update(visible=True),  # lora_dropdown_group
                                        gr.update(
                                            choices=choices, value=choices[0]
                                        ),  # lora_dropdown1
                                        gr.update(
                                            choices=choices, value=choices[0]
                                        ),  # lora_dropdown2
                                        gr.update(
                                            choices=choices, value=choices[0]
                                        ),  # lora_dropdown3
                                    ]
                                else:  # ファイルアップロード
                                    # ファイルアップロード方式の場合、ドロップダウンの値は更新しない
                                    return [
                                        gr.update(visible=True),  # lora_upload_group
                                        gr.update(visible=False),  # lora_dropdown_group
                                        gr.update(),  # lora_dropdown1 - 変更なし
                                        gr.update(),  # lora_dropdown2 - 変更なし
                                        gr.update(),  # lora_dropdown3 - 変更なし
                                    ]

                            # スキャンボタンの処理関数
                            def update_lora_dropdowns():
                                choices = scan_lora_directory()
                                # LoRAドロップダウンの値を明示的に文字列として設定
                                print(
                                    translate(
                                        "[DEBUG] LoRAドロップダウン更新 - 選択肢: {0}"
                                    ).format(choices)
                                )
                                print(
                                    translate(
                                        "[DEBUG] 最初の選択肢: {0}, 型: {1}"
                                    ).format(choices[0], type(choices[0]).__name__)
                                )

                                # すべての選択肢が確実に文字列型であることを確認
                                for i, choice in enumerate(choices):
                                    if not isinstance(choice, str):
                                        print(
                                            translate(
                                                "[DEBUG] update_lora_dropdowns - 選択肢を文字列に変換: インデックス {0}, 値 {1}, 型 {2}"
                                            ).format(i, choice, type(choice).__name__)
                                        )
                                        choices[i] = str(choice)

                                # 各ドロップダウンを更新
                                print(
                                    translate(
                                        "[DEBUG] update_lora_dropdowns - ドロップダウン更新完了。選択肢: {0}"
                                    ).format(choices)
                                )

                                return [
                                    gr.update(
                                        choices=choices, value=choices[0]
                                    ),  # lora_dropdown1
                                    gr.update(
                                        choices=choices, value=choices[0]
                                    ),  # lora_dropdown2
                                    gr.update(
                                        choices=choices, value=choices[0]
                                    ),  # lora_dropdown3
                                ]

                            # UI初期化後のLoRAドロップダウン初期化関数
                            def initialize_lora_dropdowns(use_lora_val):
                                # LoRAが有効で、「ディレクトリから選択」モードの場合のみ更新
                                if use_lora_val:
                                    print(
                                        translate(
                                            "[INFO] UIの初期化時にLoRAドロップダウンを更新します"
                                        )
                                    )
                                    return update_lora_dropdowns()
                                return [gr.update(), gr.update(), gr.update()]

                            # 前回のLoRAモードを記憶するための変数
                            previous_lora_mode = translate(
                                "ディレクトリから選択"
                            )  # デフォルトはディレクトリから選択

                            # LoRA設定の変更を2ステップで行う関数
                            def toggle_lora_full_update(use_lora_val):
                                # グローバル変数でモードを記憶
                                global previous_lora_mode

                                # まずLoRA設定全体の表示/非表示を切り替え
                                # use_loraがオフの場合、まずモード値を保存
                                if not use_lora_val:
                                    # モードの現在値を取得
                                    current_mode = getattr(
                                        lora_mode,
                                        "value",
                                        translate("ディレクトリから選択"),
                                    )
                                    if current_mode:
                                        previous_lora_mode = current_mode
                                        print(
                                            translate(
                                                "[DEBUG] 前回のLoRAモードを保存: {0}"
                                            ).format(previous_lora_mode)
                                        )

                                # 表示/非表示の設定を取得
                                settings_updates = toggle_lora_settings(use_lora_val)

                                # もしLoRAが有効になった場合
                                if use_lora_val:
                                    print(
                                        translate(
                                            "[DEBUG] LoRAが有効になりました。前回のモード: {0}"
                                        ).format(previous_lora_mode)
                                    )

                                    # 前回のモードに基づいて表示を切り替え
                                    if previous_lora_mode == translate(
                                        "ファイルアップロード"
                                    ):
                                        # ファイルアップロードモードだった場合
                                        print(
                                            translate(
                                                "[DEBUG] 前回のモードはファイルアップロードだったため、ファイルアップロードUIを表示します"
                                            )
                                        )
                                        # モードの設定を上書き（ファイルアップロードに設定）
                                        settings_updates[0] = gr.update(
                                            visible=True,
                                            value=translate("ファイルアップロード"),
                                        )  # lora_mode
                                        settings_updates[1] = gr.update(
                                            visible=True
                                        )  # lora_upload_group
                                        settings_updates[2] = gr.update(
                                            visible=False
                                        )  # lora_dropdown_group

                                        # ドロップダウンは更新しない
                                        return settings_updates + [
                                            gr.update(),
                                            gr.update(),
                                            gr.update(),
                                        ]
                                    else:
                                        # デフォルトまたはディレクトリから選択モードだった場合
                                        choices = scan_lora_directory()
                                        print(
                                            translate(
                                                "[DEBUG] toggle_lora_full_update - LoRAドロップダウン選択肢: {0}"
                                            ).format(choices)
                                        )

                                        # ドロップダウンの更新を行う
                                        dropdown_updates = [
                                            gr.update(
                                                choices=choices, value=choices[0]
                                            ),  # lora_dropdown1
                                            gr.update(
                                                choices=choices, value=choices[0]
                                            ),  # lora_dropdown2
                                            gr.update(
                                                choices=choices, value=choices[0]
                                            ),  # lora_dropdown3
                                        ]

                                        # モードの設定を明示的に上書き
                                        settings_updates[0] = gr.update(
                                            visible=True,
                                            value=translate("ディレクトリから選択"),
                                        )  # lora_mode
                                        return settings_updates + dropdown_updates

                                # LoRAが無効な場合は設定の更新のみ
                                return settings_updates + [
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                ]

                            # チェックボックスの変更イベントにLoRA設定全体の表示/非表示を切り替える関数を紐づけ
                            use_lora.change(
                                fn=toggle_lora_full_update,
                                inputs=[use_lora],
                                outputs=[
                                    lora_mode,
                                    lora_upload_group,
                                    lora_dropdown_group,
                                    lora_scales_text,
                                    lora_dropdown1,
                                    lora_dropdown2,
                                    lora_dropdown3,
                                ],
                            )

                            # LoRAモードの変更を処理する関数
                            def toggle_lora_mode_with_memory(mode_value):
                                # グローバル変数に選択を保存
                                global previous_lora_mode
                                previous_lora_mode = mode_value
                                print(
                                    translate("[DEBUG] LoRAモードを変更: {0}").format(
                                        mode_value
                                    )
                                )

                                # 標準のtoggle_lora_mode関数を呼び出し
                                return toggle_lora_mode(mode_value)

                            # LoRA読み込み方式の変更イベントに表示切替関数を紐づけ
                            lora_mode.change(
                                fn=toggle_lora_mode_with_memory,
                                inputs=[lora_mode],
                                outputs=[
                                    lora_upload_group,
                                    lora_dropdown_group,
                                    lora_dropdown1,
                                    lora_dropdown2,
                                    lora_dropdown3,
                                ],
                            )

                            # スキャンボタンの処理を紐づけ
                            lora_scan_button.click(
                                fn=update_lora_dropdowns,
                                inputs=[],
                                outputs=[
                                    lora_dropdown1,
                                    lora_dropdown2,
                                    lora_dropdown3,
                                ],
                            )

                            # 代替の初期化方法：チェックボックスの初期値をチェックし、
                            # LoRAドロップダウンを明示的に初期化する補助関数
                            def lora_ready_init():
                                """LoRAドロップダウンの初期化を行う関数"""
                                print(
                                    translate(
                                        "[INFO] LoRAドロップダウンの初期化を開始します"
                                    )
                                )

                                # 現在のuse_loraとlora_modeの値を取得
                                use_lora_value = getattr(use_lora, "value", False)
                                lora_mode_value = getattr(
                                    lora_mode,
                                    "value",
                                    translate("ディレクトリから選択"),
                                )

                                print(
                                    translate(
                                        "[DEBUG] 初期化時の状態 - use_lora: {0}, lora_mode: {1}"
                                    ).format(use_lora_value, lora_mode_value)
                                )

                                # グローバル変数を更新
                                global previous_lora_mode
                                previous_lora_mode = lora_mode_value

                                if use_lora_value:
                                    # LoRAが有効な場合
                                    if lora_mode_value == translate(
                                        "ディレクトリから選択"
                                    ):
                                        # ディレクトリから選択モードの場合はドロップダウンを初期化
                                        print(
                                            translate(
                                                "[INFO] ディレクトリから選択モードでLoRAが有効なため、ドロップダウンを初期化します"
                                            )
                                        )
                                        choices = scan_lora_directory()
                                        print(
                                            translate(
                                                "[DEBUG] 初期化時のLoRA選択肢: {0}"
                                            ).format(choices)
                                        )
                                        return [
                                            gr.update(
                                                choices=choices, value=choices[0]
                                            ),  # lora_dropdown1
                                            gr.update(
                                                choices=choices, value=choices[0]
                                            ),  # lora_dropdown2
                                            gr.update(
                                                choices=choices, value=choices[0]
                                            ),  # lora_dropdown3
                                        ]
                                    else:
                                        # ファイルアップロードモードの場合はドロップダウンを更新しない
                                        print(
                                            translate(
                                                "[INFO] ファイルアップロードモードでLoRAが有効なため、ドロップダウンは更新しません"
                                            )
                                        )
                                        return [gr.update(), gr.update(), gr.update()]

                                # LoRAが無効な場合は何も更新しない
                                return [gr.update(), gr.update(), gr.update()]

                            # スキャンボタンの代わりにロード時の更新を行うボタン（非表示）
                            lora_init_btn = gr.Button(
                                visible=False, elem_id="lora_init_btn"
                            )
                            lora_init_btn.click(
                                fn=lora_ready_init,
                                inputs=[],
                                outputs=[
                                    lora_dropdown1,
                                    lora_dropdown2,
                                    lora_dropdown3,
                                ],
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
                                gr.Markdown(
                                    translate(
                                        "LoRAサポートは現在無効です。lora_utilsモジュールが必要です。"
                                    )
                                )

                # 入力イベント
                def toggle_combine_mode_update(mode):
                    if mode == COMBINE_MODE_OPTIONS_KEYS[0]:
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)

                # チェックボックスの変更イベントにテンソルデータの先頭の削除フレーム数の表示/非表示を切り替える関数を紐づけ
                combine_mode.change(
                    fn=toggle_combine_mode_update,
                    inputs=[combine_mode],
                    outputs=[trim_start_latent_size],
                )

                # ここで、メタデータ取得処理の登録を移動する
                # ここでは、promptとseedの両方が定義済み
                input_image.change(
                    fn=update_from_image_metadata,
                    inputs=[input_image, copy_metadata],
                    outputs=[prompt, seed],
                )

                # チェックボックスの変更時に再読み込みを行う
                def check_metadata_on_checkbox_change(copy_enabled, image_path):
                    if not copy_enabled or image_path is None:
                        return [gr.update()] * 2
                    # チェックボックスオン時に、画像があれば再度メタデータを読み込む
                    return update_from_image_metadata(image_path, copy_enabled)

                copy_metadata.change(
                    fn=check_metadata_on_checkbox_change,
                    inputs=[copy_metadata, input_image],
                    outputs=[prompt, seed],
                )

                def check_inputs_required(image_file, tensor_file):
                    return gr.update(interactive=bool(image_file and tensor_file))

                tensor_data_input.change(
                    check_inputs_required,
                    inputs=[input_image, tensor_data_input],
                    outputs=[start_button],
                )

                input_image.change(
                    check_inputs_required,
                    inputs=[input_image, tensor_data_input],
                    outputs=[start_button],
                )

                def process_tensor_file(file):
                    # テンソルデータのアップロードがあれば読み込み
                    if file is not None:
                        try:
                            tensor_path = file.name
                            print(
                                translate("テンソルデータを読み込み: {0}").format(
                                    os.path.basename(file)
                                )
                            )

                            # safetensorsからテンソルを読み込み
                            tensor_dict = sf.load_file(tensor_path)

                            # テンソルに含まれているキーとシェイプを確認
                            print(translate("テンソルデータの内容:"))
                            for key, tensor in tensor_dict.items():
                                print(
                                    f"  - {key}: shape={tensor.shape}, dtype={tensor.dtype}"
                                )

                            # history_latentsと呼ばれるキーが存在するか確認
                            if "history_latents" in tensor_dict:
                                # テンソルデータからlatentデータを取得
                                uploaded_tensor_latents = tensor_dict["history_latents"]
                                if uploaded_tensor_latents.shape[2] > 0:
                                    print(
                                        translate(
                                            "テンソルデータに 'history_latents' が見つかりました。{}"
                                        ).format(uploaded_tensor_latents.shape)
                                    )
                                    # 情報
                                    metadata = (
                                        [
                                            str(v)
                                            for v in tensor_dict["metadata"].tolist()
                                        ]
                                        if "metadata" in tensor_dict
                                        else ["metadata is not included"]
                                    )
                                    tensor_info = translate("""#### テンソルデータファイル情報:
                                        - keys: {keys}
                                        - history_latents: {history_latents_shape}
                                        - metadata: {metadata}
                                    """).format(
                                        keys=", ".join(list(tensor_dict.keys())),
                                        history_latents_shape=tensor_dict[
                                            "history_latents"
                                        ].shape,
                                        metadata=", ".join(metadata),
                                    )
                                    # テンソルデータ
                                    return gr.update(visible=True, value=tensor_info)
                                else:
                                    print(
                                        translate(
                                            "異常: テンソルデータに 'history_latents' が見つかりましたが、サイズが0です。"
                                        )
                                    )
                                    return gr.update(visible=False)
                            else:
                                print(
                                    translate(
                                        "異常: テンソルデータに 'history_latents' キーが見つかりません"
                                    )
                                )
                                return gr.update(visible=False)
                        except Exception as e:
                            print(
                                translate("テンソルデータ読み込みエラー: {0}").format(e)
                            )
                            import traceback

                            traceback.print_exc()
                            return gr.update(visible=False)
                    else:
                        # ファイル解除
                        return gr.update(
                            visible=True,
                            value=translate(
                                "ファイルを解除しました（エラーの場合も含む）。"
                            ),
                        )

                tensor_data_input.change(
                    process_tensor_file,
                    inputs=tensor_data_input,
                    outputs=[
                        preview_tensor_desc,
                    ],
                )

            # 出力
            with gr.Column():
                with gr.Group():
                    gr.Markdown(
                        "### " + translate("出力：生成動画"),
                        elem_classes="markdown-title",
                    )
                    with gr.Row():
                        # 動画
                        result_video = gr.Video(
                            label=translate("Finished Frames"),
                            autoplay=True,
                            show_share_button=False,
                            height=512,
                            loop=True,
                            format="mp4",
                            interactive=False,
                        )

                    with gr.Row():
                        # 処理状態
                        progress_desc = gr.Markdown(
                            "", elem_classes="no-generating-animation"
                        )

                    with gr.Row():
                        progress_bar = gr.HTML(
                            "", elem_classes="no-generating-animation"
                        )

                    with gr.Row():
                        preview_image = gr.Image(
                            label=translate("Next Latents"), height=200, visible=False
                        )

                    with gr.Row():
                        # 計算結果を表示するエリア
                        section_calc_display = gr.HTML("", label="")

                # 保存設定
                with gr.Group():
                    gr.Markdown(
                        "### " + translate("保存設定"),
                        elem_classes="markdown-title",
                    )
                    with gr.Row():
                        # MP4圧縮設定
                        mp4_crf = gr.Slider(
                            label=translate("MP4 Compression"),
                            minimum=0,
                            maximum=100,
                            value=16,
                            step=1,
                            info=translate(
                                "数値が小さいほど高品質になります。0は無圧縮。黒画面が出る場合は16に設定してください。"
                            ),
                        )
                    with gr.Row():
                        # 中間動画保存
                        keep_section_videos = gr.Checkbox(
                            label=translate(
                                "完了時に中間動画を残す - チェックがない場合は最終動画のみ保存されます（デフォルトOFF）"
                            ),
                            value=False,
                        )

                    with gr.Row():
                        # テンソルデータ保存
                        save_tensor_data = gr.Checkbox(
                            label=translate(
                                "完了時にテンソルデータ(.safetensors)も保存 - このデータを別の動画の前後に結合可能"
                            ),
                            value=False,
                            info=translate(
                                "チェックすると、生成されたテンソルデータを保存します。アップロードされたテンソルがあれば、結合したテンソルデータも保存されます。"
                            ),
                        )

                    with gr.Row():
                        # 中間ファイルの静止画保存
                        save_intermediate_frames = gr.Checkbox(
                            label=translate("中間ファイルの静止画保存"),
                            value=False,
                            info=translate(
                                "完了時に中間ファイルを静止画として保存します"
                            ),
                        )

                    with gr.Row():
                        # 出力フォルダ設定
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=4):
                                # フォルダ名だけを入力欄に設定
                                output_dir = gr.Textbox(
                                    label=translate(
                                        "出力フォルダ名 - 出力先は、webuiフォルダ配下に限定されます"
                                    ),
                                    value=output_folder_name,  # 設定から読み込んだ値を使用
                                    info=translate(
                                        "動画やキーフレーム画像の保存先フォルダ名"
                                    ),
                                    placeholder="outputs",
                                )
                            with gr.Column(scale=1, min_width=100):
                                open_folder_btn = gr.Button(
                                    value=translate("📂 保存および出力フォルダを開く"),
                                    size="sm",
                                )

                        # 実際の出力パスを表示
                        with gr.Row(visible=False):
                            path_display = gr.Textbox(
                                label=translate("出力フォルダの完全パス"),
                                value=os.path.join(base_path, output_folder_name),
                                interactive=False,
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
                            old_folder_name = settings.get("output_folder")

                            if old_folder_name != folder_name:
                                settings["output_folder"] = folder_name
                                save_result = save_settings(settings)
                                if save_result:
                                    # グローバル変数も更新
                                    global output_folder_name, outputs_folder
                                    output_folder_name = folder_name
                                    outputs_folder = folder_path
                                print(
                                    translate(
                                        "出力フォルダ設定を保存しました: {folder_name}"
                                    ).format(folder_name=folder_name)
                                )

                            # フォルダを開く
                            open_output_folder(folder_path)

                            # 出力ディレクトリ入力欄とパス表示を更新
                            return gr.update(value=folder_name), gr.update(
                                value=folder_path
                            )

                        open_folder_btn.click(
                            fn=handle_open_folder_btn,
                            inputs=[output_dir],
                            outputs=[output_dir, path_display],
                        )

    with gr.Tab("Performance"):
        # FP8最適化設定
        with gr.Row():
            fp8_optimization = gr.Checkbox(
                label=translate("FP8 最適化"),
                value=True,
                info=translate(
                    "メモリ使用量を削減し速度を改善（PyTorch 2.1以上が必要）"
                ),
            )

        with gr.Row():
            available_cuda_memory_gb = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
            )
            default_gpu_memory_preservation_gb = (
                6
                if available_cuda_memory_gb >= 20
                else (8 if available_cuda_memory_gb > 16 else 10)
            )
            gpu_memory_preservation = gr.Slider(
                label=translate(
                    "GPU Memory to Preserve (GB) (smaller = more VRAM usage)"
                ),
                minimum=6,
                maximum=128,
                value=default_gpu_memory_preservation_gb,
                step=0.1,
                info=translate(
                    "空けておくGPUメモリ量を指定。小さい値=より多くのVRAMを使用可能=高速、大きい値=より少ないVRAMを使用=安全"
                ),
            )

        with gr.Row():
            use_teacache = gr.Checkbox(
                label=translate("Use TeaCache"),
                value=True,
                info=translate(
                    "Faster speed, but often makes hands and fingers slightly worse."
                ),
            )

        with gr.Row():
            # VAEキャッシュ設定
            use_vae_cache = gr.Checkbox(
                label=translate("VAEキャッシュを使用"),
                value=False,
                info=translate(
                    "デコードを1フレームずつ処理し、速度向上（メモリ使用量増加。VRAM24GB以上推奨。それ以下の場合、メモリスワップで逆に遅くなります）"
                ),
            )
            print(f"VAEキャッシュCheckbox初期化: id={id(use_vae_cache)}")

        def update_vae_cache_state(value):
            global vae_cache_enabled
            vae_cache_enabled = value
            print(f"VAEキャッシュ状態を更新: {vae_cache_enabled}")
            return None

        # チェックボックスの状態が変更されたときにグローバル変数を更新
        use_vae_cache.change(
            fn=update_vae_cache_state, inputs=[use_vae_cache], outputs=[]
        )

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
        vae_controls["current_settings_md"].value = update_vae_settings_display()

    with gr.Tab("Presets"):
        # プロンプト管理パネルの追加
        with gr.Group(visible=True) as prompt_management:
            gr.Markdown(
                "### " + translate("プロンプト管理"),
                elem_classes="markdown-title",
            )

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
                    edit_name = gr.Textbox(
                        label=translate("プリセット名"),
                        placeholder=translate("名前を入力..."),
                        value=default_name,
                    )

                edit_prompt = gr.Textbox(
                    label=translate("プロンプト"), lines=5, value=default_prompt
                )

                with gr.Row():
                    # 起動時デフォルトをデフォルト選択に設定
                    default_preset = translate("起動時デフォルト")
                    # プリセットデータから全プリセット名を取得
                    presets_data = load_presets()
                    choices = [preset["name"] for preset in presets_data["presets"]]
                    default_presets = [
                        name
                        for name in choices
                        if any(
                            p["name"] == name and p.get("is_default", False)
                            for p in presets_data["presets"]
                        )
                    ]
                    user_presets = [
                        name for name in choices if name not in default_presets
                    ]
                    sorted_choices = [
                        (name, name)
                        for name in sorted(default_presets) + sorted(user_presets)
                    ]
                    preset_dropdown = gr.Dropdown(
                        label=translate("プリセット"),
                        choices=sorted_choices,
                        value=default_preset,
                        type="value",
                    )

                with gr.Row():
                    save_btn = gr.Button(value=translate("保存"), variant="primary")
                    apply_preset_btn = gr.Button(
                        value=translate("反映"), variant="primary"
                    )
                    clear_btn = gr.Button(value=translate("クリア"))
                    delete_preset_btn = gr.Button(value=translate("削除"))

            # メッセージ表示用
            result_message = gr.Markdown("")

    with gr.Tab("Tools"):
        # テンソルファイルの結合
        with gr.Group(visible=True):
            gr.Markdown(
                "### " + translate("テンソルファイルの結合"),
                elem_classes="markdown-title",
            )
            gr.Markdown(
                translate(
                    "safetensors形式のテンソルファイルを2つ選択して結合します。結合順序は「テンソル1 + テンソル2」です。"
                ),
                elem_classes="markdown-desc",
            )
            with gr.Row():
                with gr.Column(elem_classes="group-border"):
                    gr.Markdown(
                        "#### " + translate("入力"), elem_classes="markdown-subtitle"
                    )

                    with gr.Column(scale=1):
                        tool_tensor_file1 = gr.File(
                            label=translate("テンソルファイル1 (.safetensors)"),
                            file_types=[".safetensors"],
                            height=200,
                        )
                    with gr.Column(scale=1):
                        tool_tensor_file2 = gr.File(
                            label=translate("テンソルファイル2 (.safetensors)"),
                            file_types=[".safetensors"],
                            height=200,
                        )

                with gr.Column(elem_classes="group-border"):
                    gr.Markdown(
                        "#### " + translate("出力"), elem_classes="markdown-subtitle"
                    )

                    # テンソルデータ情報
                    tool_combined_tensor_data_desc = gr.Markdown(
                        "",
                        elem_classes="markdown-desc",
                        height=240,
                    )

            with gr.Row():
                tool_combine_btn = gr.Button(
                    translate("テンソルファイルを結合"),
                    variant="primary",
                    interactive=False,
                )

            def check_tool_tensor_files(file1, file2):
                # 両方のファイルが選択されている場合のみボタンを有効化
                return gr.update(interactive=bool(file1 and file2))

            tool_tensor_file1.change(
                check_tool_tensor_files,
                inputs=[tool_tensor_file1, tool_tensor_file2],
                outputs=tool_combine_btn,
            )

            tool_tensor_file2.change(
                check_tool_tensor_files,
                inputs=[tool_tensor_file1, tool_tensor_file2],
                outputs=tool_combine_btn,
            )

            def combine_tensor_files(file1_path, file2_path, output_path=None):
                """2つのsafetensorsファイルを読み込み、結合して新しいファイルに保存する

                Args:
                    file1_path (str): 1つ目のsafetensorsファイルパス
                    file2_path (str): 2つ目のsafetensorsファイルパス
                    output_path (str, optional): 出力ファイルパス。指定しない場合は自動生成

                Returns:
                    tuple: (成功したかどうかのbool, 出力ファイルパス, 結果メッセージ)
                """
                try:
                    # ファイル1を読み込み
                    print(
                        translate("ファイル1を読み込み中: {0}").format(
                            os.path.basename(file1_path)
                        )
                    )
                    tensor_dict1 = sf.load_file(file1_path)

                    # ファイル2を読み込み
                    print(
                        translate("ファイル2を読み込み中: {0}").format(
                            os.path.basename(file2_path)
                        )
                    )
                    tensor_dict2 = sf.load_file(file2_path)

                    # テンソルを取得
                    if (
                        "history_latents" in tensor_dict1
                        and "history_latents" in tensor_dict2
                    ):
                        tensor1 = tensor_dict1["history_latents"]
                        tensor2 = tensor_dict2["history_latents"]

                        # テンソル情報の表示
                        print(
                            translate(
                                "テンソル1: shape={0}, dtype={1}, フレーム数={2}"
                            ).format(tensor1.shape, tensor1.dtype, tensor1.shape[2])
                        )
                        print(
                            translate(
                                "テンソル2: shape={0}, dtype={1}, フレーム数={2}"
                            ).format(tensor2.shape, tensor2.dtype, tensor2.shape[2])
                        )

                        # サイズチェック
                        if (
                            tensor1.shape[3] != tensor2.shape[3]
                            or tensor1.shape[4] != tensor2.shape[4]
                        ):
                            error_msg = translate(
                                "エラー: テンソルサイズが異なります: {0} vs {1}"
                            ).format(tensor1.shape, tensor2.shape)
                            print(error_msg)
                            return False, None, error_msg

                        # データ型とデバイスの調整
                        if tensor1.dtype != tensor2.dtype:
                            print(
                                translate("データ型の変換: {0} → {1}").format(
                                    tensor2.dtype, tensor1.dtype
                                )
                            )
                            tensor2 = tensor2.to(dtype=tensor1.dtype)

                        # 両方CPUに移動
                        tensor1 = tensor1.cpu()
                        tensor2 = tensor2.cpu()

                        # 結合（テンソル1の後にテンソル2を追加）
                        combined_tensor = torch.cat([tensor1, tensor2], dim=2)

                        # 結合されたテンソルの情報を表示
                        tensor1_frames = tensor1.shape[2]
                        tensor2_frames = tensor2.shape[2]
                        combined_frames = combined_tensor.shape[2]
                        print(
                            translate(
                                "結合成功: 結合後のフレーム数={0} ({1}+{2}フレーム)"
                            ).format(combined_frames, tensor1_frames, tensor2_frames)
                        )

                        # メタデータを更新
                        height, width = tensor1.shape[3], tensor1.shape[4]
                        metadata = torch.tensor(
                            [height, width, combined_frames], dtype=torch.int32
                        )

                        # 出力ファイルパスが指定されていない場合は自動生成
                        if output_path is None:
                            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
                            output_dir = os.path.dirname(file1_path)
                            output_path = os.path.join(
                                output_dir, f"{timestamp}_combined.safetensors"
                            )

                        # 結合したテンソルをファイルに保存
                        tensor_dict = {
                            "history_latents": combined_tensor,
                            "metadata": metadata,
                        }

                        # ファイル保存
                        sf.save_file(tensor_dict, output_path)

                        # テンソルデータの保存サイズの概算
                        tensor_size_mb = (
                            combined_tensor.element_size() * combined_tensor.nelement()
                        ) / (1024 * 1024)

                        # 情報文字列の作成
                        # 情報
                        metadata = (
                            [str(v) for v in tensor_dict["metadata"].tolist()]
                            if "metadata" in tensor_dict
                            else ["metadata is not included"]
                        )
                        info_text = translate("""結合成功
                            #### 結合後のテンソルファイル情報:
                            - ファイル名: {filename}
                            - フレーム数: {frames}フレーム ({frames1}+{frames2}フレーム)
                            - サイズ: {tensor_size_mb:.2f}MB
                            - keys: {keys}
                            - history_latents: {history_latents_shape}
                            - metadata: {metadata}
                        """).format(
                            filename=output_path,
                            frames=combined_frames,
                            frames1=tensor1_frames,
                            frames2=tensor2_frames,
                            tensor_size_mb=tensor_size_mb,
                            keys=", ".join(list(tensor_dict.keys())),
                            history_latents_shape=tensor_dict["history_latents"].shape,
                            metadata=", ".join(metadata),
                        )

                        return True, output_path, info_text
                    else:
                        error_msg = translate(
                            "エラー: テンソルファイルに必要なキー'history_latents'がありません"
                        )
                        print(error_msg)
                        return False, None, error_msg

                except Exception as e:
                    error_msg = translate("テンソル結合中にエラーが発生: {0}").format(e)
                    print(error_msg)
                    traceback.print_exc()
                    return False, None, error_msg

            def combine_tensors(file1, file2):
                if file1 is None or file2 is None:
                    return translate("エラー: 2つのテンソルファイルを選択してください")

                file1_path = file1.name
                file2_path = file2.name

                # 出力パス
                job_id = generate_timestamp() + "_combined"
                output_filename = os.path.join(outputs_folder, f"{job_id}.safetensors")

                success, result_path, message = combine_tensor_files(
                    file1_path, file2_path, output_filename
                )
                if success:
                    return message
                else:
                    return translate("結合失敗: {0}").format(message)

            def disable_tool_combine_btn():
                return gr.update(interactive=False)

            def enable_tool_combine_btn():
                return gr.update(interactive=True)

            tool_combine_btn.click(
                disable_tool_combine_btn,
                inputs=[],
                outputs=tool_combine_btn,
                queue=True,
            ).then(
                combine_tensors,  # メイン処理を実行
                inputs=[tool_tensor_file1, tool_tensor_file2],
                outputs=[tool_combined_tensor_data_desc],
                queue=True,
            ).then(
                enable_tool_combine_btn,
                inputs=[],
                outputs=tool_combine_btn,
            )

        # テンソルファイルのMP4化
        with gr.Group(visible=True):
            gr.Markdown(
                "### " + translate("テンソルファイルのMP4化"),
                elem_classes="markdown-title",
            )
            gr.Markdown(
                translate("safetensors形式のテンソルファイルをMP4動画にします。"),
                elem_classes="markdown-desc",
            )
            with gr.Row():
                with gr.Column(elem_classes="group-border"):
                    gr.Markdown(
                        "#### " + translate("入力"), elem_classes="markdown-subtitle"
                    )
                    # テンソルファイルのアップロード
                    tool_tensor_data_input = gr.File(
                        label=translate(
                            "テンソルデータアップロード (.safetensors) - eichiにて生成したファイル"
                        ),
                        file_types=[".safetensors"],
                        type="filepath",
                        height=200,
                    )

                    # テンソルデータ読み込み状態
                    tool_preview_tensor_desc = gr.Markdown(
                        "", elem_classes="markdown-desc"
                    )

                with gr.Column(elem_classes="group-border"):
                    gr.Markdown(
                        "#### " + translate("出力"), elem_classes="markdown-subtitle"
                    )
                    # 動画
                    tool_tensor_video = gr.Video(
                        label=translate("Tensor file frames"),
                        autoplay=True,
                        show_share_button=False,
                        height=256,
                        loop=False,
                        format="mp4",
                        interactive=False,
                    )

            with gr.Row():
                # MP4ファイル作成ボタン
                tool_create_mp4_button = gr.Button(
                    value=translate("MP4ファイル作成"),
                    variant="primary",
                    interactive=False,
                )

            def create_mp4_from_tensor(file, mp4_crf):
                if file is not None:
                    tensor_path = file.name
                    # safetensorsからテンソルを読み込み
                    tensor_dict = sf.load_file(tensor_path)
                    # テンソルデータからlatentデータを取得
                    uploaded_tensor_latents = tensor_dict["history_latents"]
                    job_id = generate_timestamp() + "_tensor_to_mp4"

                    if not high_vram:
                        # 減圧時に使用するGPUメモリ値も明示的に浮動小数点に設定
                        preserved_memory_offload = 8.0  # こちらは固定値のまま
                        print(
                            translate(
                                "Offloading transformer with memory preservation: {0} GB"
                            ).format(preserved_memory_offload)
                        )
                        offload_model_from_device_for_memory_preservation(
                            transformer,
                            target_device=gpu,
                            preserved_memory_gb=preserved_memory_offload,
                        )
                        load_model_as_complete(vae, target_device=gpu)

                    uploaded_tensor_pixels, _ = process_tensor_chunks(
                        tensor=uploaded_tensor_latents,
                        frames=uploaded_tensor_latents.shape[2],
                        use_vae_cache=use_vae_cache.value,
                        job_id=job_id,
                        outputs_folder=outputs_folder,
                        mp4_crf=mp4_crf,
                        stream=stream,
                        vae=vae,
                    )
                    # 入力されたテンソルデータの動画
                    input_tensor_output_filename = os.path.join(
                        outputs_folder, f"{job_id}_input_safetensors.mp4"
                    )
                    save_bcthw_as_mp4(
                        uploaded_tensor_pixels,
                        input_tensor_output_filename,
                        fps=30,
                        crf=mp4_crf,
                    )
                    print(
                        translate(
                            "入力されたテンソルデータの動画を保存しました: {input_tensor_output_filename}"
                        ).format(
                            input_tensor_output_filename=input_tensor_output_filename
                        )
                    )
                    return gr.update(value=input_tensor_output_filename)
                else:
                    return gr.update()

            def disable_tool_create_mp4_button():
                return gr.update(interactive=False)

            def enable_tool_create_mp4_button(file):
                if file:
                    return gr.update(interactive=True)
                else:
                    return gr.update(interactive=False)

            tool_tensor_data_input.change(
                process_tensor_file,
                inputs=tool_tensor_data_input,
                outputs=tool_preview_tensor_desc,
            ).then(
                enable_tool_create_mp4_button,
                inputs=tool_tensor_data_input,
                outputs=tool_create_mp4_button,
            )

            tool_create_mp4_button.click(
                disable_tool_create_mp4_button,
                inputs=[],
                outputs=tool_create_mp4_button,
                queue=True,
            ).then(
                create_mp4_from_tensor,  # メイン処理を実行
                inputs=[tool_tensor_data_input, mp4_crf],
                outputs=tool_tensor_video,
                queue=True,
            ).then(
                enable_tool_create_mp4_button,
                inputs=tool_tensor_data_input,
                outputs=tool_create_mp4_button,
            )

        # MP4ファイルのテンソルファイル化
        with gr.Group(visible=True):
            gr.Markdown(
                "### " + translate("MP4ファイルのテンソルファイル化"),
                elem_classes="markdown-title",
            )
            gr.Markdown(
                translate(
                    "MP4ファイルをeichiで使用可能なsafetensors形式のテンソルファイルにします。"
                ),
                elem_classes="markdown-desc",
            )
            with gr.Row():
                with gr.Column(elem_classes="group-border"):
                    gr.Markdown(
                        "#### " + translate("入力"), elem_classes="markdown-subtitle"
                    )
                    # MP4ファイルのアップロード
                    tool_mp4_data_input = gr.File(
                        label=translate("MP4ファイルアップロード (.mp4)"),
                        file_types=[".mp4"],
                        type="filepath",
                        height=200,
                    )

                    # MP4ファイル読み込み状態
                    tool_preview_mp4_desc = gr.Markdown(
                        "", elem_classes="markdown-desc"
                    )

                with gr.Column(elem_classes="group-border"):
                    gr.Markdown(
                        "#### " + translate("出力"), elem_classes="markdown-subtitle"
                    )
                    # テンソルデータ情報
                    tool_tensor_data_desc = gr.Markdown(
                        "",
                        elem_classes="markdown-desc",
                        height=240,
                    )

            with gr.Row():
                # MP4ファイル作成ボタン
                tool_create_tensor_button = gr.Button(
                    value=translate("テンソルファイル作成"),
                    variant="primary",
                    interactive=False,
                )

            def process_mp4_file(file):
                """アップロードされたMP4ファイルの情報を表示する

                Args:
                    file: アップロードされたMP4ファイルパス

                Returns:
                    tuple: (プレビュー説明文, ボタンの有効/無効状態)
                """
                # MP4ファイルのアップロードがあれば読み込み
                if file is not None:
                    try:
                        mp4_path = file.name
                        # OpenCVでMP4ファイルを開く
                        cap = cv2.VideoCapture(mp4_path)

                        if not cap.isOpened():
                            return (
                                translate("エラー: MP4ファイルを開けませんでした"),
                                gr.update(interactive=False),
                            )

                        # 基本情報の取得
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        duration = frame_count / fps if fps > 0 else 0

                        # キャプチャを解放
                        cap.release()

                        # 情報文字列の作成
                        info_text = translate("""#### MP4ファイル情報:
                            - ファイル名: {filename}
                            - フレーム数: {frames}
                            - フレームレート: {fps:.2f} fps
                            - 解像度: H{height}xW{width}
                            - 長さ: {duration:.2f} 秒
                        """).format(
                            filename=os.path.basename(mp4_path),
                            frames=frame_count,
                            fps=fps,
                            width=width,
                            height=height,
                            duration=duration,
                        )

                        # ボタンを有効化して情報を表示
                        return (
                            info_text,
                            gr.update(interactive=True),
                        )

                    except Exception as e:
                        error_msg = translate(
                            "MP4ファイルの読み込み中にエラーが発生: {0}"
                        ).format(str(e))
                        print(error_msg)
                        traceback.print_exc()
                        return (
                            error_msg,
                            gr.update(interactive=False),
                        )

                # ファイルが選択されていない場合
                return (
                    translate("MP4ファイルを選択してください"),
                    gr.update(interactive=False),
                )

            tool_mp4_data_input.change(
                process_mp4_file,
                inputs=tool_mp4_data_input,
                outputs=[
                    tool_preview_mp4_desc,
                    tool_create_tensor_button,
                ],
            )

            def create_tensor_from_mp4(file):
                """MP4ファイルからテンソルデータを生成する

                Args:
                    file: アップロードされたMP4ファイルパス

                Returns:
                    gr.update: video要素の更新情報
                """
                try:
                    if file is not None:
                        mp4_path = file.name
                        job_id = generate_timestamp() + "_mp4_to_tensor"

                        print(
                            translate("[INFO] MP4ファイルの処理を開始: {0}").format(
                                mp4_path
                            )
                        )

                        # VAEをGPUに移動
                        if not high_vram:
                            load_model_as_complete(vae, target_device=gpu)

                        # MP4ファイルから画像シーケンスを読み込み
                        cap = cv2.VideoCapture(mp4_path)
                        frames = []

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            # BGRからRGBに変換
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)

                        cap.release()

                        # フレーム数を確認
                        if len(frames) == 0:
                            raise ValueError(
                                translate("MP4ファイルにフレームが含まれていません")
                            )

                        print(
                            translate("[INFO] 読み込んだフレーム数: {0}").format(
                                len(frames)
                            )
                        )

                        # フレームをテンソルに変換
                        # shape: [T, H, W, C] → [T, C, H, W] でスタック
                        frames_tensor = torch.from_numpy(np.stack(frames))
                        # # [C, T, H, W]
                        frames_tensor = frames_tensor.permute(3, 0, 1, 2)
                        # 正規化
                        frames_tensor = frames_tensor.float() / 127.5 - 1.0
                        # バッチ次元を追加: [B, C, T, H, W]
                        frames_tensor = frames_tensor.unsqueeze(0)

                        print(
                            translate("  - frames_tensorの形状: {0}").format(
                                frames_tensor.shape
                            )
                        )

                        # VAEエンコード
                        frames_tensor = ensure_tensor_properties(
                            frames_tensor, vae.device
                        )
                        latents = vae_encode(frames_tensor, vae)
                        latents = latents.to(torch.float16)

                        print(translate("[INFO] VAEエンコード完了"))
                        print(translate("  - Latentsの形状: {0}").format(latents.shape))

                        # テンソルデータを保存
                        tensor_output_path = os.path.join(
                            outputs_folder, f"{job_id}.safetensors"
                        )

                        # メタデータの準備（フレーム数も含める）
                        metadata = torch.tensor(
                            [
                                frames_tensor.shape[3],
                                frames_tensor.shape[4],
                                latents.shape[2],
                            ],
                            dtype=torch.int32,
                        )

                        tensor_dict = {
                            "history_latents": latents,
                            "metadata": metadata,
                        }
                        sf.save_file(tensor_dict, tensor_output_path)

                        print(
                            translate(
                                "[INFO] テンソルデータを保存しました: {0}"
                            ).format(tensor_output_path)
                        )

                        # 生成されたテンソルデータからMP4を作成して確認
                        decoded_pixels, _ = process_tensor_chunks(
                            tensor=latents,
                            frames=latents.shape[2],
                            use_vae_cache=vae_cache_enabled,
                            job_id=job_id,
                            outputs_folder=outputs_folder,
                            mp4_crf=16,  # デフォルトのCRF値
                            stream=stream,
                            vae=vae,
                        )

                        print(
                            translate("  - decoded_pixelsの形状: {0}").format(
                                decoded_pixels.shape
                            )
                        )

                        # 情報
                        metadata = (
                            [str(v) for v in tensor_dict["metadata"].tolist()]
                            if "metadata" in tensor_dict
                            else ["metadata is not included"]
                        )
                        tensor_info = translate("""#### テンソルファイル情報:
                            - 出力先: {file_path}
                            - keys: {keys}
                            - history_latents: {history_latents_shape}
                            - metadata: {metadata}
                        """).format(
                            file_path=tensor_output_path,
                            keys=", ".join(list(tensor_dict.keys())),
                            history_latents_shape=tensor_dict["history_latents"].shape,
                            metadata=", ".join(metadata),
                        )

                        # テンソルデータ
                        return gr.update(value=tensor_info)
                    else:
                        return gr.update()
                except Exception as e:
                    error_msg = translate(
                        "MP4からテンソルデータの生成中にエラーが発生: {0}"
                    ).format(str(e))
                    print(error_msg)
                    traceback.print_exc()
                    return gr.update()

            def disable_tool_create_tensor_button():
                return gr.update(interactive=False)

            def enable_tool_create_tensor_button():
                return gr.update(interactive=True)

            tool_create_tensor_button.click(
                disable_tool_create_tensor_button,
                inputs=[],
                outputs=tool_create_tensor_button,
                queue=True,
            ).then(
                create_tensor_from_mp4,  # メイン処理を実行
                inputs=[tool_mp4_data_input],
                outputs=tool_tensor_data_desc,
                queue=True,
            ).then(
                enable_tool_create_tensor_button,
                inputs=[],
                outputs=tool_create_tensor_button,
            )

    # 実行前のバリデーション関数
    def validate_and_process(
        input_image,
        prompt,
        seed,
        steps,
        cfg,
        gs,
        rs,
        gpu_memory_preservation,
        use_teacache,
        mp4_crf=16,
        end_frame_strength=1.0,
        keep_section_videos=False,
        lora_files=None,
        lora_files2=None,
        lora_files3=None,
        lora_scales_text="0.8,0.8,0.8",
        output_dir=None,
        save_intermediate_frames=False,
        use_lora=False,
        lora_mode=None,
        lora_dropdown1=None,
        lora_dropdown2=None,
        lora_dropdown3=None,
        save_tensor_data=False,
        tensor_data_input=None,
        trim_start_latent_size=0,
        generation_latent_size=0,
        combine_mode=COMBINE_MODE_DEFAULT,
        fp8_optimization=False,
        batch_count=1,
        use_vae_cache=False,
    ):
        # グローバル変数の宣言 - 関数の先頭で行う
        global \
            batch_stopped, \
            queue_enabled, \
            queue_type, \
            prompt_queue_file_path, \
            vae_cache_enabled, \
            image_queue_files
        # すべての入力パラメーターの型情報をデバッグ出力（問題診断用）
        print("=== 入力パラメーター型情報 ===")
        print("=========================")
        """入力画像または最後のキーフレーム画像のいずれかが有効かどうかを確認し、問題がなければ処理を実行する"""
        # 引数のデバッグ出力
        print(f"validate_and_process 引数数: {len(locals())}")

        # LoRA関連の引数を確認
        print(f"[DEBUG] LoRA関連の引数確認")
        print(f"[DEBUG] lora_files: {lora_files}, 型: {type(lora_files)}")
        print(f"[DEBUG] lora_files2: {lora_files2}, 型: {type(lora_files2)}")
        print(f"[DEBUG] lora_files3: {lora_files3}, 型: {type(lora_files3)}")
        print(
            f"[DEBUG] lora_scales_text: {lora_scales_text}, 型: {type(lora_scales_text)}"
        )
        print(f"[DEBUG] use_lora: {use_lora}, 型: {type(use_lora)}")
        print(f"[DEBUG] lora_mode: {lora_mode}, 型: {type(lora_mode)}")
        print(f"[DEBUG] lora_dropdown1: {lora_dropdown1}, 型: {type(lora_dropdown1)}")
        print(f"[DEBUG] lora_dropdown2: {lora_dropdown2}, 型: {type(lora_dropdown2)}")
        print(f"[DEBUG] lora_dropdown3: {lora_dropdown3}, 型: {type(lora_dropdown3)}")

        # バッチカウント引数の確認
        print(f"[DEBUG] batch_count: {batch_count}, 型: {type(batch_count)}")

        # VAEキャッシュの引数位置を確認
        print(
            f"[DEBUG] VAEキャッシュ設定値: {use_vae_cache}, 型: {type(use_vae_cache)}"
        )

        # グローバル変数宣言
        global vae_cache_enabled

        input_img = input_image  # 入力の最初が入力画像

        # デバッグ: 引数の型と値を表示
        print(
            translate("[DEBUG] batch_count 型: {0}, 値: {1}").format(
                type(batch_count).__name__, batch_count
            )
        )

        # VAEキャッシュを取得（グローバル変数を優先）
        use_vae_cache_ui_value = use_vae_cache

        # UIの値よりもグローバル変数を優先
        use_vae_cache_value = vae_cache_enabled

        print(
            f"VAEキャッシュ設定値(UI): {use_vae_cache_ui_value}, 型: {type(use_vae_cache_ui_value)}"
        )
        print(
            f"VAEキャッシュ設定値(グローバル変数): {vae_cache_enabled}, 型: {type(vae_cache_enabled)}"
        )
        print(
            f"最終的なVAEキャッシュ設定値: {use_vae_cache_value}, 型: {type(use_vae_cache_value)}"
        )

        # バッチ回数を有効な範囲に制限
        # 型チェックしてから変換（数値でない場合はデフォルト値の1を使用）
        try:
            batch_count_val = int(batch_count)
            batch_count = max(1, min(batch_count_val, 100))  # 1〜100の間に制限
        except (ValueError, TypeError):
            print(
                translate(
                    "[WARN] validate_and_process: バッチ処理回数が無効です。デフォルト値の1を使用します: {0}"
                ).format(batch_count)
            )
            batch_count = 1  # デフォルト値

        # ドロップダウン選択に基づいてuse_loraフラグを調整は既に引数で受け取り済み
        # 詳細なデバッグ出力を追加
        print(translate("[DEBUG] validate_and_process詳細："))
        print(
            f"[DEBUG] lora_dropdown1 詳細: 値={repr(lora_dropdown1)}, 型={type(lora_dropdown1).__name__}"
        )
        print(
            f"[DEBUG] lora_dropdown2 詳細: 値={repr(lora_dropdown2)}, 型={type(lora_dropdown2).__name__}"
        )
        print(
            f"[DEBUG] lora_dropdown3 詳細: 値={repr(lora_dropdown3)}, 型={type(lora_dropdown3).__name__}"
        )

        # ディレクトリ選択モードの場合の処理
        if lora_mode == translate("ディレクトリから選択") and has_lora_support:
            # ドロップダウン選択があるか確認
            has_dropdown_selection = False
            dropdown_values = [
                (1, lora_dropdown1),
                (2, lora_dropdown2),
                (3, lora_dropdown3),
            ]

            for idx, dropdown in dropdown_values:
                # 詳細なデバッグ情報
                print(
                    translate(
                        "[DEBUG] ドロップダウン{0}の検出処理: 値={1!r}, 型={2}"
                    ).format(idx, dropdown, type(dropdown).__name__)
                )

                # 処理用にローカル変数にコピー（元の値を保持するため）
                processed_value = dropdown

                # 通常の値が0や0.0などの数値の場合の特別処理（GradioのUIの問題によるもの）
                if (
                    processed_value == 0
                    or processed_value == "0"
                    or processed_value == 0.0
                ):
                    # 数値の0を"なし"として扱う
                    print(
                        translate(
                            "[DEBUG] validate_and_process: ドロップダウン{0}の値が数値0として検出されました。'なし'として扱います"
                        ).format(idx)
                    )
                    processed_value = translate("なし")

                # 文字列でない場合は変換
                if processed_value is not None and not isinstance(processed_value, str):
                    processed_value = str(processed_value)
                    print(
                        translate(
                            "[DEBUG] validate_and_process: ドロップダウン{0}の値を文字列に変換: {1!r}"
                        ).format(idx, processed_value)
                    )

                # 有効な選択かチェック
                if processed_value and processed_value != translate("なし"):
                    has_dropdown_selection = True
                    print(
                        translate(
                            "[DEBUG] validate_and_process: ドロップダウン{0}で有効な選択を検出: {1!r}"
                        ).format(idx, processed_value)
                    )
                    break

            # 選択があれば有効化
            if has_dropdown_selection:
                use_lora = True
                print(
                    translate(
                        "[INFO] validate_and_process: ドロップダウンでLoRAが選択されているため、LoRA使用を自動的に有効化しました"
                    )
                )

        # フラグの最終状態を出力
        print(translate("[DEBUG] LoRA使用フラグの最終状態: {0}").format(use_lora))

        # 現在の動画長設定とフレームサイズ設定を渡す
        is_valid, error_message = validate_images(input_img)

        if not is_valid:
            # 画像が無い場合はエラーメッセージを表示して終了
            yield (
                None,
                gr.update(visible=False),
                translate("エラー: 画像が選択されていません"),
                error_message,
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(),
            )
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
        resolution_value = 640

        # グローバル変数からVAEキャッシュ設定を取得
        use_vae_cache = vae_cache_enabled
        print(f"最終的なVAEキャッシュ設定フラグ: {use_vae_cache}")

        # 最終的なフラグを設定（名前付き引数で渡すため変数の整理のみ）
        print(
            translate("[DEBUG] 最終的なuse_loraフラグを{0}に設定しました").format(
                use_lora
            )
        )
        # 最終的な設定値の確認
        print(
            f"[DEBUG] 最終的なフラグ設定 - use_vae_cache: {use_vae_cache}, use_lora: {use_lora}"
        )

        # process関数のジェネレータを返す - 明示的に全ての引数を渡す
        yield from process(
            input_image=input_image,
            prompt=prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            gpu_memory_preservation=gpu_memory_preservation,
            use_teacache=use_teacache,
            mp4_crf=mp4_crf,
            end_frame_strength=end_frame_strength,
            keep_section_videos=keep_section_videos,
            lora_files=lora_files,
            lora_files2=lora_files2,
            lora_files3=lora_files3,
            lora_scales_text=lora_scales_text,
            output_dir=output_dir,
            save_intermediate_frames=save_intermediate_frames,
            use_lora=use_lora,
            lora_mode=lora_mode,
            lora_dropdown1=lora_dropdown1,
            lora_dropdown2=lora_dropdown2,
            lora_dropdown3=lora_dropdown3,
            save_tensor_data=save_tensor_data,
            tensor_data_input=tensor_data_input,
            trim_start_latent_size=trim_start_latent_size,
            generation_latent_size=generation_latent_size,
            combine_mode=combine_mode,
            fp8_optimization=fp8_optimization,
            batch_count=batch_count,
            use_vae_cache=use_vae_cache,
        )

    # 実行ボタンのイベント
    # UIから渡されるパラメーターリスト
    ips = [
        input_image,
        prompt,
        seed,
        steps,
        cfg,
        gs,
        rs,
        gpu_memory_preservation,
        use_teacache,
        mp4_crf,
        end_frame_strength,
        keep_section_videos,
        lora_files,
        lora_files2,
        lora_files3,
        lora_scales_text,
        output_dir,
        save_intermediate_frames,
        use_lora,
        lora_mode,
        lora_dropdown1,
        lora_dropdown2,
        lora_dropdown3,
        save_tensor_data,
        tensor_data_input,
        trim_start_latent_size,
        generation_latent_size,
        combine_mode,
        fp8_optimization,
        batch_count,
        use_vae_cache,
    ]

    # デバッグ: チェックボックスの現在値を出力
    print(
        f"use_vae_cacheチェックボックス値: {use_vae_cache.value if hasattr(use_vae_cache, 'value') else 'no value attribute'}, id={id(use_vae_cache)}"
    )
    start_button.click(
        fn=validate_and_process,
        inputs=ips,
        outputs=[
            result_video,
            preview_image,
            progress_desc,
            progress_bar,
            start_button,
            end_button,
            seed,
        ],
    )
    end_button.click(fn=end_process, outputs=[end_button])

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
        default_presets = [
            n
            for n in choices
            if any(
                p["name"] == n and p.get("is_default", False)
                for p in presets_data["presets"]
            )
        ]
        user_presets = [n for n in choices if n not in default_presets]
        sorted_choices = [
            (n, n) for n in sorted(default_presets) + sorted(user_presets)
        ]

        # メインプロンプトは更新しない（保存のみを行う）
        return result_msg, gr.update(choices=sorted_choices), gr.update()

    # 保存ボタンのクリックイベントを接続
    save_btn.click(
        fn=save_button_click_handler,
        inputs=[edit_name, edit_prompt],
        outputs=[result_message, preset_dropdown, prompt],
    )

    # クリアボタン処理
    def clear_fields():
        return gr.update(value=""), gr.update(value="")

    clear_btn.click(fn=clear_fields, inputs=[], outputs=[edit_name, edit_prompt])

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
        outputs=[edit_name, edit_prompt],
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
        default_presets = [
            name
            for name in choices
            if any(
                p["name"] == name and p.get("is_default", False)
                for p in presets_data["presets"]
            )
        ]
        user_presets = [name for name in choices if name not in default_presets]
        sorted_names = sorted(default_presets) + sorted(user_presets)
        updated_choices = [(name, name) for name in sorted_names]

        return result, gr.update(choices=updated_choices)

    apply_preset_btn.click(fn=apply_to_prompt, inputs=[edit_prompt], outputs=[prompt])

    delete_preset_btn.click(
        fn=delete_preset_handler,
        inputs=[preset_dropdown],
        outputs=[result_message, preset_dropdown],
    )

# enable_keyframe_copyの初期化（グローバル変数）
enable_keyframe_copy = True

allowed_paths = [
    os.path.abspath(
        os.path.realpath(os.path.join(os.path.dirname(__file__), "./outputs"))
    )
]

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

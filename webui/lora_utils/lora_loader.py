# FramePack-eichi LoRA Loader
#
# LoRAモデルの読み込みと適用のための機能を提供します。

import os
import torch
from tqdm import tqdm
from .lora_utils import merge_lora_to_state_dict
from eichi_utils import lora_state_cache

# 国際化対応
from locales.i18n_extended import translate as _

def load_and_apply_lora(
    model_files,
    lora_paths,
    lora_scales=None,
    fp8_enabled=False,
    device=None,
    cache_enabled=False
):
    """
    LoRA重みをロードして重みに適用する

    Args:
        model_files: モデルファイルのリスト
        lora_paths: LoRAファイルのパスのリスト
        lora_scales: LoRAの適用強度のリスト
        fp8_enabled: FP8最適化の有効/無効
        device: 計算に使用するデバイス
        cache_enabled: キャッシュが有効な場合、マージ済み状態辞書を保存/再利用

    Returns:
        LoRAが適用されたモデルの状態辞書
    """
    if lora_paths is None:
        lora_paths = []
    for lora_path in lora_paths:
        if not os.path.exists(lora_path):
            raise FileNotFoundError(_("LoRAファイルが見つかりません: {0}").format(lora_path))

    if lora_scales is None:
        lora_scales = [0.8] * len(lora_paths)
    if len(lora_scales)> len(lora_paths):
        lora_scales = lora_scales[:len(lora_paths)]
    if len(lora_scales) < len(lora_paths):
        lora_scales += [0.8] * (len(lora_paths) - len(lora_scales))

    if device is None:
        device = torch.device("cpu") # CPUに fall back

    cache_key = None
    if cache_enabled:
        cache_key = lora_state_cache.generate_cache_key(
            model_files, lora_paths, lora_scales, fp8_enabled
        )
        cached = lora_state_cache.load_from_cache(cache_key)
        if cached is not None:
            print(_("LoRAのFP8状態をキャッシュから読み込みました"))
            return cached

    for lora_path, lora_scale in zip(lora_paths, lora_scales):
        print(_("LoRAを読み込み中: {0} (スケール: {1})").format(os.path.basename(lora_path), lora_scale))

    # LoRA重みを状態辞書にマージ
    print(_("フォーマット: HunyuanVideo"))

    # LoRAをマージ
    merged_state_dict = merge_lora_to_state_dict(model_files, lora_paths, lora_scales, fp8_enabled, device)

    if cache_enabled and cache_key is not None:
        lora_state_cache.save_to_cache(cache_key, merged_state_dict)
        print(_("LoRAのFP8状態をキャッシュに保存しました"))

    # # LoRAが適用されたことを示すフラグを設定
    # model._lora_applied = True

    print(_("LoRAの適用が完了しました"))
    return merged_state_dict

def check_lora_applied(model):
    """
    モデルにLoRAが適用されているかをチェック

    Args:
        model: チェック対象のモデル

    Returns:
        (bool, str): LoRAが適用されているかどうかとその適用方法
    """
    # _lora_appliedフラグのチェック
    has_flag = hasattr(model, '_lora_applied') and model._lora_applied

    if has_flag:
        return True, "direct_application"

    return False, "none"

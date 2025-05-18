# FramePack-eichi LoRA Check Helper
#
# LoRAの適用状態確認のための機能を提供します。

import torch

# 国際化対応
from locales.i18n_extended import translate as _

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

    # モデル内の名前付きモジュールをチェックして、LoRAフックがあるかを確認
    has_hooks = False
    for name, module in model.named_modules():
        if hasattr(module, '_lora_hooks'):
            has_hooks = True
            break

    if has_hooks:
        return True, "hooks"

    return False, "none"

def analyze_lora_application(model):
    """
    モデルのLoRA適用率と影響を詳細に分析

    Args:
        model: 分析対象のモデル

    Returns:
        dict: 分析結果の辞書
    """
    total_params = 0
    lora_affected_params = 0

    # トータルパラメータ数とLoRAの影響を受けるパラメータ数をカウント
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            param_count = module.weight.numel()
            total_params += param_count

            # LoRA適用されたモジュールかチェック
            if hasattr(module, '_lora_hooks') or hasattr(module, '_lora_applied'):
                lora_affected_params += param_count

    # 適用率の計算
    application_rate = 0.0
    if total_params > 0:
        application_rate = lora_affected_params / total_params * 100.0

    return {
        "total_params": total_params,
        "lora_affected_params": lora_affected_params,
        "application_rate": application_rate,
        "has_lora": lora_affected_params > 0
    }

def print_lora_status(model):
    """
    モデルのLoRA適用状況を出力

    Args:
        model: 出力対象のモデル
    """
    has_lora, source = check_lora_applied(model)

    if has_lora:
        print(_("LoRAステータス: {0}").format(_("適用済み")))
        print(_("適用方法: {0}").format(_(source)))

        # 詳細な分析
        analysis = analyze_lora_application(model)
        application_rate = analysis["application_rate"]

        print(_("LoRA適用状況: {0}/{1} パラメータ ({2:.2f}%)").format(
            analysis["lora_affected_params"],
            analysis["total_params"],
            application_rate
        ))
    else:
        print(_("LoRAステータス: {0}").format(_("未適用")))
        print(_("モデルにLoRAは適用されていません"))

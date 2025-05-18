"""
FramePack-eichi FP8最適化モジュール
モデルを8ビット浮動小数点形式に量子化して、メモリ使用量と処理速度を最適化するモジュールです。
FramePack-LoRAReadyから移植されています。

基本的な特徴:
- E4M3およびE5M2 FP8フォーマットのサポート
- 異なるGPUアーキテクチャに対する最適化
- モンキーパッチによる透過的な統合
- RTX 40シリーズ向けのscaled_mm最適化対応
"""

import os
import torch

# 警告メッセージが表示されたかを追跡するフラグ
FP8_E4M3_WARNING_SHOWN = False
FP8_DIMENSIONS_WARNING_SHOWN = False
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 国際化対応
from locales.i18n_extended import translate

def calculate_fp8_maxval(exp_bits=4, mantissa_bits=3, sign_bits=1):
    """
    FP8形式で表現可能な最大値を計算
    デフォルトはE4M3形式（4ビット指数部、3ビット仮数部、1ビット符号部）

    Args:
        exp_bits (int): 指数部のビット数
        mantissa_bits (int): 仮数部のビット数
        sign_bits (int): 符号部のビット数（0または1）

    Returns:
        float: FP8形式で表現可能な最大値
    """
    assert exp_bits + mantissa_bits + sign_bits == 8, translate("合計ビット数は8でなければなりません")

    # 指数バイアスを計算
    bias = 2 ** (exp_bits - 1) - 1

    # 最大仮数値を計算
    mantissa_max = 1.0
    for i in range(mantissa_bits - 1):
        mantissa_max += 2 ** -(i + 1)

    # 最大値を計算
    max_value = mantissa_max * (2 ** (2**exp_bits - 1 - bias))

    return max_value

def quantize_tensor_to_fp8(tensor, scale, exp_bits=4, mantissa_bits=3, sign_bits=1, max_value=None, min_value=None):
    """
    テンソルをFP8形式に量子化する

    Args:
        tensor (torch.Tensor): 量子化するテンソル
        scale (float or torch.Tensor): スケールファクター
        exp_bits (int): 指数部のビット数
        mantissa_bits (int): 仮数部のビット数
        sign_bits (int): 符号部のビット数
        max_value (float, optional): 最大値（Noneの場合は自動計算）
        min_value (float, optional): 最小値（Noneの場合は自動計算）

    Returns:
        tuple: (量子化されたテンソル, スケールファクター)
    """
    # スケーリングされたテンソルを作成
    scaled_tensor = tensor / scale

    # FP8パラメータを計算
    bias = 2 ** (exp_bits - 1) - 1

    if max_value is None:
        # 最大値と最小値を計算
        max_value = calculate_fp8_maxval(exp_bits, mantissa_bits, sign_bits)
        min_value = -max_value if sign_bits > 0 else 0.0

    # テンソルを範囲内に制限
    clamped_tensor = torch.clamp(scaled_tensor, min_value, max_value)

    # 量子化プロセス
    abs_values = torch.abs(clamped_tensor)
    nonzero_mask = abs_values > 0

    # logFスケールを計算（非ゼロ要素のみ）
    log_scales = torch.zeros_like(clamped_tensor)
    if nonzero_mask.any():
        log_scales[nonzero_mask] = torch.floor(torch.log2(abs_values[nonzero_mask]) + bias).detach()

    # logスケールを制限し、量子化係数を計算
    log_scales = torch.clamp(log_scales, min=1.0)
    quant_factor = 2.0 ** (log_scales - mantissa_bits - bias)

    # 量子化と逆量子化
    quantized = torch.round(clamped_tensor / quant_factor) * quant_factor

    return quantized, scale

def optimize_state_dict_with_fp8_on_the_fly(
    model_files,
    calc_device,
    target_layer_keys=None,
    exclude_layer_keys=None,
    exp_bits=4,
    mantissa_bits=3,
    move_to_device=False,
    weight_hook=None,
):
    """
    モデルの状態辞書内の線形レイヤーの重みをFP8形式に最適化

    Args:
        model_files (list): 最適化するモデルファイルのリスト（読み込みながら更新）
        calc_device (str): テンソルを量子化するデバイス
        target_layer_keys (list, optional): 対象とするレイヤーキーのパターン（Noneの場合はすべての線形レイヤー）
        exclude_layer_keys (list, optional): 除外するレイヤーキーのパターン
        exp_bits (int): 指数部のビット数
        mantissa_bits (int): 仮数部のビット数
        move_to_device (bool): 最適化されたテンソルを計算デバイスに移動するかどうか
        weight_hook (callable, optional): 重みのフック関数（Noneの場合は使用しない）、FP8最適化前に、FP8最適化の有無に関係なくすべての重みに適用される。

    Returns:
        dict: FP8最適化された状態辞書
    """
    # FP8データ型の選択
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(translate("サポートされていないFP8形式: E{0}M{1}").format(exp_bits, mantissa_bits))

    # FP8の最大値を計算
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # この関数は符号付きFP8のみサポート

    # 最適化された状態辞書を作成する

    def is_target_key(key):
        # 重みキーが対象パターンに一致し、除外パターンに一致しないかを確認
        is_target = (target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        is_target = is_target and not is_excluded
        return is_target

    # 最適化されたレイヤーのカウンター
    optimized_count = 0
    
    # それぞれのモデルファイルを処理        
    from lora_utils.safetensors_utils import MemoryEfficientSafeOpen

    state_dict = {}
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            keys = f.keys()
            for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                value = f.get_tensor(key)
                if weight_hook is not None:
                    # 重みフックが指定されている場合、フックを適用
                    value = weight_hook(key, value)

                if not is_target_key(key):
                    state_dict[key] = value
                    continue

                # 元のデバイスとデータ型を保存
                original_device = value.device
                original_dtype = value.dtype

                # 計算デバイスに移動
                if calc_device is not None:
                    value = value.to(calc_device)

                # スケールファクターを計算
                scale = torch.max(torch.abs(value.flatten())) / max_value

                # 重みをFP8に量子化
                quantized_weight, _ = quantize_tensor_to_fp8(value, scale, exp_bits, mantissa_bits, 1, max_value, min_value)

                # 重みに元のキー、スケールに新しいキーを使用
                fp8_key = key
                scale_key = key.replace(".weight", ".scale_weight")

                # FP8データ型に変換
                quantized_weight = quantized_weight.to(fp8_dtype)

                # デバイスの指定がない場合は元のデバイスに戻す
                if not move_to_device:
                    quantized_weight = quantized_weight.to(original_device)

                # スケールテンソルを作成
                scale_tensor = torch.tensor([scale], dtype=original_dtype, device=quantized_weight.device)

                # 状態辞書に追加
                state_dict[fp8_key] = quantized_weight
                state_dict[scale_key] = scale_tensor

                optimized_count += 1

                # 計算デバイスのメモリを定期的に解放
                if calc_device is not None  and optimized_count % 10 == 0:
                    torch.cuda.empty_cache()

    print(translate("最適化された線形レイヤー数: {0}").format(optimized_count))
    return state_dict

def fp8_linear_forward_patch(self: nn.Linear, x, use_scaled_mm=False, max_value=None):
    """
    FP8重みを持つ線形レイヤー用のパッチ適用済みフォワードメソッド

    Args:
        self: 線形レイヤーのインスタンス
        x (torch.Tensor): 入力テンソル
        use_scaled_mm (bool): FP8線形レイヤーに scaled_mm を使用するかどうか（SM 8.9+、RTX 40シリーズが必要）
        max_value (float): FP8量子化の最大値（Noneの場合、入力テンソルに量子化は適用されない）

    Returns:
        torch.Tensor: 線形変換の結果
    """
    if use_scaled_mm:
        # scaled_mmを使用する場合（RTX 40シリーズのGPUでのみ動作）
        input_dtype = x.dtype
        original_weight_dtype = self.scale_weight.dtype
        weight_dtype = self.weight.dtype
        target_dtype = torch.float8_e5m2

        # E4M3FNでない場合は通常方式にフォールバック
        # scaled_mmはFP8でもE4M3FN形式のみ対応しているため、他の形式では使用不可
        global FP8_E4M3_WARNING_SHOWN
        if weight_dtype != torch.float8_e4m3fn:
            if not FP8_E4M3_WARNING_SHOWN:
                print(translate("警告: scaled_mmはFP8 E4M3FN形式を必要としますが、{weight_dtype}が検出されました。通常方式にフォールバックします。").format(weight_dtype=weight_dtype))
                FP8_E4M3_WARNING_SHOWN = True
            # 通常の方式にフォールバック
            return fp8_linear_forward_patch(self, x, False, max_value)

        # 入力テンソルの次元チェック
        # scaled_mmは3次元テンソル（batch_size, seq_len, hidden_dim）を想定しているため、それ以外では機能しない
        global FP8_DIMENSIONS_WARNING_SHOWN
        if x.ndim != 3:
            if not FP8_DIMENSIONS_WARNING_SHOWN:
                print(translate("警告: scaled_mmは3次元入力が必要ですが、{0}次元が検出されました。通常方式にフォールバックします。").format(x.ndim))
                FP8_DIMENSIONS_WARNING_SHOWN = True
            # 通常の方式にフォールバック
            return fp8_linear_forward_patch(self, x, False, max_value)

        if max_value is None:
            # 入力の量子化なし
            scale_x = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        else:
            # 入力テンソルのスケールファクターを計算
            scale_x = (torch.max(torch.abs(x.flatten())) / max_value).to(torch.float32)

            # 入力テンソルをFP8に量子化（メモリを大量に消費する可能性あり）
            x, _ = quantize_tensor_to_fp8(x, scale_x, 5, 2, 1, max_value, -max_value)

        # テンソルの形状を変換
        original_shape = x.shape
        x = x.reshape(-1, x.shape[2]).to(target_dtype)

        # 重みを転置
        weight = self.weight.t()
        scale_weight = self.scale_weight.to(torch.float32)

        # scaled_mmを使用して計算（バイアス有無で処理を分ける）
        if self.bias is not None:
            # バイアスがある場合、float32はサポートされていません
            o = torch._scaled_mm(x, weight, out_dtype=original_weight_dtype, bias=self.bias, scale_a=scale_x, scale_b=scale_weight)
        else:
            o = torch._scaled_mm(x, weight, out_dtype=input_dtype, scale_a=scale_x, scale_b=scale_weight)

        # 元の形状に戻して返す
        return o.reshape(original_shape[0], original_shape[1], -1).to(input_dtype)
    else:
        # 通常の方式（重みを逆量子化して計算）
        original_dtype = self.scale_weight.dtype
        dequantized_weight = self.weight.to(original_dtype) * self.scale_weight

        # 線形変換を実行
        if self.bias is not None:
            output = F.linear(x, dequantized_weight, self.bias)
        else:
            output = F.linear(x, dequantized_weight)

        return output

def apply_fp8_monkey_patch(model, optimized_state_dict, use_scaled_mm=False):
    """
    FP8最適化された状態辞書を使用してモデルにモンキーパッチを適用

    Args:
        model (nn.Module): パッチを適用するモデルインスタンス
        optimized_state_dict (dict): FP8最適化された状態辞書
        use_scaled_mm (bool): FP8線形レイヤーに scaled_mm を使用するかどうか（SM 8.9+、RTX 40シリーズが必要）

    Returns:
        nn.Module: パッチが適用されたモデル（同じインスタンスをインプレースで修正）
    """
    # 入力テンソルの量子化には使用しない（デフォルト）
    max_value = None

    # スケールキーを見つけてFP8最適化されたレイヤーを特定
    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight")]

    # パッチ適用済みモジュールのパスを設定
    patched_module_paths = set()
    for scale_key in scale_keys:
        # スケールキーからモジュールパスを抽出（".scale_weight"を削除）
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)

    patched_count = 0

    # FP8重みを持つ各レイヤーにモンキーパッチを適用
    for name, module in model.named_modules():
        # このモジュールに対応するスケール重みがあるかチェック
        has_scale = name in patched_module_paths

        # FP8スケールを持つ線形レイヤーにパッチを適用
        if isinstance(module, nn.Linear) and has_scale:
            # スケール重みをバッファとして登録（状態辞書をロードするため）
            module.register_buffer("scale_weight", torch.tensor(1.0, dtype=module.weight.dtype))

            # パッチ適用済みのフォワードメソッドを作成
            def new_forward(self, x):
                return fp8_linear_forward_patch(self, x, use_scaled_mm, max_value)

            # メソッドをモジュールにバインド
            module.forward = new_forward.__get__(module, type(module))

            patched_count += 1

    print(translate("モンキーパッチ適用済みの線形レイヤー数: {0}").format(patched_count))
    # モデルにFP8適用済みフラグを設定
    model._fp8_optimized = True
    return model

def check_fp8_support():
    """
    FP8サポートをチェックする関数

    Returns:
        tuple: (E4M3サポート, E5M2サポート, scaled_mmサポート)
    """
    # FP8サポートのチェック
    has_e4m3 = hasattr(torch, 'float8_e4m3fn')
    has_e5m2 = hasattr(torch, 'float8_e5m2')

    # scaled_mm サポートのチェック
    has_scaled_mm = hasattr(torch, '_scaled_mm')

    if has_e4m3 and has_e5m2:
        print(translate("FP8サポート検出: E4M3およびE5M2フォーマットが利用可能です"))
        if has_scaled_mm:
            print(translate("scaled_mmサポート検出: RTX 40シリーズのGPUでFP8の高速化が可能です"))
    else:
        print(translate("警告: FP8サポートが検出されませんでした。PyTorch 2.1以上が必要です"))

    return has_e4m3, has_e5m2, has_scaled_mm

def reset_fp8_warning_flags():
    """
    FP8警告フラグをリセットする関数
    各生成処理の開始時に呼び出すことで、生成ごとに警告を表示できるようにする
    """
    global FP8_E4M3_WARNING_SHOWN, FP8_DIMENSIONS_WARNING_SHOWN
    FP8_E4M3_WARNING_SHOWN = False
    FP8_DIMENSIONS_WARNING_SHOWN = False


def reset_warning_flags():
    """
    警告フラグをリセットする関数（新しい生成プロセスが開始されるたびに呼び出す）
    """
    global FP8_E4M3_WARNING_SHOWN, FP8_DIMENSIONS_WARNING_SHOWN
    FP8_E4M3_WARNING_SHOWN = False
    FP8_DIMENSIONS_WARNING_SHOWN = False

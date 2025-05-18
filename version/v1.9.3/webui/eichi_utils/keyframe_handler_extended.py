"""
拡張キーフレーム処理モジュール
キーフレーム処理に関する拡張機能を提供します
"""

import gradio as gr

from locales.i18n_extended import translate

from eichi_utils.video_mode_settings import (
    get_total_sections,
    get_important_keyframes,
    get_video_seconds,
    MODE_TYPE_LOOP
)
from eichi_utils.keyframe_handler import code_to_ui_index, get_max_keyframes_count
from eichi_utils.frame_calculator import calculate_sections_for_mode_and_size

def extended_mode_length_change_handler(mode, length, section_number_inputs, section_row_groups=None, frame_size_setting="_KEY_FRAME_SIZE_1SEC", enable_keyframe_copy=None):
    """モードと動画長の変更を統一的に処理する関数（セクション行の表示/非表示制御を追加）

    Args:
        mode: モード ("通常" or "ループ")
        length: 動画長 ("6秒", "8秒", "10秒", "12秒", "16秒", "20秒")
        section_number_inputs: セクション番号入力欄のリスト
        section_row_groups: セクション行のUIグループリスト（オプション）
        frame_size_setting: フレームサイズ設定 ("1秒 (33フレーム)" or "0.5秒 (17フレーム)")
        enable_keyframe_copy: キーフレーム自動コピー機能の有効/無効状態（オプション）

    Returns:
        更新リスト: 各UI要素の更新情報のリスト
    """
    # 通常モードでは全ての赤枠青枠を強制的に非表示にする処理を追加
    is_loop_mode = (mode == MODE_TYPE_LOOP)
    
    # キーフレーム自動コピー機能の状態を確認
    # モードとコピー有効フラグの両方が指定されている場合のみコピーを有効にする
    if is_loop_mode and enable_keyframe_copy is not None:
        is_copy_enabled = enable_keyframe_copy
        print(translate("[keyframe_handler_extended] ループモードでキーフレーム自動コピー機能の状態: {state}").format(state=is_copy_enabled))
    else:
        # デフォルト状態を使用 - ループモードならTrue、それ以外はFalse
        is_copy_enabled = is_loop_mode
        print(translate("[keyframe_handler_extended] モード変更によるキーフレーム自動コピー機能のデフォルト状態: {state}").format(state=is_copy_enabled))
    
    if not is_loop_mode:
        print(translate("[keyframe_handler_extended] 通常モードで強制的に赤枠/青枠を非表示に設定"))
    else:
        print(translate("[keyframe_handler_extended] ループモードで赤枠/青枠を表示可能に設定"))
    
    # 基本要素のクリア（入力画像と終了フレーム）
    updates = [gr.update(value=None) for _ in range(2)]

    # すべてのキーフレーム画像をクリア
    section_image_count = get_max_keyframes_count()
    for _ in range(section_image_count):
        updates.append(gr.update(value=None, elem_classes=""))

    # セクション番号ラベルをリセット
    for i in range(len(section_number_inputs)):
        section_number_inputs[i].elem_classes = ""

    # 重要なキーフレームを強調表示
    important_kfs = get_important_keyframes(length)
    for idx in important_kfs:
        ui_idx = code_to_ui_index(idx)
        update_idx = ui_idx + 1  # 入力画像と終了フレームの2つを考慮
        if update_idx < len(updates):
            # 通常モードの場合はすべての枠を非表示にする
            if not is_loop_mode:
                updates[update_idx] = gr.update(value=None, elem_classes="")
                if idx < len(section_number_inputs):
                    section_number_inputs[idx].elem_classes = ""
                continue

            # ループモードでもキーフレーム自動コピー機能が無効なら枠を表示しない
            if not is_copy_enabled:
                updates[update_idx] = gr.update(value=None, elem_classes="")
                if idx < len(section_number_inputs):
                    section_number_inputs[idx].elem_classes = ""
                continue

            # ループモードでキーフレーム自動コピー機能が有効の場合のみセクションによって枠の色を変える
            if idx == 0:
                # セクション0は赤枠
                updates[update_idx] = gr.update(value=None, elem_classes="highlighted-keyframe-red")
                if idx < len(section_number_inputs):
                    section_number_inputs[idx].elem_classes = "highlighted-label-red"
            elif idx == 1:
                # セクション1は青枠
                updates[update_idx] = gr.update(value=None, elem_classes="highlighted-keyframe-blue")
                if idx < len(section_number_inputs):
                    section_number_inputs[idx].elem_classes = "highlighted-label-blue"
            else:
                # その他のセクションは通常の枠
                updates[update_idx] = gr.update(value=None, elem_classes="highlighted-keyframe")
                if idx < len(section_number_inputs):
                    section_number_inputs[idx].elem_classes = "highlighted-label"

    # ループモードの場合はキーフレーム0も強調（まだ強調されていない場合）
    # セクション0は赤枠にする - ループモードでコピー機能が有効の場合のみ
    if is_loop_mode and is_copy_enabled and 0 not in important_kfs:
        print(translate("[keyframe_handler_extended] ループモードでセクション0に赤枠を適用"))
        updates[2] = gr.update(value=None, elem_classes="highlighted-keyframe-red")
        if 0 < len(section_number_inputs):
            section_number_inputs[0].elem_classes = "highlighted-label-red"
    elif not (is_loop_mode and is_copy_enabled):
        print(translate("[keyframe_handler_extended] モードまたはコピー機能状態により、セクション0の赤枠を適用せず"))
        # 通常モードまたはコピー機能が無効の場合は強制的にクリア
        updates[2] = gr.update(value=None, elem_classes="")

    # 動画長の設定
    video_length = get_video_seconds(length)

    # 最終的な動画長設定を追加
    updates.append(gr.update(value=video_length))

    # セクション行の表示/非表示を制御
    if section_row_groups is not None:
        # 動画モードとフレームサイズから必要なセクション数を計算
        required_sections = calculate_sections_for_mode_and_size(length, frame_size_setting)
        print(translate("動画モード '{length}' とフレームサイズ '{frame_size_setting}' で必要なセクション数: {required_sections}").format(length=length, frame_size_setting=frame_size_setting, required_sections=required_sections))

        # 各セクション行の表示/非表示を設定
        row_updates = []
        for i, _ in enumerate(section_row_groups):
            if i < required_sections:
                row_updates.append(gr.update(visible=True))
            else:
                row_updates.append(gr.update(visible=False))

        # 更新リストに行の表示/非表示設定を追加
        updates.extend(row_updates)
    
    # キーフレーム自動コピー機能のチェックボックス更新は直接出力に追加しない
    # 呼び出し元で必要なら別途追加する方が安全
    # （既存の出力要素数が変更されると表示崩れの原因になる）
        
    return updates

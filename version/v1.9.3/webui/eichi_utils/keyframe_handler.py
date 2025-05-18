"""
キーフレーム処理モジュール
endframe_ichi.pyから外出しされたキーフレーム関連処理を含む
"""

import gradio as gr
from locales.i18n_extended import translate

from eichi_utils.video_mode_settings import (
    get_max_keyframes_count,
    get_copy_targets,
    get_important_keyframes,
    get_video_seconds,
    MODE_TYPE_LOOP
)

# インデックス変換のユーティリティ関数
def ui_to_code_index(ui_index):
    """UI表示のキーフレーム番号(1始まり)をコード内インデックス(0始まり)に変換"""
    return ui_index - 1

def code_to_ui_index(code_index):
    """コード内インデックス(0始まり)をUI表示のキーフレーム番号(1始まり)に変換"""
    return code_index + 1

# 1. 統一的なキーフレーム変更ハンドラ
def unified_keyframe_change_handler(keyframe_idx, img, mode, length, enable_copy=True, dynamic_sections=None):
    """すべてのキーフレーム処理を統一的に行う関数

    Args:
        keyframe_idx: UIのキーフレーム番号-1 (0始まりのインデックス)
        img: 変更されたキーフレーム画像
        mode: モード ("通常" or "ループ")
        length: 動画長 ("6秒", "8秒", "10秒", "12秒", "16秒", "20秒")
        enable_copy: コピー機能が有効かどうか

    Returns:
        更新リスト: 変更するキーフレーム画像の更新情報のリスト
    """
    # 通常モードでは常にコピー機能を無効化
    if mode != MODE_TYPE_LOOP:
        enable_copy = False
        print(translate("[keyframe_handler] 通常モードでコピー機能を強制的に無効化"))

    if img is None or not enable_copy:
        # 画像が指定されていない、またはコピー機能が無効の場合は何もしない
        max_keyframes = get_max_keyframes_count()
        remaining = max(0, max_keyframes - keyframe_idx - 1)
        return [gr.update() for _ in range(remaining)]

    # 動画長から実際のセクション数を取得
    # セクション数の決定: 動的に計算された値があればそちらを優先
    if dynamic_sections is not None:
        sections = dynamic_sections
        print(translate("[keyframe_handler] 動的に計算されたセクション数を使用: {sections}").format(sections=sections))
    else:
        # video_mode_settings.pyのデフォルト値を取得
        sections = get_total_sections(length)
        print(translate("[keyframe_handler] 設定からのデフォルトセクション数を使用: {sections}").format(sections=sections))
    print(translate("[keyframe_handler] 動画長 {length} のセクション数: {sections}").format(length=length, sections=sections))

    # 赤枠（セクション0）からのコピー先を計算
    targets = []
    if keyframe_idx == 0:
        # 2から始まる偶数番号を、実際のセクション数の範囲内でリストに追加
        targets = [i for i in range(2, sections) if i % 2 == 0]
        print(translate("[keyframe_handler] 赤枠(セクション0)から偶数番号へのコピーターゲット: {targets} (動的セクション数: {sections})").format(targets=targets, sections=sections))
        # コピー先が空の場合はログに出力
        if not targets:
            print(translate("[keyframe_handler] 赤枠(セクション0)からのコピー先がありません。セクション数が少ない可能性があります。"))

    # 青枠（セクション1）からのコピー先を計算
    elif keyframe_idx == 1:
        # 3から始まる奇数番号を、実際のセクション数の範囲内でリストに追加
        targets = [i for i in range(3, sections) if i % 2 == 1]
        print(translate("[keyframe_handler] 青枠(セクション1)から奇数番号へのコピーターゲット: {targets} (動的セクション数: {sections})").format(targets=targets, sections=sections))
        # コピー先が空の場合はログに出力
        if not targets:
            print(translate("[keyframe_handler] 青枠(セクション1)からのコピー先がありません。セクション数が少ない可能性があります。"))

    # 結果の更新リスト作成
    max_keyframes = get_max_keyframes_count()
    updates = []

    # このキーフレーム以降のインデックスに対してのみ処理
    for i in range(keyframe_idx + 1, max_keyframes):
        # 現在の処理対象のセクションがセクション数を超えた場合はスキップ
        if i >= sections:
            updates.append(gr.update())
            continue

        # コピーパターン定義では相対インデックスでなく絶対インデックスが使われているため、
        # iがtargets内にあるかをチェック
        if i in targets:
            # コピー先リストに含まれている場合は画像をコピー
            print(translate("[keyframe_handler] セクション{i}へ画像をコピーします").format(i=i))
            updates.append(gr.update(value=img))
        else:
            # 含まれていない場合は変更なし
            updates.append(gr.update())

    return updates

# 2. モード変更の統一ハンドラ
def unified_mode_length_change_handler(mode, length, section_number_inputs):
    """モードと動画長の変更を統一的に処理する関数

    Args:
        mode: モード ("通常" or "ループ")
        length: 動画長 ("6秒", "8秒", "10秒", "12秒", "16秒")
        section_number_inputs: セクション番号入力欄のリスト

    Returns:
        更新リスト: 各UI要素の更新情報のリスト
    """
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
            # セクションによって枠の色を変える
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

    # ループモードの場合はキーフレーム1も強調（まだ強調されていない場合）
    # セクション0は赤枠にする
    if mode == MODE_TYPE_LOOP and 0 not in important_kfs:
        updates[2] = gr.update(value=None, elem_classes="highlighted-keyframe-red")
        if 0 < len(section_number_inputs):
            section_number_inputs[0].elem_classes = "highlighted-label-red"

    # 動画長の設定
    video_length = get_video_seconds(length)

    # 最終的な動画長設定を追加
    updates.append(gr.update(value=video_length))

    return updates

# 3. 入力画像変更の統一ハンドラ
def unified_input_image_change_handler(img, mode, length, enable_copy=True):
    """入力画像変更時の処理を統一的に行う関数

    Args:
        img: 変更された入力画像
        mode: モード ("通常" or "ループ")
        length: 動画長 ("6秒", "8秒", "10秒", "12秒", "16秒", "20秒")
        enable_copy: コピー機能が有効かどうか

    Returns:
        更新リスト: 終了フレームとすべてのキーフレーム画像の更新情報のリスト
    """
    # 通常モードでは常にコピー機能を無効化
    if mode != MODE_TYPE_LOOP:
        enable_copy = False
        print(translate("[keyframe_handler] 入力画像変更時に通常モードでコピー機能を強制的に無効化"))

    if img is None or not enable_copy:
        # 画像が指定されていない、またはコピー機能が無効の場合は何もしない
        section_count = get_max_keyframes_count()
        return [gr.update() for _ in range(section_count + 1)]  # +1 for end_frame

    # ループモードかどうかで処理を分岐
    if mode == MODE_TYPE_LOOP:
        # ループモード: FinalFrameに入力画像をコピー
        updates = [gr.update(value=img)]  # end_frame

        # キーフレーム画像は更新なし
        section_count = get_max_keyframes_count()
        updates.extend([gr.update() for _ in range(section_count)])

    else:
        # 通常モード: FinalFrameは更新なし
        updates = [gr.update()]  # end_frame

        # 動画長/モードに基づいてコピー先のキーフレームを取得
        # これが設定ファイルに基づく方法
        copy_targets = []

        # 特殊処理のモードでは設定によって異なるキーフレームにコピー
        if length == translate("10秒"):
            # 10秒の場合は5～8にコピー (インデックス4-7)
            copy_targets = [4, 5, 6, 7]
        elif length == translate("12秒"):
            # 12秒の場合は7～9にコピー (インデックス6-8)
            copy_targets = [6, 7, 8]
        elif length == translate("16秒"):
            # 16秒の場合は10～12にコピー (インデックス9-11)
            copy_targets = [9, 10, 11]
        elif length == translate("20秒"):
            # 20秒の場合は13～15にコピー (インデックス12-14)
            copy_targets = [12, 13, 14]
        else:
            # 通常の動画長の場合は最初のいくつかのキーフレームにコピー
            if length == translate("6秒"):
                copy_targets = [0, 1, 2, 3]  # キーフレーム1-4
            elif length == translate("8秒"):
                copy_targets = [0, 1, 2, 3, 4, 5]  # キーフレーム1-6

        # キーフレーム画像の更新リスト作成
        section_count = get_max_keyframes_count()
        for i in range(section_count):
            if i in copy_targets:
                updates.append(gr.update(value=img))
            else:
                updates.append(gr.update())

    return updates

# 4. デバッグ情報表示関数
def print_keyframe_debug_info():
    """キーフレーム設定の詳細情報を表示"""
    # print("\n[INFO] =========== キーフレーム設定デバッグ情報 ===========")
    #
    # # 設定内容の確認表示
    # print("\n[INFO] 動画モード設定の確認:")
    # for mode_key in VIDEO_MODE_SETTINGS:
    #     mode_info = VIDEO_MODE_SETTINGS[mode_key]
    #     print(translate("  - {0}: {1}秒, {2}フレーム").format(mode_key, mode_info['display_seconds'], mode_info['frames']))
    #
    #     # 重要キーフレームの表示（UIインデックスに変換）
    #     important_kfs = mode_info['important_keyframes']
    #     important_kfs_ui = [code_to_ui_index(kf) for kf in important_kfs]
    #     print(translate("    重要キーフレーム: {0}").format(important_kfs_ui))
    #
    #     # コピーパターンの表示
    #     for mode_type in ["通常", "ループ"]:
    #         if mode_type in mode_info["copy_patterns"]:
    #             print(translate("    {0}モードのコピーパターン:").format(mode_type))
    #             for src, targets in mode_info["copy_patterns"][mode_type].items():
    #                 src_ui = code_to_ui_index(int(src))
    #                 targets_ui = [code_to_ui_index(t) for t in targets]
    #                 print(translate("      キーフレーム{0} → {1}").format(src_ui, targets_ui))
    #
    # print("[INFO] =================================================\n")
    pass

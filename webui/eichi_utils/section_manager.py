"""
セクション情報の一括操作（アップロード/ダウンロード）を管理するモジュール
"""
import os
import yaml
import zipfile
import time
import shutil
from datetime import datetime
import tempfile

from locales.i18n import translate

# セクション情報一時ディレクトリ
TEMP_DIR = "./temp_for_zip_section_info"

def process_uploaded_zipfile(file, max_keyframes):
    """
    アップロードされたzipファイルからセクション情報を抽出する

    Args:
        file: アップロードされたzipファイル
        max_keyframes: キーフレームの最大数

    Returns:
        dict: セクション情報のリスト（各セクションの番号、プロンプト、画像パス）とエンド/スタートフレーム
    """
    if file is None:
        # Noneの場合、空の結果を返す
        return {
            "section_numbers": [i for i in range(max_keyframes)],
            "section_prompts": ["" for _ in range(max_keyframes)],
            "section_images": [None for _ in range(max_keyframes)],
            "end_frame": None,
            "start_frame": None,
            "lora_settings": None
        }

    # 一時ディレクトリで処理
    # temp_dir配下のフォルダを削除（前回アップロードファイルをクリア）
    if os.path.exists(TEMP_DIR):
        for root, dirs, files in os.walk(TEMP_DIR, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(TEMP_DIR)

    # zip展開
    with zipfile.ZipFile(file.name, "r") as zip_ref:
        zip_ref.extractall(TEMP_DIR)

    # 展開されたファイルをフルパスでリストアップ - 完全スキャンに変更
    extracted_files = []
    # print(f"[IMPORTANT] ZIPファイルの抽出を開始: {file.name}")
    
    # すべてのディレクトリを再帰的にスキャン
    for root, dirs, files in os.walk(TEMP_DIR):
        if files:  # ファイルがあればリストに追加
            for f in files:
                file_path = os.path.join(root, f)
                extracted_files.append(file_path)
                # print(f"[DEBUG-EXTRACT] ファイル抽出: {file_path}")
    
    # ディレクトリ構造の確認
    # print(f"[DEBUG-STRUCTURE] ディレクトリ構造:")
    # for root, dirs, files in os.walk(TEMP_DIR):
    #     level = root.replace(TEMP_DIR, '').count(os.sep)
    #     indent = ' ' * 4 * level
    #     print(f"{indent}{os.path.basename(root)}/")
    #     sub_indent = ' ' * 4 * (level + 1)
    #     for f in files:
    #         print(f"{sub_indent}{f}")
    
    # 結果の確認
    # print(f"[DEBUG-FILES] 抽出されたファイル数: {len(extracted_files)}")
    if not extracted_files:
        print(f"[WARNING] 抽出されたファイルがありません！")

    # プロンプトファイルを取得（1つのみ）
    prompt_files = [f for f in extracted_files if f.endswith("sections.yml") or f.endswith("sections.yaml")]
    if not prompt_files:
        raise ValueError(translate("zipファイルにsections.yaml（またはsections.yml）が見つかりません"))
    prompt_file = prompt_files[0]

    # 画像ファイルを取得
    image_files = [f for f in extracted_files if f.lower().endswith((".png", ".jpeg", ".jpg", ".webp"))]
    # print(f"[DEBUG-IMAGES] 見つかった画像ファイル数: {len(image_files)}")
    
    # 画像ファイルとその名前を詳細出力
    for i, img_file in enumerate(image_files):
        file_name = os.path.basename(img_file)
        # print(f"[DEBUG-IMAGE-DETAIL] 画像{i}: {file_name} -> {img_file}")
        
        # 重要：end.pngとstart.pngの特別チェック
        # if file_name.lower() == "end.png":
        #     print(f"[CRITICAL] end.pngを発見: {img_file}")
        # elif file_name.lower() == "start.png":
        #     print(f"[CRITICAL] start.pngを発見: {img_file}")
    
    # セクション用画像のファイルを取得しソートする。ファイル名は3桁の0始まりの数字とする。
    section_image_files = sorted([f for f in image_files if os.path.basename(f)[:3].isdigit()])
    
    # end_frame、start_frame向けの画像ファイルを取得
    end_frame_image_from_zip = None
    start_frame_image_from_zip = None
    
    # すべての画像ファイルをデバッグ出力
    # print(f"[DEBUG-FILES] 抽出されたすべての画像ファイル:")
    # for i, img_file in enumerate(image_files):
    #     print(f"  {i}: {img_file} (ファイル名: {os.path.basename(img_file)})")
    
    # エンドフレーム画像の検索を強化
    # 1. 'end'で始まる場合
    end_files = [f for f in image_files if os.path.basename(f).lower().startswith("end")]
    # 2. 'end.png'や'end.jpg'など
    if not end_files:
        end_files = [f for f in image_files if os.path.basename(f).lower() == "end.png" or 
                                             os.path.basename(f).lower() == "end.jpg" or
                                             os.path.basename(f).lower() == "end.jpeg" or 
                                             os.path.basename(f).lower() == "end.webp"]
    # 3. ファイル名を含むディレクトリ（例：sections/end.png）
    if not end_files:
        end_files = [f for f in image_files if "end.png" in f.lower() or 
                                             "end.jpg" in f.lower() or
                                             "end.jpeg" in f.lower() or 
                                             "end.webp" in f.lower()]
    
    if len(end_files) > 0:
        end_frame_image_from_zip = end_files[0]
        # print(f"[IMPORTANT] ZIPファイルからエンドフレーム画像を検出: {end_frame_image_from_zip}")
    # else:
    #     print(f"[WARNING] エンドフレーム画像が見つかりませんでした。すべての画像ファイル: {[os.path.basename(f) for f in image_files]}")
    
    # スタートフレーム画像の検索を強化
    # 1. 'start'で始まる場合
    start_files = [f for f in image_files if os.path.basename(f).lower().startswith("start")]
    # 2. 'start.png'や'start.jpg'など
    if not start_files:
        start_files = [f for f in image_files if os.path.basename(f).lower() == "start.png" or 
                                               os.path.basename(f).lower() == "start.jpg" or
                                               os.path.basename(f).lower() == "start.jpeg" or 
                                               os.path.basename(f).lower() == "start.webp"]
    # 3. ファイル名を含むディレクトリ（例：sections/start.png）
    if not start_files:
        start_files = [f for f in image_files if "start.png" in f.lower() or 
                                               "start.jpg" in f.lower() or
                                               "start.jpeg" in f.lower() or 
                                               "start.webp" in f.lower()]
                                                
    if len(start_files) > 0:
        start_frame_image_from_zip = start_files[0]
        # print(f"[IMPORTANT] ZIPファイルからスタートフレーム画像を検出: {start_frame_image_from_zip}")
    # else:
    #     print(f"[WARNING] スタートフレーム画像が見つかりませんでした。すべての画像ファイル: {[os.path.basename(f) for f in image_files]}")

    # プロンプトファイルを読み込んでセクションプロンプトに設定
    with open(prompt_file, "r", encoding="utf-8") as file:
        prompt_data = yaml.safe_load(file)

    # セクション入力情報（zipファイルから取得した情報）
    section_number_list_from_zip = []
    section_image_list_from_zip = []
    section_prompt_list_from_zip = []

    # yamlファイルのsection_infoからプロンプトを抽出してリスト化
    for section_num in range(0, max_keyframes):
        section_number_list_from_zip.append(section_num)
        section_prompt_list_from_zip.append(next((section["prompt"] for section in prompt_data.get("section_info", []) if section.get("section") == section_num), ""))

    # デフォルトプロンプトとSEED値を取得
    default_prompt_from_yaml = prompt_data.get("default_prompt", "")
    seed_from_yaml = prompt_data.get("SEED", -1)
    # print(f"[IMPORTANT] YAMLからデフォルトプロンプトを読み込み: '{default_prompt_from_yaml}'")
    # print(f"[IMPORTANT] YAMLからSEED値を読み込み: {seed_from_yaml}")
    
    # LoRA設定を取得（存在しない場合はNone）
    lora_settings_from_yaml = prompt_data.get("lora_settings", None)
    if lora_settings_from_yaml:
        print(f"[IMPORTANT] YAMLからLoRA設定を読み込み: {lora_settings_from_yaml}")

    # image_filesからファイル名の先頭番号を抽出してマッピング
    image_file_map = {
        int(os.path.basename(img_file)[:3]): img_file for img_file in section_image_files
    }
    
    # セクション番号に対応する画像がない場合はNoneを設定
    for section_num in section_number_list_from_zip:
        if section_num not in image_file_map:
            image_file_map[section_num] = None
            
    # セクション番号順にソートしてリスト化
    section_image_list_from_zip = [image_file_map.get(section_num) for section_num in section_number_list_from_zip]
    
    print(translate("sections.yamlファイルに従ってセクションに設定します。"))

    # 最終結果をデバッグ出力
    # print(f"[DEBUG-FINAL] ZIPファイルからの最終データ:")
    # print(f"  セクション数: {len(section_number_list_from_zip)}")
    # print(f"  プロンプト数: {len(section_prompt_list_from_zip)}")
    # print(f"  画像数: {len(section_image_list_from_zip)}")
    # print(f"  エンドフレーム: {end_frame_image_from_zip}")
    # print(f"  スタートフレーム: {start_frame_image_from_zip}")
    
    # 結果を返す
    result = {
        "section_numbers": section_number_list_from_zip,
        "section_prompts": section_prompt_list_from_zip,
        "section_images": section_image_list_from_zip,
        "end_frame": end_frame_image_from_zip,
        "start_frame": start_frame_image_from_zip,
        "default_prompt": default_prompt_from_yaml,
        "seed": seed_from_yaml,
        "lora_settings": lora_settings_from_yaml
    }
    
    # エンドフレームとスタートフレームの確認
    # if end_frame_image_from_zip:
    #     print(f"[IMPORTANT-FINAL] エンドフレーム画像({end_frame_image_from_zip})が見つかりました")
    # else:
    #     print(f"[WARNING-FINAL] エンドフレーム画像が見つかりませんでした")
        
    # if start_frame_image_from_zip:
    #     print(f"[IMPORTANT-FINAL] スタートフレーム画像({start_frame_image_from_zip})が見つかりました")
    # else:
    #     print(f"[WARNING-FINAL] スタートフレーム画像が見つかりませんでした")
    
    return result

def create_section_zipfile(section_settings, end_frame=None, start_frame=None, additional_info=None):
    """
    セクション情報からzipファイルを作成する

    Args:
        section_settings: セクション設定（各セクションの番号、プロンプト、画像）
        end_frame: 最終フレーム画像のパス（オプション）
        start_frame: 最初のフレーム画像のパス（オプション）
        additional_info: 追加情報（デフォルトプロンプトなど）

    Returns:
        str: 作成されたzipファイルのパス
    """
    # デバッグ出力（詳細表示）
    # print(f"create_section_zipfile: end_frame={end_frame}, start_frame={start_frame}")
    # print(f"start_frame 型 = {type(start_frame)}")
    # if isinstance(start_frame, dict):
    #     print(f"start_frame キー一覧 = {start_frame.keys()}")
    #     if 'path' in start_frame:
    #         print(f"start_frame path = {start_frame['path']}")
    # print(f"受け取ったsection_settings数: {len(section_settings)}")
    
    # 重要: すべてのセクションのセクション番号を検証
    section_nums = [int(row[0]) for row in section_settings if row and len(row) > 0]
    has_section0 = 0 in section_nums
    has_section1 = 1 in section_nums
    # print(f"[INFO] セクション番号一覧: {section_nums}")
    # print(f"[INFO] セクション0が含まれているか: {has_section0}")
    # print(f"[INFO] セクション1が含まれているか: {has_section1}")
    
    # section_settingsから最初の2つを強制的に含める
    first_sections = []
    
    # 有効なセクション設定も初期化（後で使用）
    valid_settings = []
    
    # セクション0と1を確実に含める（存在する場合）
    for section_idx in [0, 1]:
        section_found = False
        
        # 該当するセクション番号のデータを探す
        for row in section_settings:
            if row and len(row) > 0 and int(row[0]) == section_idx:
                print(f"セクション{section_idx}を強制的に含めます: {row}")
                
                # プロンプトの値をそのまま使用（強制設定はしない）
                if len(row) > 2 and (row[2] is None):
                    row[2] = ""  # Noneの場合のみ空文字に変換
                    print(f"  セクション{section_idx}のプロンプトがNoneのため空文字に変換")
                
                first_sections.append(row)
                section_found = True
                break
        
        if not section_found:
            print(f"[WARNING] セクション{section_idx}が見つかりませんでした")
    
    # 一時ディレクトリを作成
    temp_dir = tempfile.mkdtemp(prefix="section_export_")
    print(f"一時ディレクトリを作成: {temp_dir}")
    
    try:
        # セクション情報のYAMLを作成
        sections_yaml = {"section_info": []}
        
        # 追加情報がある場合は追加
        if additional_info:
            # デフォルトプロンプトの追加
            if "default_prompt" in additional_info:
                print(f"[DEBUG-EXPORT] default_prompt値の詳細: '{additional_info['default_prompt']}' (型: {type(additional_info['default_prompt'])}, 長さ: {len(additional_info['default_prompt']) if isinstance(additional_info['default_prompt'], str) else 'N/A'})")
                if additional_info["default_prompt"]:
                    prompt_len = len(additional_info['default_prompt'])
                    print(f"[DEBUG-EXPORT] デフォルトプロンプトを追加: '{additional_info['default_prompt'][:50]}...' (全長: {prompt_len})")
                    sections_yaml["default_prompt"] = additional_info["default_prompt"]
                else:
                    print(f"[DEBUG-EXPORT] default_promptは空文字列またはFalsy値のため追加されません: 値='{additional_info['default_prompt']}'")
            
            # シード値を追加（デフォルトは -1 = ランダム）
            sections_yaml["SEED"] = additional_info.get("seed", -1)
            # print(f"[DEBUG-EXPORT] シード値を追加: {sections_yaml['SEED']}")
            # print(f"[DEBUG-EXPORT] additional_info全体: {additional_info}")
            
            # LoRA設定を追加（存在する場合）
            if "lora_settings" in additional_info and additional_info["lora_settings"]:
                # print(f"LoRA設定を追加: {additional_info['lora_settings']}")
                sections_yaml["lora_settings"] = additional_info["lora_settings"]
        
        # 画像ファイルのコピー先ディレクトリ
        section_images_dir = os.path.join(temp_dir, "sections")
        os.makedirs(section_images_dir, exist_ok=True)
        # 最初の2つのセクションを強制的に含める
        for i, row in enumerate(first_sections):
            # セクション情報をYAMLに追加
            section_num = int(row[0])
            print(f"最初のセクション{i}を処理: {row}")
            
            # プロンプト（強制設定済み）
            prompt_value = ""
            if len(row) > 2 and row[2] is not None:
                prompt_value = row[2]
            
            print(f"  セクション {section_num}: プロンプト「{prompt_value}」を追加（強制）")
            sections_yaml["section_info"].append({
                "section": section_num,
                "prompt": prompt_value
            })
            
            # 画像があればコピー
            if len(row) > 1 and row[1] is not None:
                print(f"  セクション {section_num}: 画像あり - {type(row[1])}")
                # Gradioコンポーネントから実際のパスを取得
                src_image_path = None
                
                # 辞書型データの処理（新しいGradioのFileData形式）
                if isinstance(row[1], dict):
                    if "name" in row[1]:
                        src_image_path = row[1]["name"]
                        print(f"  辞書型(name): {row[1]}, パス={src_image_path}")
                    elif "path" in row[1]:
                        src_image_path = row[1]["path"]
                        print(f"  辞書型(path): {row[1]}, パス={src_image_path}")
                    else:
                        print(f"  辞書型だが有効なパスキーなし: {row[1]}")
                else:
                    src_image_path = row[1]
                    print(f"  値: {src_image_path}")
                
                # パスが有効な場合のみコピー
                if isinstance(src_image_path, str) and os.path.exists(src_image_path):
                    # 3桁のゼロ埋め数字でファイル名を作成
                    dest_filename = f"{section_num:03d}{os.path.splitext(src_image_path)[1]}"
                    dest_path = os.path.join(section_images_dir, dest_filename)
                    # 画像をコピー
                    print(f"  画像コピー: {src_image_path} -> {dest_path}")
                    try:
                        shutil.copy2(src_image_path, dest_path)
                        print(f"  画像コピー成功: セクション{section_num}の画像 ({dest_filename})")
                    except Exception as e:
                        print(f"  [ERROR] 画像コピー失敗: {e}")
                else:
                    print(f"  無効な画像パス: {src_image_path}, 型={type(src_image_path)}, 存在={isinstance(src_image_path, str) and os.path.exists(src_image_path) if isinstance(src_image_path, str) else False}")
                    # 重要: セクション番号が1の場合は特別に警告
                    if section_num == 1:
                        print(f"  [WARNING] セクション1の画像パスが無効です。このセクションの画像が001.pngとして保存できません。")
                    
        # 残りのセクションの処理（既に処理したセクションはスキップ）
        print("=====================================")
        # valid_settingsを直接使わず、section_settingsを直接処理
        
        # 既に処理したセクション番号のセット
        try:
            # first_sectionsから処理済みセクション番号のセットを作成
            processed_sections = set(int(row[0]) for row in first_sections)
            print(f"処理済みセクション番号: {processed_sections}")
        except Exception as e:
            print(f"[ERROR] 処理済みセクション番号作成エラー: {e}")
            processed_sections = set()  # 空のセットで再初期化
        
        # 残りの有効なセクションを処理 - section_settingsから直接処理
        remaining_settings = [row for row in section_settings if len(row) > 0 and int(row[0]) not in processed_sections]
        print(f"処理対象の残りのセクション数: {len(remaining_settings)}")
        
        for i, row in enumerate(remaining_settings):
            # セクション番号、プロンプト、画像を取得
            if not row or len(row) < 1:
                print(f"行 {i}: 無効な行（空またはセクション番号なし）")
                continue
                
            section_num = int(row[0])
            
            # 既に処理済みのセクションはスキップ
            try:
                if section_num in processed_sections:
                    print(f"行 {i}: セクション {section_num} は既に処理済みのためスキップ")
                    continue
            except Exception as e:
                print(f"[ERROR] 処理済みチェックエラー: {e}")
                # エラーが発生した場合は処理を続行
                
            print(f"行 {i}: セクション {section_num} を処理")
            
            # プロンプトがあれば追加（空文字列も保存、画像がなくてもプロンプトだけ保存）
            # 常にすべてのセクションのプロンプトを保存（空でも）
            prompt_value = ""
            if len(row) > 2:
                if row[2] is not None:
                    prompt_value = row[2]
            
            print(f"  セクション {section_num}: プロンプト「{prompt_value}」を追加")
            
            try:
                # 処理済みリストに追加
                processed_sections.add(section_num) 
            except Exception as e:
                print(f"[ERROR] 処理済みリスト追加エラー: {e}")
                processed_sections = set([section_num])
            
            # section_infoにエントリを追加
            sections_yaml["section_info"].append({
                "section": section_num,
                "prompt": prompt_value
            })
            
            # 画像があればコピー
            if len(row) > 1 and row[1] is not None:
                print(f"  セクション {section_num}: 画像あり - {type(row[1])}")
                # Gradioコンポーネントから実際のパスを取得
                src_image_path = None
                
                # 辞書型データの処理（新しいGradioのFileData形式）
                if isinstance(row[1], dict):
                    if "name" in row[1]:
                        src_image_path = row[1]["name"]
                        print(f"  辞書型(name): {row[1]}, パス={src_image_path}")
                    elif "path" in row[1]:
                        src_image_path = row[1]["path"]
                        print(f"  辞書型(path): {row[1]}, パス={src_image_path}")
                    else:
                        print(f"  辞書型だが有効なパスキーなし: {row[1]}")
                else:
                    src_image_path = row[1]
                    print(f"  値: {src_image_path}")
                
                # パスが有効な場合のみコピー
                if isinstance(src_image_path, str) and os.path.exists(src_image_path):
                    # 3桁のゼロ埋め数字でファイル名を作成
                    dest_filename = f"{section_num:03d}{os.path.splitext(src_image_path)[1]}"
                    dest_path = os.path.join(section_images_dir, dest_filename)
                    # 画像をコピー
                    print(f"  画像コピー: {src_image_path} -> {dest_path}")
                    try:
                        shutil.copy2(src_image_path, dest_path)
                        print(f"  画像コピー成功: セクション{section_num}の画像 ({dest_filename})")
                    except Exception as e:
                        print(f"  [ERROR] 画像コピー失敗: {e}")
                else:
                    print(f"  無効な画像パス: {src_image_path}, 型={type(src_image_path)}, 存在={isinstance(src_image_path, str) and os.path.exists(src_image_path) if isinstance(src_image_path, str) else False}")
                    # 重要: セクション番号が1の場合は特別に警告
                    if section_num == 1:
                        print(f"  [WARNING] セクション1の画像パスが無効です。このセクションの画像が001.pngとして保存できません。")
        
        # YAMLの内容を詳細デバッグ出力
        print(f"sections_yaml 詳細内容: {sections_yaml}")
        print(f"section_info の項目数: {len(sections_yaml['section_info'])}")
        
        # section_infoの各項目をログ出力
        print("section_info 各項目の詳細:")
        for i, item in enumerate(sections_yaml['section_info']):
            print(f"  項目{i}: section={item.get('section')}, prompt='{item.get('prompt')}'")
        
        if 'default_prompt' in sections_yaml:
            print(f"default_prompt の長さ: {len(sections_yaml['default_prompt'])}")
            print(f"default_prompt の内容: '{sections_yaml['default_prompt']}'")
        else:
            print("default_promptが設定されていません")
        
        # YAMLファイルを書き込む
        yaml_path = os.path.join(section_images_dir, "sections.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(sections_yaml, f, default_flow_style=False, allow_unicode=True)
            
        # YAMLファイルの内容を直接確認
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_content = f.read()
            print(f"YAMLファイル書き込み内容：\n{yaml_content}")
            print(f"YAMLファイルサイズ: {len(yaml_content)} bytes")
        
        # 最終フレーム画像をコピー - 辞書型の場合も処理できるように拡張
        end_frame_path = None
        if end_frame:
            # 辞書型の場合（Gradioの新しいFileData形式）
            if isinstance(end_frame, dict):
                if "name" in end_frame:
                    end_frame_path = end_frame["name"]
                elif "path" in end_frame:
                    end_frame_path = end_frame["path"]
            else:
                end_frame_path = end_frame
                
            if end_frame_path and isinstance(end_frame_path, str) and os.path.exists(end_frame_path):
                print(f"最終フレーム画像: {end_frame_path}")
                end_ext = os.path.splitext(end_frame_path)[1]
                end_dest = os.path.join(section_images_dir, f"end{end_ext}")
                print(f"最終フレームコピー: {end_frame_path} -> {end_dest}")
                shutil.copy2(end_frame_path, end_dest)
            else:
                print(f"最終フレーム画像パスが無効: {end_frame_path}, 型={type(end_frame_path) if end_frame_path else None}")
        else:
            print(f"最終フレーム画像なし: {end_frame}, 型={type(end_frame)}")
        
        # 開始フレーム画像をコピー - もしセクション0の画像がある場合はそれを流用
        start_frame_path = None
        
        # まず通常のstart_frameの処理
        if start_frame:
            # 辞書型の場合（Gradioの新しいFileData形式）
            if isinstance(start_frame, dict):
                if "name" in start_frame:
                    start_frame_path = start_frame["name"]
                elif "path" in start_frame:
                    start_frame_path = start_frame["path"]
            else:
                start_frame_path = start_frame
        
        # start_frameが無効で、かつセクション0の画像がある場合はそれを代わりに使用
        if not start_frame_path or not isinstance(start_frame_path, str) or not os.path.exists(start_frame_path):
            print("[DEBUG] start_frameが無効なため、セクション0の画像があれば代わりに使用します")
            if len(section_settings) > 0 and len(section_settings[0]) > 1 and section_settings[0][1]:
                section0_image = section_settings[0][1]
                print(f"[DEBUG] セクション0の画像: {section0_image}, 型: {type(section0_image)}")
                
                # セクション0の画像が辞書型の場合
                if isinstance(section0_image, dict):
                    if "path" in section0_image:
                        start_frame_path = section0_image["path"]
                        print(f"[DEBUG] セクション0の画像パスを使用: {start_frame_path}")
                    elif "name" in section0_image:
                        start_frame_path = section0_image["name"]
                        print(f"[DEBUG] セクション0の画像nameを使用: {start_frame_path}")
                else:
                    start_frame_path = section0_image
                    print(f"[DEBUG] セクション0の画像を直接使用: {start_frame_path}")
        
        # パスが有効な場合はコピー処理
        if start_frame_path and isinstance(start_frame_path, str) and os.path.exists(start_frame_path):
            print(f"開始フレーム画像: {start_frame_path}")
            start_ext = os.path.splitext(start_frame_path)[1]
            start_dest = os.path.join(section_images_dir, f"start{start_ext}")
            print(f"開始フレームコピー: {start_frame_path} -> {start_dest}")
            shutil.copy2(start_frame_path, start_dest)
        else:
            print(f"開始フレーム画像パスが無効: {start_frame_path}, 型={type(start_frame_path) if start_frame_path else None}")
        
        # 現在の日時を取得してファイル名に使用
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"section_export_{timestamp}.zip"
        zip_path = os.path.join(os.path.dirname(temp_dir), zip_filename)
        
        # zipファイルを作成
        print(f"ZIPファイルを作成: {zip_path}")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, _, files in os.walk(section_images_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    print(f"ZIPに追加: {file_path} -> {arcname}")
                    zipf.write(file_path, arcname)
            
            # ZIPの内容を確認
            print("ZIPファイルの内容:")
            for info in zipf.infolist():
                print(f" - {info.filename}, size={info.file_size} bytes")
        
        return zip_path
        
    finally:
        # 一時ディレクトリを削除（後始末）
        shutil.rmtree(temp_dir, ignore_errors=True)

def upload_zipfile_handler(file, max_keyframes):
    """
    zipファイルアップロード時のハンドラ関数

    Args:
        file: アップロードされたzipファイル
        max_keyframes: キーフレームの最大数

    Returns:
        list: Gradio用の出力リスト
    """
    # セクション情報を取得
    section_info = process_uploaded_zipfile(file, max_keyframes)
    
    # デバッグ: エンドフレームとスタートフレームを確認
    # print(f"[DEBUG] ZIPからのend_frame: {section_info['end_frame']}")
    # print(f"[DEBUG] ZIPからのstart_frame: {section_info['start_frame']}")
    # print(f"[DEBUG] ZIPからのdefault_prompt: {section_info['default_prompt']}")
    # print(f"[DEBUG] ZIPからのseed: {section_info['seed']}")
    
    # Gradio用の出力リストを生成
    gr_outputs = []
    for i in range(max_keyframes):
        # セクション番号
        gr_outputs.append(section_info["section_numbers"][i])
        
        # セクションプロンプト - gr.updateではなく直接文字列として渡す
        prompt_value = section_info["section_prompts"][i]
        # print(f"[DEBUG] セクション{i}のプロンプト値（生値）: '{prompt_value}' (型: {type(prompt_value)})")
        
        # 重要: gr.update()ではなく純粋な文字列を返す
        # この変更によりプロンプト表示時に({value: ..., __type__: 'update'}, ...)のような表示を防ぐ
        gr_outputs.append(str(prompt_value) if prompt_value is not None else "")
        
        # セクション画像
        gr_outputs.append(section_info["section_images"][i])
    
    # end_frame, start_frameを追加
    # エンドフレームとスタートフレームを明示的にリストに追加
    end_frame_path = section_info["end_frame"]
    start_frame_path = section_info["start_frame"]
    
    # 重要: パスだけでなく、Gradioで使用できる形式に変換
    # 末尾にエンドフレームを追加
    # print(f"[IMPORTANT] end_frame ({end_frame_path}) を出力リストに追加します")
    gr_outputs.append(end_frame_path)
    
    # 末尾にスタートフレームを追加
    # print(f"[IMPORTANT] start_frame ({start_frame_path}) を出力リストに追加します")
    gr_outputs.append(start_frame_path)
    
    # デフォルトプロンプトとシード値も追加
    default_prompt = section_info.get("default_prompt", "")
    seed_value = section_info.get("seed", None)
    # print(f"[IMPORTANT] デフォルトプロンプト ({default_prompt}) とシード値 ({seed_value}) を出力リストに追加します")
    
    # 末尾にデフォルトプロンプトとシード値を追加
    gr_outputs.append(default_prompt)
    gr_outputs.append(seed_value)
    
    # LoRA設定を追加
    lora_settings = section_info.get("lora_settings", None)
    if lora_settings:
        print(f"[IMPORTANT] LoRA設定 ({lora_settings}) を出力リストに追加します")
    gr_outputs.append(lora_settings)
    
    # 出力リストの長さを確認（デバッグ）
    # print(f"[DEBUG] 出力リストの長さ: {len(gr_outputs)}")
    # print(f"[DEBUG] 出力リストの最後の5項目: {gr_outputs[-5:]}")
    
    return gr_outputs

def download_zipfile_handler(section_settings, end_frame, start_frame, additional_info=None):
    """
    zipファイルダウンロード時のハンドラ関数

    Args:
        section_settings: セクション設定（各セクションの番号、プロンプト、画像）
        end_frame: 最終フレーム画像のパス（Gradioのコンポーネント値）
        start_frame: 最初のフレーム画像のパス（Gradioのコンポーネント値）- input_imageに対応
        additional_info: 追加情報（デフォルトプロンプトなど）

    Returns:
        str: 作成されたzipファイルのパス
    """
    # valid_settingsを初期化
    valid_settings = []
    
    # セクション番号のリストを作成して、内容を確認
    section_nums = [int(row[0]) for row in section_settings if row and len(row) > 0]
    has_section0 = 0 in section_nums
    has_section1 = 1 in section_nums
    # print(f"[INFO] セクション番号一覧: {section_nums}")
    # print(f"[INFO] セクション0が含まれているか: {has_section0}")
    # print(f"[INFO] セクション1が含まれているか: {has_section1}")
    
    # 重要: セクション0と1が確実に含まれることを確認
    # セクション番号が0と1のデータを抽出
    section0_data = None
    section1_data = None
    
    for row in section_settings:
        if not row or len(row) < 1:
            continue
            
        section_num = int(row[0])
        if section_num == 0:
            section0_data = row
            # print(f"[INFO] セクション0のデータ: {section0_data}")
        elif section_num == 1:
            section1_data = row
            # print(f"[INFO] セクション1のデータ: {section1_data}")
    
    # セクション0のプロンプトを詳細にデバッグ出力
    # if section_settings and len(section_settings) > 0:
    #     first_section = section_settings[0]
    #     print(f"セクション0の詳細: {first_section}")
    #     if len(first_section) > 2:
    #         print(f"セクション0のプロンプト詳細: '{first_section[2]}', 型: {type(first_section[2])}")
    
    # デバッグ出力
    # print("=== セクション設定一覧 ===")
    # for i, row in enumerate(section_settings):
    #     print(f"セクション {i}: {row}")
    
    # valid_settingsは初期化済み - このコードは通常未使用
    # ここではなく、各セクション設定を直接処理する
    
    # 各セクションを検証して、有効なものだけをvalid_settingsに追加
    for idx, row in enumerate(section_settings):
        if row:
            # 詳細デバッグ出力
            # print(f"評価中のセクション {idx}: {row}")
            is_valid = False
            
            # 最初の2つのセクションは常に含める
            if idx < 2:
                # print(f"  常に含める範囲内 (idx={idx} < 2)")
                is_valid = True
            
            # 画像またはプロンプトがあれば含める
            if len(row) > 1 and row[1] is not None:
                # print(f"  画像あり: {row[1]}")
                is_valid = True
            if len(row) > 2 and row[2] is not None and row[2].strip() != "":
                # print(f"  プロンプトあり: '{row[2]}', 型: {type(row[2])}")
                is_valid = True
            
            # if is_valid:
            #     print(f"  セクションは有効: {row}")
            #     # valid_settingsはdownload_zipfile_handlerでのみ使用
            #     # このCreateSectionZipfileからは使用しない
            # else:
            #     print(f"  セクションは無効: {row}")
    
    # デバッグ出力
    # print("=== 処理対象セクション設定 ===")
    # すべてのセクションを対象にする
    # for i, row in enumerate(section_settings):
    #     if i < 10:  # 多すぎるので最初の10個だけ出力
    #         print(f"セクション {i}: {row}")
    
    # Gradioコンポーネントから実際のパスを取得
    # 最終フレームパスの処理
    end_frame_path = None
    if end_frame:
        if isinstance(end_frame, dict):
            if "name" in end_frame:
                end_frame_path = end_frame.get("name")
            elif "path" in end_frame:
                end_frame_path = end_frame.get("path")
        else:
            end_frame_path = end_frame
    
    # 開始フレームパスの処理    
    start_frame_path = None
    if start_frame:
        if isinstance(start_frame, dict):
            if "name" in start_frame:
                start_frame_path = start_frame.get("name")
            elif "path" in start_frame:
                start_frame_path = start_frame.get("path")
        else:
            start_frame_path = start_frame
    
    # print(f"処理後のパス: end_frame_path={end_frame_path}, start_frame_path={start_frame_path}")
    
    # 追加情報
    # if additional_info:
    #     print(f"追加情報: {additional_info}")
    
    # セクション1の画像データのデバッグ出力
    # if section1_data and len(section1_data) > 1 and section1_data[1] is not None:
    #     print(f"[INFO] セクション1の画像データがあります: {section1_data[1]}")
    # else:
    #     print(f"[WARNING] セクション1の画像データがありません！")
    
    # zipファイルを作成して保存
    # 検証したセクション設定を使用
    # print(f"create_section_zipfileに渡すセクション設定数: {len(section_settings)}")
    zip_path = create_section_zipfile(section_settings, end_frame_path, start_frame_path, additional_info)
    
    return zip_path
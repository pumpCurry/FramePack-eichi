# png_metadata.py
# FramePack-eichiのためのPNGメタデータ処理ユーティリティ

import json
import os
from PIL import Image, PngImagePlugin
import traceback

from locales.i18n_extended import translate

# メタデータキーの定義
PROMPT_KEY = "prompt"
SEED_KEY = "seed"
SECTION_PROMPT_KEY = "section_prompt"
SECTION_NUMBER_KEY = "section_number"
PARAMETERS_KEY = "parameters"  # SD系との互換性のため

def embed_metadata_to_png(image_path, metadata_dict):
    """PNGファイルにメタデータを埋め込む

    Args:
        image_path (str): PNGファイルのパス
        metadata_dict (dict): 埋め込むメタデータの辞書

    Returns:
        str: 処理したファイルのパス
    """
    try:
        # print(translate("[DEBUG] メタデータ埋め込み開始: {0}").format(image_path))
        # print(translate("[DEBUG] 埋め込むメタデータ: {0}").format(metadata_dict))

        img = Image.open(image_path)
        metadata = PngImagePlugin.PngInfo()

        # パラメータを結合したテキストも作成（SD系との互換性のため）
        parameters_text = ""

        # パラメータテキストの構築（個別キーは埋め込まない）
        for key, value in metadata_dict.items():
            if value is not None:
                if key == PROMPT_KEY:
                    parameters_text += f"{value}\n"
                elif key == SEED_KEY:
                    parameters_text += f"Seed: {value}\n"
                elif key == SECTION_PROMPT_KEY:
                    parameters_text += f"Section Prompt: {value}\n"
                elif key == SECTION_NUMBER_KEY:
                    parameters_text += f"Section Number: {value}\n"

        # パラメータテキストがあれば追加
        if parameters_text:
            metadata.add_text(PARAMETERS_KEY, parameters_text.strip())
            # print(translate("[DEBUG] parameters形式でメタデータ追加: {0}").format(parameters_text.strip()))

        # 保存（ファイル形式は変更せず）
        if image_path.lower().endswith('.png'):
            img.save(image_path, "PNG", pnginfo=metadata)
        else:
            # PNGでなければPNGに変換して保存
            png_path = os.path.splitext(image_path)[0] + '.png'
            img.save(png_path, "PNG", pnginfo=metadata)
            image_path = png_path

        # print(translate("[DEBUG] メタデータを埋め込みました: {0}").format(image_path))
        return image_path

    except Exception as e:
        # print(translate("[ERROR] メタデータ埋め込みエラー: {0}").format(e))
        # traceback.print_exc()
        return image_path

def extract_metadata_from_png(image_path_or_object):
    """PNGファイルからメタデータを抽出する

    Args:
        image_path_or_object: PNGファイルのパスまたはPIL.Image.Imageオブジェクト

    Returns:
        dict: 抽出したメタデータの辞書
    """
    try:
        # パスが文字列ならイメージを開く、イメージオブジェクトならそのまま使う
        if isinstance(image_path_or_object, str):
            if not os.path.exists(image_path_or_object):
                # print(translate("[DEBUG] ファイルが存在しません: {0}").format(image_path_or_object))
                return {}
            # print(translate("[DEBUG] 画像ファイルを開いています: {0}").format(image_path_or_object))
            img = Image.open(image_path_or_object)
        else:
            # print(translate("[DEBUG] 与えられた画像オブジェクトを使用します"))
            img = image_path_or_object

        # print(f"[DEBUG] PIL Image info: {img.info}")

        metadata = {}

        # 個別のキーを処理
        for key in [PROMPT_KEY, SEED_KEY, SECTION_PROMPT_KEY, SECTION_NUMBER_KEY, PARAMETERS_KEY]:
            if key in img.info:
                value = img.info[key]
                # print(translate("[DEBUG] メタデータ発見: {0}={1}").format(key, value))
                try:
                    # JSONとして解析を試みる
                    metadata[key] = json.loads(value)
                    # print(translate("[DEBUG] JSONとして解析: {0}={1}").format(key, metadata[key]))
                except (json.JSONDecodeError, TypeError):
                    # JSONでなければそのまま格納
                    metadata[key] = value
                    # print(translate("[DEBUG] 文字列として解析: {0}={1}").format(key, value))

        # parametersキーからの抽出処理（SD系との互換性のため）
        # 個別キーが無い場合でもparametersから抽出を試みる
        if PARAMETERS_KEY in img.info:
            params = img.info[PARAMETERS_KEY]
            # print(translate("[DEBUG] parameters形式のメタデータを処理: {0}").format(params))

            # 行に分割して処理
            lines = params.split("\n")
            
            # プロンプト行を収集
            prompt_lines = []
            
            # 全ての行を処理
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith("Seed:"):
                    seed_str = line_stripped.replace("Seed:", "").strip()
                    if seed_str.isdigit():
                        metadata[SEED_KEY] = int(seed_str)
                elif line_stripped.startswith("Section Number:"):
                    section_num_str = line_stripped.replace("Section Number:", "").strip()
                    if section_num_str.isdigit():
                        metadata[SECTION_NUMBER_KEY] = int(section_num_str)
                elif line_stripped.startswith("Section Prompt:"):
                    section_prompt = line_stripped.replace("Section Prompt:", "").strip()
                    if section_prompt:
                        metadata[SECTION_PROMPT_KEY] = section_prompt
                else:
                    # Seed: や Section: で始まらない行はプロンプトの一部
                    if line.strip():  # 空行は除外
                        prompt_lines.append(line.rstrip())
            
            # 複数行のプロンプトを結合
            if prompt_lines:
                metadata[PROMPT_KEY] = "\n".join(prompt_lines)
                # print(translate("[DEBUG] プロンプト結合: {0}").format(metadata[PROMPT_KEY]))

        # print(translate("[DEBUG] 最終抽出メタデータ: {0}").format(metadata))
        return metadata

    except Exception as e:
        # print(translate("[ERROR] メタデータ抽出エラー: {0}").format(e))
        # traceback.print_exc()
        return {}

def extract_metadata_from_numpy_array(numpy_image):
    """NumPy配列からメタデータを抽出する（PILを介して）

    Args:
        numpy_image: NumPy配列の画像データ

    Returns:
        dict: 抽出したメタデータの辞書
    """
    try:
        if numpy_image is None:
            # print(translate("[DEBUG] extract_metadata_from_numpy_array: 入力がNoneです"))
            return {}

        # print(translate("[DEBUG] numpy_imageのサイズと形状: {0}, データ型: {1}").format(numpy_image.shape, numpy_image.dtype))

        # NumPy配列からPIL.Imageに変換
        img = Image.fromarray(numpy_image)
        # print(translate("[DEBUG] PIL Imageサイズ: {0}, モード: {1}").format(img.size, img.mode))
        # print(f"[DEBUG] PIL Image info: {img.info}")

        # メタデータの抽出を試行
        metadata = extract_metadata_from_png(img)
        # print(translate("[DEBUG] 抽出されたメタデータ: {0}").format(metadata))

        return metadata

    except Exception as e:
        # print(translate("[ERROR] NumPy配列からのメタデータ抽出エラー: {0}").format(e))
        # traceback.print_exc()
        return {}

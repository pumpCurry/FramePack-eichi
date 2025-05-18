import os
import sys
import argparse
import torch
import traceback
import safetensors.torch as sf
from datetime import datetime
import gradio as gr

# ルートパスをシステムパスに追加
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

# ルートパスを追加した後でインポート
from locales.i18n_extended import translate

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
        print(translate("ファイル1を読み込み中: {0}").format(os.path.basename(file1_path)))
        tensor_dict1 = sf.load_file(file1_path)

        # ファイル2を読み込み
        print(translate("ファイル2を読み込み中: {0}").format(os.path.basename(file2_path)))
        tensor_dict2 = sf.load_file(file2_path)

        # テンソルを取得
        if "history_latents" in tensor_dict1 and "history_latents" in tensor_dict2:
            tensor1 = tensor_dict1["history_latents"]
            tensor2 = tensor_dict2["history_latents"]

            # テンソル情報の表示
            print(translate("テンソル1: shape={0}, dtype={1}, フレーム数={2}").format(tensor1.shape, tensor1.dtype, tensor1.shape[2]))
            print(translate("テンソル2: shape={0}, dtype={1}, フレーム数={2}").format(tensor2.shape, tensor2.dtype, tensor2.shape[2]))

            # サイズチェック
            if tensor1.shape[3] != tensor2.shape[3] or tensor1.shape[4] != tensor2.shape[4]:
                error_msg = translate("エラー: テンソルサイズが異なります: {0} vs {1}").format(tensor1.shape, tensor2.shape)
                print(error_msg)
                return False, None, error_msg

            # データ型とデバイスの調整
            if tensor1.dtype != tensor2.dtype:
                print(translate("データ型の変換: {0} → {1}").format(tensor2.dtype, tensor1.dtype))
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
            print(translate("結合成功: 結合後のフレーム数={0} ({1}+{2}フレーム)").format(combined_frames, tensor1_frames, tensor2_frames))

            # メタデータを更新
            height, width = tensor1.shape[3], tensor1.shape[4]
            metadata = torch.tensor([height, width, combined_frames], dtype=torch.int32)

            # 出力ファイルパスが指定されていない場合は自動生成
            if output_path is None:
                timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
                output_dir = os.path.dirname(file1_path)
                output_path = os.path.join(output_dir, f"combined_{timestamp}.safetensors")

            # 結合したテンソルをファイルに保存
            tensor_dict = {
                "history_latents": combined_tensor,
                "metadata": metadata
            }

            # ファイル保存
            sf.save_file(tensor_dict, output_path)

            # テンソルデータの保存サイズの概算
            tensor_size_mb = (combined_tensor.element_size() * combined_tensor.nelement()) / (1024 * 1024)

            success_msg = translate("結合テンソルを保存しました: {0}\n").format(os.path.basename(output_path))
            success_msg += translate("フレーム数: {0}フレーム ({1}+{2}フレーム)\n").format(combined_frames, tensor1_frames, tensor2_frames)
            success_msg += translate("サイズ: {0:.2f}MB, 形状: {1}").format(tensor_size_mb, combined_tensor.shape)
            print(success_msg)

            return True, output_path, success_msg
        else:
            error_msg = translate("エラー: テンソルファイルに必要なキー'history_latents'がありません")
            print(error_msg)
            return False, None, error_msg

    except Exception as e:
        error_msg = translate("テンソル結合中にエラーが発生: {0}").format(e)
        print(error_msg)
        traceback.print_exc()
        return False, None, error_msg

def create_ui():
    """Gradio UIを作成"""
    with gr.Blocks(title=translate("テンソル結合ツール")) as app:
        gr.Markdown(translate("## テンソルデータ結合ツール"))
        gr.Markdown(translate("safetensors形式のテンソルデータファイルを2つ選択して結合します。結合順序は「テンソル1 + テンソル2」です。"))

        with gr.Row():
            with gr.Column(scale=1):
                tensor_file1 = gr.File(label=translate("テンソルファイル1 (.safetensors)"), file_types=[".safetensors"])
            with gr.Column(scale=1):
                tensor_file2 = gr.File(label=translate("テンソルファイル2 (.safetensors)"), file_types=[".safetensors"])

        with gr.Row():
            output_file = gr.Textbox(label=translate("出力ファイル名 (空欄で自動生成)"), placeholder=translate("例: combined.safetensors"))

        with gr.Row():
            combine_btn = gr.Button(translate("テンソルファイルを結合"), variant="primary")

        with gr.Row():
            result_output = gr.Textbox(label=translate("結果"), lines=5)

        def combine_tensors(file1, file2, output_path):
            if file1 is None or file2 is None:
                return translate("エラー: 2つのテンソルファイルを選択してください")

            file1_path = file1.name
            file2_path = file2.name

            # 出力パスの決定
            if output_path and output_path.strip():
                # 拡張子のチェックと追加
                if not output_path.lower().endswith('.safetensors'):
                    output_path += '.safetensors'
                # ディレクトリパスの決定（入力ファイルと同じ場所）
                output_dir = os.path.dirname(file1_path)
                full_output_path = os.path.join(output_dir, output_path)
            else:
                # 自動生成の場合はNoneのまま（関数内で自動生成）
                full_output_path = None

            success, result_path, message = combine_tensor_files(file1_path, file2_path, full_output_path)
            if success:
                return message
            else:
                return translate("結合失敗: {0}").format(message)

        combine_btn.click(
            fn=combine_tensors,
            inputs=[tensor_file1, tensor_file2, output_file],
            outputs=[result_output]
        )

    return app

def main():
    """コマンドライン引数を解析して実行"""
    parser = argparse.ArgumentParser(description=translate("2つのsafetensorsファイルを結合するツール"))
    parser.add_argument('--file1', type=str, help=translate("1つ目のsafetensorsファイルパス"))
    parser.add_argument('--file2', type=str, help=translate("2つ目のsafetensorsファイルパス"))
    parser.add_argument('--output', type=str, default=None, help=translate("出力ファイルパス (省略可能)"))
    parser.add_argument('--ui', action='store_true', help=translate("GradioのUIモードで起動"))

    args = parser.parse_args()

    if args.ui:
        # UIモードで起動
        app = create_ui()
        app.launch()
    elif args.file1 and args.file2:
        # コマンドラインモードで実行
        success, output_path, message = combine_tensor_files(args.file1, args.file2, args.output)
        if success:
            print(translate("結合成功:"))
            print(message)
            return 0
        else:
            print(translate("結合失敗:"))
            print(message)
            return 1
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
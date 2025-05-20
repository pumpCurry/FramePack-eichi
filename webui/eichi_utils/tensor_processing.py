import torch
import os
from locales.i18n import translate
from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from PIL import Image
import numpy as np
from diffusers_helper.hunyuan import vae_decode
from eichi_utils.vae_cache import vae_decode_cache
from diffusers_helper.utils import save_bcthw_as_mp4


def print_tensor_info(tensor: torch.Tensor, name: str = "テンソル") -> None:
    """テンソルの詳細情報を出力する

    Args:
        tensor (torch.Tensor): 分析対象のテンソル
        name (str, optional): テンソルの名前. デフォルトは"テンソル"

    Returns:
        None: 標準出力に情報を表示
    """
    try:
        print(translate("[DEBUG] {0}の詳細分析:").format(name))
        print(translate("  - 形状: {0}").format(tensor.shape))
        print(translate("  - 型: {0}").format(tensor.dtype))
        print(translate("  - デバイス: {0}").format(tensor.device))
        print(
            translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(
                tensor.min().item(),
                tensor.max().item(),
                tensor.mean().item(),
            )
        )
        if torch.cuda.is_available():
            print(
                translate("  - 使用GPUメモリ: {0:.2f}GB/{1:.2f}GB").format(
                    torch.cuda.memory_allocated() / 1024**3,
                    torch.cuda.get_device_properties(0).total_memory / 1024**3,
                )
            )
    except Exception as e:
        print(translate("[警告] テンソル情報の出力に失敗: {0}").format(str(e)))


def ensure_tensor_properties(
    tensor: torch.Tensor,
    target_device: torch.device,
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """テンソルのデバイスと型を確認・調整する

    Args:
        tensor (torch.Tensor): 調整対象のテンソル
        target_device (torch.device): 目標のデバイス
        target_dtype (torch.dtype, optional): 目標のデータ型. デフォルトはfloat16

    Returns:
        torch.Tensor: デバイスと型が調整されたテンソル

    Raises:
        RuntimeError: テンソルの調整に失敗した場合
    """
    try:
        if tensor.device != target_device:
            tensor = tensor.to(target_device)
        if tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)
        return tensor
    except Exception as e:
        raise RuntimeError(
            translate("テンソルプロパティの調整に失敗: {0}").format(str(e))
        )


def process_tensor_chunk(
    chunk_idx: int,
    current_chunk: torch.Tensor,
    num_chunks: int,
    chunk_start: int,
    chunk_end: int,
    frames: int,
    use_vae_cache: bool,
    vae: torch.nn.Module,
    stream: any,
    reverse: bool,
) -> torch.Tensor:
    """個別のテンソルチャンクを処理する

    Args:
        chunk_idx (int): 現在のチャンクのインデックス
        current_chunk (torch.Tensor): 処理対象のチャンク
        num_chunks (int): 総チャンク数
        chunk_start (int): チャンクの開始フレーム位置
        chunk_end (int): チャンクの終了フレーム位置
        frames (int): 総フレーム数
        use_vae_cache (bool): VAEキャッシュを使用するかどうか
        vae (torch.nn.Module): VAEモデル
        stream: 進捗表示用のストリームオブジェクト
        reverse (bool): フレーム順序を反転するかどうか

    Returns:
        torch.Tensor: 処理済みのピクセルテンソル

    Raises:
        RuntimeError: チャンク処理中にエラーが発生した場合
    """
    try:
        chunk_frames = chunk_end - chunk_start

        # チャンクサイズの検証
        if chunk_frames <= 0:
            raise ValueError(
                f"不正なチャンクサイズ: {chunk_frames} (start={chunk_start}, end={chunk_end})"
            )

        if current_chunk.shape[2] <= 0:
            raise ValueError(f"不正なテンソル形状: {current_chunk.shape}")

        # 進捗状況を更新
        chunk_progress = (chunk_idx + 1) / num_chunks * 100
        progress_message = translate(
            "テンソルデータ結合中: チャンク {0}/{1} (フレーム {2}-{3}/{4})"
        ).format(
            chunk_idx + 1,
            num_chunks,
            chunk_start,
            chunk_end,
            frames,
        )
        stream.output_queue.push(
            (
                "progress",
                (
                    None,
                    progress_message,
                    make_progress_bar_html(
                        int(80 + chunk_progress * 0.1),
                        translate("テンソルデータ処理中"),
                    ),
                ),
            )
        )

        print(
            translate("チャンク{0}/{1}処理中: フレーム {2}-{3}/{4}").format(
                chunk_idx + 1,
                num_chunks,
                chunk_start,
                chunk_end,
                frames,
            )
        )

        # メモリ状態を出力
        if torch.cuda.is_available():
            print(
                translate(
                    "[MEMORY] チャンク{0}処理前のGPUメモリ: {1:.2f}GB/{2:.2f}GB"
                ).format(
                    chunk_idx + 1,
                    torch.cuda.memory_allocated() / 1024**3,
                    torch.cuda.get_device_properties(0).total_memory / 1024**3,
                )
            )
            # メモリキャッシュをクリア
            torch.cuda.empty_cache()

        # 各チャンク処理前にGPUメモリを解放
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            import gc

            gc.collect()

        # VAEデコード処理
        print(translate("[INFO] VAEデコード開始: チャンク{0}").format(chunk_idx + 1))
        stream.output_queue.push(
            (
                "progress",
                (
                    None,
                    translate("チャンク{0}/{1}のVAEデコード中...").format(
                        chunk_idx + 1, num_chunks
                    ),
                    make_progress_bar_html(
                        int(80 + chunk_progress * 0.1),
                        translate("デコード処理"),
                    ),
                ),
            )
        )

        print_tensor_info(current_chunk, "チャンク{0}".format(chunk_idx + 1))

        # 明示的にデバイスを合わせる
        current_chunk = ensure_tensor_properties(current_chunk, vae.device)

        chunk_pixels = process_latents(
            current_chunk,
            vae,
            use_vae_cache,
            translate("チャンク"),
        )
        print(
            translate(
                "チャンク{0}のVAEデコード完了 (フレーム数: {1}, デコード後フレーム: {2})"
            ).format(chunk_idx + 1, chunk_frames, chunk_pixels.shape)
        )

        if reverse:
            chunk_pixels = reorder_tensor(chunk_pixels)
        return chunk_pixels

    except Exception as e:
        error_msg = translate("チャンク{0}の処理中にエラーが発生: {1}").format(
            chunk_idx + 1, str(e)
        )
        print(f"[エラー] {error_msg}")
        raise RuntimeError(error_msg)


def process_tensor_chunks(
    tensor: torch.Tensor,
    frames: int,
    use_vae_cache: bool,
    job_id: str,
    outputs_folder: str,
    mp4_crf: int,
    stream: any,
    vae: torch.nn.Module,
    reverse: bool = False,
    skip_save: bool = True,
) -> tuple[torch.Tensor, int]:
    """テンソルデータをチャンクに分割して処理する

    Args:
        tensor (torch.Tensor): 処理対象のテンソル
        frames (int): フレーム数
        use_vae_cache (bool): VAEキャッシュを使用するかどうか
        job_id (str): ジョブID
        outputs_folder (str): 出力フォルダパス
        mp4_crf (int): MP4のCRF値
        stream: 進捗表示用のストリームオブジェクト
        vae (torch.nn.Module): VAEモデル
        reverse (bool, optional): フレーム順序を反転するかどうか. デフォルトはFalse
        skip_save (bool, optional): 中間結果の保存をスキップするかどうか. デフォルトはTrue

    Returns:
        tuple[torch.Tensor, int]: (結合されたピクセルテンソル, 処理したチャンク数)

    Raises:
        RuntimeError: テンソル処理中にエラーが発生した場合
    """
    try:
        if frames <= 0:
            raise ValueError(f"不正なフレーム数: {frames}")

        combined_pixels = None
        # チャンクサイズは5以上、フレーム数以下に制限
        chunk_size = min(5, frames)
        num_chunks = (frames + chunk_size - 1) // chunk_size

        # テンソルデータの詳細を出力
        print(f"[DEBUG] フレーム総数: {frames}")
        print(f"[DEBUG] チャンクサイズ: {chunk_size}")
        print(f"[DEBUG] チャンク数: {num_chunks}")
        print(f"[DEBUG] 入力テンソル形状: {tensor.shape}")
        print(f"[DEBUG] VAEキャッシュ: {use_vae_cache}")
        print(f"[DEBUG] ジョブID: {job_id}")
        print_tensor_info(tensor, "入力テンソル")

        # テンソルの形状を確認
        if tensor.shape[2] != frames:
            raise ValueError(
                f"テンソル形状不一致: テンソルのフレーム数 {tensor.shape[2]} != 指定フレーム数 {frames}"
            )

        # チャンク処理
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, frames)

            # チャンクサイズの確認
            if chunk_end <= chunk_start:
                print(
                    f"[警告] 不正なチャンク範囲をスキップ: start={chunk_start}, end={chunk_end}"
                )
                continue

            try:
                # 現在のチャンクを取得
                current_chunk = tensor[:, :, chunk_start:chunk_end, :, :]
                chunk_pixels = process_tensor_chunk(
                    chunk_idx,
                    current_chunk,
                    num_chunks,
                    chunk_start,
                    chunk_end,
                    frames,
                    use_vae_cache,
                    vae,
                    stream,
                    reverse,
                )

                # 結果の結合
                if combined_pixels is None:
                    combined_pixels = chunk_pixels
                else:
                    # 両方とも必ずCPUに移動してから結合
                    current_chunk = ensure_tensor_properties(
                        current_chunk, torch.device("cpu")
                    )
                    combined_pixels = ensure_tensor_properties(
                        combined_pixels, torch.device("cpu")
                    )
                    # 結合処理
                    combined_pixels = torch.cat(
                        [combined_pixels, chunk_pixels],
                        dim=2,
                    )

                # 結合後のフレーム数を確認
                current_total_frames = combined_pixels.shape[2]
                print(
                    translate(
                        "チャンク{0}の結合完了: 現在の組み込みフレーム数 = {1}"
                    ).format(chunk_idx + 1, current_total_frames)
                )

                # 中間結果の保存（チャンクごとに保存すると効率が悪いので、最終チャンクのみ保存）
                if chunk_idx == num_chunks - 1 or (
                    chunk_idx > 0 and (chunk_idx + 1) % 5 == 0
                ):
                    # 5チャンクごと、または最後のチャンクで保存
                    interim_output_filename = os.path.join(
                        outputs_folder,
                        f"{job_id}_combined_interim_{chunk_idx + 1}.mp4",
                    )
                    print(
                        translate("中間結果を保存中: チャンク{0}/{1}").format(
                            chunk_idx + 1, num_chunks
                        )
                    )

                    # （中間）動画を保存するかどうか
                    if not skip_save:
                        chunk_progress = (chunk_idx + 1) / num_chunks * 100
                        stream.output_queue.push(
                            (
                                "progress",
                                (
                                    None,
                                    translate(
                                        "中間結果のMP4変換中... (チャンク{0}/{1})"
                                    ).format(chunk_idx + 1, num_chunks),
                                    make_progress_bar_html(
                                        int(85 + chunk_progress * 0.1),
                                        translate("MP4保存中"),
                                    ),
                                ),
                            )
                        )

                        # MP4として保存
                        save_bcthw_as_mp4(
                            combined_pixels,
                            interim_output_filename,
                            fps=30,
                            crf=mp4_crf,
                        )
                        print(
                            translate("中間結果を保存しました: {0}").format(
                                interim_output_filename
                            )
                        )

                        # 結合した動画をUIに反映するため、出力フラグを立てる
                        stream.output_queue.push(("file", interim_output_filename))

                # メモリ解放
                del current_chunk
                del chunk_pixels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                error_msg = translate("チャンク{0}の処理中にエラー: {1}").format(
                    chunk_idx + 1, str(e)
                )
                print(f"[エラー] {error_msg}")
                raise

        return combined_pixels, num_chunks

    except Exception as e:
        error_msg = translate("テンソル処理中に重大なエラーが発生: {0}").format(str(e))
        print(f"[重大エラー] {error_msg}")
        raise RuntimeError(error_msg)


def output_latent_to_image(
    latent: torch.Tensor,
    file_path: str,
    vae: torch.nn.Module,
    use_vae_cache: bool = False,
) -> None:
    """VAEを使用してlatentを画像化してファイル出力する

    Args:
        latent (torch.Tensor): 変換対象のlatentテンソル
        file_path (str): 出力ファイルパス
        vae (torch.nn.Module): VAEモデル
        use_vae_cache (bool, optional): VAEキャッシュを使用するかどうか. デフォルトはFalse

    Raises:
        Exception: latentの次元が5でない場合
    """
    if latent.dim() != 5:
        raise Exception(f"Error: latent dimension must be 5, got {latent.dim()}")

    image_pixels = process_latents(latent, vae, use_vae_cache)
    image_pixels = (
        (image_pixels[0, :, 0] * 127.5 + 127.5)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )
    Image.fromarray(image_pixels).save(file_path)


def fix_tensor_size(
    tensor: torch.Tensor,
    target_size: int = 1 + 2 + 16,
    fix_edge=True,
) -> torch.Tensor:
    """tensorのフレーム情報（[batch_size, chanel, frames, height, width]のframes）が小さい場合に補間する
    latentの補間は動画にしてもきれいにはならないので注意

    Args:
        latent (torch.Tensor): 補間対象のテンソル
        target_size (int, optional): 目標フレームサイズ. デフォルトは19(1+2+16)
        fix_edge (bool): 入力されたテンソルの開始、終了フレームを固定する

    Returns:
        torch.Tensor: サイズ調整されたlatentテンソル

    Raises:
        Exception: latentの次元が5でない場合
    """
    if tensor.dim() != 5:
        raise Exception(f"Programing Error: latent dim != 5. {tensor.shape}")
    if tensor.shape[2] < target_size:
        print(
            f"[WARN] latentサイズが足りないので補間します。{tensor.shape[2]}=>{target_size}"
        )

        if fix_edge:
            first_frame = tensor[:, :, :1, :, :]
            last_frame = tensor[:, :, -1:, :, :]
            middle_frames = tensor[:, :, 1:-1, :, :]

            # 足りない部分を補間（再生の補間ではなく4の倍数にするための補間）
            # これをこのまま再生しても綺麗ではない
            fixed_tensor = torch.nn.functional.interpolate(
                middle_frames,
                size=(
                    target_size - 2,
                    tensor.shape[3],
                    tensor.shape[4],
                ),
                mode="nearest",
            )
            # 結合処理
            fixed_tensor = torch.cat(
                [first_frame, fixed_tensor, last_frame],
                dim=2,
            )
        else:
            fixed_tensor = torch.nn.functional.interpolate(
                tensor,
                size=(
                    target_size - 2,
                    tensor.shape[3],
                    tensor.shape[4],
                ),
                mode="nearest",
            )
    else:
        fixed_tensor = tensor
    return fixed_tensor


def process_latents(
    latents: torch.Tensor,
    vae: torch.nn.Module,
    use_vae_cache: bool = False,
    debug_str: str = "",
) -> torch.Tensor:
    """VAEデコード：latentをpixels化する

    Args:
        latents (torch.Tensor): デコード対象のlatentテンソル
        vae (torch.nn.Module): VAEモデル
        use_vae_cache (bool, optional): VAEキャッシュを使用するかどうか. デフォルトはFalse
        debug_str (str, optional): デバッグ用の文字列. デフォルトは空文字

    Returns:
        torch.Tensor: デコードされたピクセルテンソル

    Raises:
        Exception: latentの次元が5でない場合
    """
    if latents.dim() != 5:
        raise Exception(f"Programing Error: latent dim != 5. {latents.shape}")

    print(f"[DEBUG] VAEデコード前のlatentsの形状: {latents.shape}")
    print(f"[DEBUG] VAEデコード前のlatentsのデバイス: {latents.device}")
    print(f"[DEBUG] VAEデコード前のlatentsのデータ型: {latents.dtype}")

    if use_vae_cache:
        print(f"[INFO] VAEキャッシュを使用 {debug_str}")
        pixels = vae_decode_cache(latents, vae).cpu()
    else:
        print(f"[INFO] 通常デコード使用 {debug_str}")
        # デバイスとデータ型を明示的に合わせる
        latents = latents.to(device=vae.device, dtype=vae.dtype)
        pixels = vae_decode(latents, vae).cpu()
    return pixels


def reorder_tensor(tensor: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    """テンソルのフレーム順序を操作する

    Args:
        tensor (torch.Tensor): 操作対象のテンソル
        reverse (bool, optional): 順序を反転するかどうか. デフォルトはFalse

    Returns:
        torch.Tensor: フレーム順序が調整されたテンソル
    """
    if reverse:
        return tensor.flip(dims=[2])
    return tensor

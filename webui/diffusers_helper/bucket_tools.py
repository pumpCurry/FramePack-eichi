# 安全な解像度値のリスト（64刻みで384～3840）
SAFE_RESOLUTIONS = list(range(384, 3841, 64))


def _generate_bucket_options(resolution: int, step: int = 64):
    """指定解像度向けのバケット候補を生成する"""
    min_val = int((resolution * 2 / 3) // step * step)
    max_val = int((resolution * 4 / 3) // step * step)
    buckets = []
    for h in range(min_val, max_val + step, step):
        w = int(round(resolution * resolution / h / step) * step)
        buckets.append((h, w))
    return buckets


bucket_options = {res: _generate_bucket_options(res) for res in SAFE_RESOLUTIONS}


def find_nearest_bucket(h, w, resolution=640):
    """最も適切なアスペクト比のバケットを見つける関数"""
    # 安全な解像度に丸める
    if resolution not in SAFE_RESOLUTIONS:
        # 最も近い安全な解像度を選択
        closest_resolution = min(SAFE_RESOLUTIONS, key=lambda x: abs(x - resolution))
        print(f"Warning: Resolution {resolution} is not in safe list. Using {closest_resolution} instead.")
        resolution = closest_resolution
    
    min_metric = float('inf')
    best_bucket = None
    for (bucket_h, bucket_w) in bucket_options[resolution]:
        # アスペクト比の差を計算
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    
    return best_bucket


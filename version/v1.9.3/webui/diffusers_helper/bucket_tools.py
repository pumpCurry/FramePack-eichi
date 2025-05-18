# 安全な解像度値のリスト
SAFE_RESOLUTIONS = [512, 640, 768, 960, 1080]

# 標準解像度のバケット設定
bucket_options = {
    512: [
        (352, 704),
        (384, 640),
        (416, 608),
        (448, 576),
        (480, 544),
        (512, 512),
        (544, 480),
        (576, 448),
        (608, 416),
        (640, 384),
        (704, 352),
    ],
    640: [
        (416, 960),
        (448, 864),
        (480, 832),
        (512, 768),
        (544, 704),
        (576, 672),
        (608, 640),
        (640, 608),
        (672, 576),
        (704, 544),
        (768, 512),
        (832, 480),
        (864, 448),
        (960, 416),
    ],
    768: [
        (512, 1024),
        (576, 960),
        (640, 896),
        (704, 832),
        (768, 768),
        (832, 704),
        (896, 640),
        (960, 576),
        (1024, 512),
    ],
    960: [
        (640, 1280),
        (704, 1152),
        (768, 1024),
        (832, 960),
        (896, 896),
        (960, 832),
        (1024, 768),
        (1152, 704),
        (1280, 640),
    ],
    1080: [
        (720, 1440),
        (768, 1344),
        (832, 1248),
        (896, 1152),
        (960, 1080),
        (1024, 1024),
        (1080, 960),
        (1152, 896),
        (1248, 832),
        (1344, 768),
        (1440, 720),
    ],
}


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


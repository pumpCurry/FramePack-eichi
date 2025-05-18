import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import snapshot_download

class ModelDownloader:
    def __init__(self, max_workers_per_model=4):    
        # max_parallel_models > 1 は tqdm._lock が競合し異常終了するため、当面は1に固定する
        self.max_parallel_models = 1
        self.max_workers_per_model = max_workers_per_model

    def _download_models(self, models_to_download):
        def download_model(model_info):
            kwargs = {
                "repo_id": model_info["repo_id"],
                "allow_patterns": model_info.get("allow_patterns", "*"),
                "max_workers": self.max_workers_per_model,
            }
            snapshot_download(**kwargs)

        with ThreadPoolExecutor(max_workers=self.max_parallel_models) as executor:
            futures = [executor.submit(download_model, model) for model in models_to_download]
            for future in as_completed(futures):
                future.result()
    
    def download_original(self):
        self._download_models([
            {"repo_id": "hunyuanvideo-community/HunyuanVideo", "allow_patterns": ["tokenizer/*", "tokenizer_2/*", "vae/*", "text_encoder/*", "text_encoder_2/*"]},
            {"repo_id": "lllyasviel/flux_redux_bfl", "allow_patterns": ["feature_extractor/*", "image_encoder/*"]},
            {"repo_id": "lllyasviel/FramePackI2V_HY"},
        ])

    def download_f1(self):
        self._download_models([
            {"repo_id": "hunyuanvideo-community/HunyuanVideo", "allow_patterns": ["tokenizer/*", "tokenizer_2/*", "vae/*", "text_encoder/*", "text_encoder_2/*"]},
            {"repo_id": "lllyasviel/flux_redux_bfl", "allow_patterns": ["feature_extractor/*", "image_encoder/*"]},
            {"repo_id": "lllyasviel/FramePack_F1_I2V_HY_20250503"},
        ])

"""
VAE Cache Utility for FramePack-eichi

1フレームずつVAEデコードを行うためのキャッシュ機能を提供するモジュール。
Hunyuan VideoのVAEに対して、フレームごとに処理しながらキャッシュを活用することで
メモリ使用効率と処理速度を改善します。
"""

import torch
import torch.nn.functional as F
from typing import Optional
import time

def hook_forward_conv3d(self):
    """HunyuanVideoCausalConv3dのforwardをフック置換する関数"""
    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        if self.time_causal_padding[4] > 0:
            if hasattr(self, "cache") and self.cache is not None:
                hidden_states[:, :, :self.time_causal_padding[4]] = self.cache.clone() # 先頭フレームにキャッシュをコピー
            self.cache = hidden_states[:, :, -self.time_causal_padding[4]:].clone() # 末尾フレームをキャッシュ
        return self.conv(hidden_states)
    return forward

def hook_forward_upsample(self):
    """HunyuanVideoUpsampleCausal3Dのforwardをフック置換する関数"""
    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(self.conv, "cache") and self.conv.cache is not None:
            # キャッシュを使用している場合は全フレームをアップサンプリング
            hidden_states = F.interpolate(hidden_states.contiguous(), scale_factor=self.upsample_factor, mode="nearest")
        else:
            num_frames = hidden_states.size(2)

            first_frame, other_frames = hidden_states.split((1, num_frames - 1), dim=2)
            first_frame = F.interpolate(
                first_frame.squeeze(2), scale_factor=self.upsample_factor[1:], mode="nearest"
            ).unsqueeze(2)

            if num_frames > 1:
                other_frames = other_frames.contiguous()
                other_frames = F.interpolate(other_frames, scale_factor=self.upsample_factor, mode="nearest")
                hidden_states = torch.cat((first_frame, other_frames), dim=2)
            else:
                hidden_states = first_frame

        hidden_states = self.conv(hidden_states)
        return hidden_states
    return forward

# Attention用のKVキャッシュプロセッサ
class AttnProcessor2_0_KVCache:
    """KVキャッシュを使用するAttentionプロセッサ"""
    
    def __init__(self):
        self.k_cache = None
        self.v_cache = None
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # KVキャッシュの統合
        if self.k_cache is not None:
            key = torch.cat([self.k_cache, key], dim=2)
            value = torch.cat([self.v_cache, value], dim=2)
            attention_mask = torch.cat(
                [torch.zeros(attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[2], self.k_cache.shape[2]).to(attention_mask), attention_mask], dim=3
            )
        
        # 現在のKVをキャッシュとして保存
        self.k_cache = key.clone()
        self.v_cache = value.clone()

        # Scaled Dot-Product Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # 線形変換
        hidden_states = attn.to_out[0](hidden_states)
        # ドロップアウト
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def hook_vae(vae):
    """VAEをキャッシュモードに変更"""
    # 元の設定を保存
    vae._original_use_framewise_decoding = vae.use_framewise_decoding
    vae._original_use_slicing = vae.use_slicing
    vae._original_use_tiling = vae.use_tiling
    
    # キャッシュモード用の設定に変更
    vae.use_framewise_decoding = False
    vae.use_slicing = False
    vae.use_tiling = False
    
    # 各モジュールをフック
    for module in vae.decoder.modules():
        if module.__class__.__name__ == "HunyuanVideoCausalConv3d":
            module._orginal_forward = module.forward
            module.forward = hook_forward_conv3d(module)
        if module.__class__.__name__ == "HunyuanVideoUpsampleCausal3D":
            module._orginal_forward = module.forward
            module.forward = hook_forward_upsample(module)
        if module.__class__.__name__ == "Attention":
            module._orginal_processor = module.processor
            module.processor = AttnProcessor2_0_KVCache()

def restore_vae(vae):
    """VAEを元の状態に戻す"""
    # 設定を元に戻す
    vae.use_framewise_decoding = vae._original_use_framewise_decoding
    vae.use_slicing = vae._original_use_slicing
    vae.use_tiling = vae._original_use_tiling

    # キャッシュをクリアして元の実装に戻す
    for module in vae.decoder.modules():
        if module.__class__.__name__ == "HunyuanVideoCausalConv3d":
            module.forward = module._orginal_forward
            if hasattr(module, "cache"):
                module.cache = None
        if module.__class__.__name__ == "HunyuanVideoUpsampleCausal3D":
            module.forward = module._orginal_forward
            if hasattr(module.conv, "cache"):
                module.conv.cache = None
        if module.__class__.__name__ == "Attention":
            if hasattr(module.processor, "k_cache"):
                module.processor.k_cache = None
                module.processor.v_cache = None
            module.processor = module._orginal_processor

@torch.no_grad()
def vae_decode_cache(latents, vae):
    """1フレームずつVAEデコードを行う関数"""
    # デバッグログを追加
    print("=== VAEキャッシュデコード開始 ===")
    print(f"入力latents形状: {latents.shape}, デバイス: {latents.device}, 型: {latents.dtype}")
    
    # スケーリング係数の適用
    latents = latents / vae.config.scaling_factor
    frames = latents.shape[2]
    print(f"処理フレーム数: {frames}")
    
    # VAEにフックを適用
    print("VAEにフックを適用...")
    hook_vae(vae)
    print("フック適用完了")
    
    # 1フレームずつ処理
    image = None
    try:
        for i in range(frames):
            print(f"フレーム {i+1}/{frames} 処理中...")
            latents_slice = latents[:, :, i:i+1, :, :]
            # デコード処理（内部でキャッシュを活用）
            image_slice = vae.decode(latents_slice.to(device=vae.device, dtype=vae.dtype)).sample
            print(f"フレーム {i+1} デコード完了: 形状 {image_slice.shape}")
            
            # 結果の結合
            if image is None:
                image = image_slice
            else:
                image = torch.cat((image, image_slice), dim=2)
            print(f"現在の結合結果形状: {image.shape}")
    except Exception as e:
        print(f"VAEキャッシュデコード中のエラー: {e}")
        print(f"エラー詳細: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        # エラーが発生した場合、VAEを元の状態に戻してから例外を再スロー
        restore_vae(vae)
        raise e
    
    # VAEを元の状態に戻す
    print("VAEを元の状態に戻しています...")
    restore_vae(vae)
    print("VAEを元の状態に戻しました")
    
    print(f"出力image形状: {image.shape}, デバイス: {image.device}, 型: {image.dtype}")
    print("=== VAEキャッシュデコード完了 ===")
    return image

# 元のデコード関数（比較用）
@torch.no_grad()
def vae_decode(latents, vae):
    """通常のVAEデコード処理（全フレーム一括）"""
    latents = latents / vae.config.scaling_factor
    # 一括でデコード
    image = vae.decode(latents.to(device=vae.device, dtype=vae.dtype)).sample
    return image

# メモリ・速度のベンチマーク関数
def benchmark_vae_decode(latents, vae, method="both"):
    """VAEデコードのベンチマーク関数"""
    results = {}
    
    if method in ["original", "both"]:
        # 通常のデコード
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        with torch.no_grad():
            start = time.time()
            images_o = vae_decode(latents, vae)
            torch.cuda.synchronize()
            end = time.time()
            
            mem_o = torch.cuda.max_memory_allocated()
            results["original"] = {
                "images": images_o,
                "memory": mem_o / (1024**2),
                "time": end - start
            }
            print(f"vae_decode() メモリ使用量: {mem_o / (1024**2):.2f} MB 実行時間: {end - start:.4f} 秒")
    
    if method in ["cache", "both"]:
        # キャッシュを使用したデコード
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        with torch.no_grad():
            start = time.time()
            images_c = vae_decode_cache(latents, vae)
            torch.cuda.synchronize()
            end = time.time()
            
            mem_c = torch.cuda.max_memory_allocated()
            results["cache"] = {
                "images": images_c,
                "memory": mem_c / (1024**2),
                "time": end - start
            }
            print(f"vae_decode_cache() メモリ使用量: {mem_c / (1024**2):.2f} MB 実行時間: {end - start:.4f} 秒")
    
    # 両方のメソッドを実行した場合に結果の差異を表示
    if method == "both":
        diff = (results["original"]["images"] - results["cache"]["images"]).abs().mean()
        print(f"出力画像の平均差異: {diff.item():.6f}")
    
    return results
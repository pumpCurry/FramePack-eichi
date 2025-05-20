import torch
import traceback
import gc
from diffusers_helper.memory import DynamicSwapInstaller
from locales.i18n_extended import translate

class TextEncoderManager:
    """text_encoderとtext_encoder_2の状態管理を行うクラス
    
    このクラスは以下の責務を持ちます：
    - text_encoderとtext_encoder_2のライフサイクル管理
    
    設定の変更はすぐには適用されず、次回のリロード時に適用されます。
    """

    def __init__(self, device, high_vram_mode=False):
        self.text_encoder = None
        self.text_encoder_2 = None
        self.device = device

        # 現在適用されている設定
        self.current_state = {
            'is_loaded': False,
            'high_vram': high_vram_mode
        }

        # 次回のロード時に適用する設定
        self.next_state = self.current_state.copy()
        
    def set_next_settings(self, high_vram_mode=False):
        """次回のロード時に使用する設定をセット（即時のリロードは行わない）
        
        Args:
            high_vram_mode: High-VRAMモードの有効/無効
        """
        self.next_state = {
            'high_vram': high_vram_mode,
            'is_loaded': self.current_state['is_loaded']
        }
        print(translate("次回のtext_encoder設定を設定しました:"))
        print(f"  - High-VRAM mode: {high_vram_mode}")
    
    def _needs_reload(self):
        """現在の状態と次回の設定を比較し、リロードが必要かどうかを判断"""
        if not self._is_loaded():
            return True

        # High-VRAMモードの比較
        if self.current_state['high_vram'] != self.next_state['high_vram']:
            return True

        return False
    
    def _is_loaded(self):
        """text_encoderとtext_encoder_2が読み込まれているかどうかを確認"""
        return (self.text_encoder is not None and 
                self.text_encoder_2 is not None and 
                self.current_state['is_loaded'])
    
    def get_text_encoders(self):
        """現在のtext_encoderとtext_encoder_2インスタンスを取得"""
        return self.text_encoder, self.text_encoder_2

    def dispose_text_encoders(self):
        """text_encoderとtext_encoder_2のインスタンスを破棄し、メモリを完全に解放"""
        try:
            print(translate("\ntext_encoderとtext_encoder_2のメモリを解放します..."))
            
            # text_encoderの破棄
            if hasattr(self, 'text_encoder') and self.text_encoder is not None:
                try:
                    self.text_encoder.cpu()
                    del self.text_encoder
                    self.text_encoder = None
                except Exception as e:
                    print(translate("text_encoderの破棄中にエラー: {0}").format(e))

            # text_encoder_2の破棄
            if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
                try:
                    self.text_encoder_2.cpu()
                    del self.text_encoder_2
                    self.text_encoder_2 = None
                except Exception as e:
                    print(translate("text_encoder_2の破棄中にエラー: {0}").format(e))

            # 明示的なメモリ解放
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # 状態を更新
            self.current_state['is_loaded'] = False
            self.next_state['is_loaded'] = False
            
            print(translate("text_encoderとtext_encoder_2のメモリ解放が完了しました"))
            return True
            
        except Exception as e:
            print(translate("text_encoderとtext_encoder_2のメモリ解放中にエラー: {0}").format(e))
            traceback.print_exc()
            return False

    def ensure_text_encoder_state(self):
        """text_encoderとtext_encoder_2の状態を確認し、必要に応じてリロード"""
        if self._needs_reload():
            print(translate("text_encoderとtext_encoder_2をリロードします"))
            return self._reload_text_encoders()        
        print(translate("ロード済みのtext_encoderとtext_encoder_2を再度利用します"))
        return True
    
    def _reload_text_encoders(self):
        """next_stateの設定でtext_encoderとtext_encoder_2をリロード"""
        try:
            # 既存のモデルが存在する場合は先にメモリを解放
            if self._is_loaded():
                self.dispose_text_encoders()

            # 新しいtext_encoderとtext_encoder_2インスタンスを作成
            from transformers import LlamaModel, CLIPTextModel
            self.text_encoder = LlamaModel.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo", 
                subfolder='text_encoder', 
                torch_dtype=torch.float16
            ).cpu()
            
            self.text_encoder_2 = CLIPTextModel.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo", 
                subfolder='text_encoder_2', 
                torch_dtype=torch.float16
            ).cpu()
            
            self.text_encoder.eval()
            self.text_encoder_2.eval()
            
            self.text_encoder.to(dtype=torch.float16)
            self.text_encoder_2.to(dtype=torch.float16)
            
            self.text_encoder.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
            
            # VRAMモードに応じた設定
            if not self.next_state['high_vram']:
                DynamicSwapInstaller.install_model(self.text_encoder, device=self.device)
                DynamicSwapInstaller.install_model(self.text_encoder_2, device=self.device)
            else:
                self.text_encoder.to(self.device)
                self.text_encoder_2.to(self.device)
            
            # 状態を更新
            self.next_state['is_loaded'] = True
            self.current_state = self.next_state.copy()
            
            print(translate("text_encoderとtext_encoder_2のリロードが完了しました"))
            return True
            
        except Exception as e:
            print(translate("text_encoderとtext_encoder_2リロードエラー: {0}").format(e))
            traceback.print_exc()
            self.current_state['is_loaded'] = False
            return False 

"""
VAE設定を管理するためのモジュール。
タイリングや分割処理のパラメータを管理し、ghosting問題を軽減するための設定を提供します。
"""

import os
import json
import torch

# 設定ファイルのパス
VAE_SETTINGS_FILENAME = 'vae_settings.json'

# VAE設定のデフォルト値
DEFAULT_VAE_SETTINGS = {
    'use_tiling': True,              # タイリングを使用するかどうか
    'use_slicing': True,             # スライシングを使用するかどうか
    'tile_size': 512,                # タイルサイズ (画像空間)
    'latent_tile_size': 64,          # タイルサイズ (潜在空間)
    'tile_overlap': 0.25,            # タイル間のオーバーラップ係数 (0-1)
    'custom_vae_settings': False     # カスタム設定を使用するかどうか
}

def get_vae_settings_path():
    """VAE設定ファイルのパスを取得する"""
    # eichi_utils直下からwebuiフォルダに移動
    webui_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    settings_folder = os.path.join(webui_path, 'settings')
    return os.path.join(settings_folder, VAE_SETTINGS_FILENAME)

def load_vae_settings():
    """VAE設定をロードする"""
    settings_path = get_vae_settings_path()
    
    if os.path.exists(settings_path):
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                # 新しい設定項目がある場合はデフォルト値を設定
                for key, default_value in DEFAULT_VAE_SETTINGS.items():
                    if key not in settings:
                        settings[key] = default_value
                
                # カスタム設定が有効な場合のみ詳細ログを表示
                if settings.get('custom_vae_settings', False):
                    print(f"[VAE設定] 現在の設定: カスタム設定=有効" + 
                          f", タイリング={settings.get('use_tiling', True)}, " +
                          f"スライシング={settings.get('use_slicing', True)}")
                
                return settings
        except Exception as e:
            print(f"[VAE設定] 設定ファイル読み込みエラー - デフォルト使用")
            return DEFAULT_VAE_SETTINGS.copy()
    else:
        # 設定ファイルがない場合は静かにデフォルト値を返す
        return DEFAULT_VAE_SETTINGS.copy()

def save_vae_settings(settings):
    """VAE設定を保存する"""
    settings_path = get_vae_settings_path()
    
    # settings_pathの親ディレクトリが存在することを確認
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    
    try:
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        print(f"[VAE設定] 設定を保存しました: {settings_path}")
        return True
    except Exception as e:
        print(f"[VAE設定] 設定ファイルの保存エラー: {e}")
        return False

def debug_vae_properties(vae):
    """VAEオブジェクトのタイリング/スライシング関連のプロパティとメソッドを出力する"""
    if vae is None:
        print("VAEがロードされていません")
        return
        
    print("[VAE DEBUG] VAEオブジェクトのタイリング/スライシング関連プロパティとメソッド:")
    # classの基本情報
    print(f"VAEクラス: {vae.__class__.__name__}")
    
    # タイリング/スライシング関連プロパティの検索
    tile_props = []
    slice_props = []
    for name in dir(vae):
        if name.startswith('_'):  # プライベートメンバーを除外
            continue
            
        attr = getattr(vae, name)
        if 'tile' in name.lower():
            if callable(attr):
                tile_props.append(f"{name}() (メソッド)")
            else:
                tile_props.append(f"{name} = {attr}")
                
        if 'slice' in name.lower() or 'slic' in name.lower():
            if callable(attr):
                slice_props.append(f"{name}() (メソッド)")
            else:
                slice_props.append(f"{name} = {attr}")
    
    if tile_props:
        print("タイリング関連: " + ", ".join(f"{prop}" for prop in tile_props))
    
    if slice_props:
        print("スライシング関連: " + ", ".join(f"{prop}" for prop in slice_props))
    
    # configプロパティの確認（重要な設定情報がここにある可能性）
    if hasattr(vae, 'config'):
        print("[VAE DEBUG] config内のタイリング/スライシング関連プロパティ:")
        config_props = []
        for key, value in vars(vae.config).items():
            if 'tile' in key.lower() or 'slice' in key.lower() or 'slic' in key.lower():
                config_props.append(f"{key} = {value}")
        
        if config_props:
            print("  " + ", ".join(config_props))
        else:
            print("  関連プロパティはconfigには見つかりませんでした")

def apply_vae_settings(vae, settings=None):
    """VAEにタイリングとスライシングの設定を適用する

    Args:
        vae: VAEモデル
        settings (dict, optional): VAE設定。指定されない場合はロードする

    Returns:
        vae: 設定適用後のVAEモデル
    """
    import time
    import torch
    
    if vae is None:
        return vae
    
    # 設定がなければロード
    if settings is None:
        settings = load_vae_settings()
    
    # 設定値を明示的に保存するためのカスタム属性を追加
    if not hasattr(vae, '_vae_custom_settings'):
        vae._vae_custom_settings = {}
    
    # カスタム設定が有効な場合のみ詳細処理
    custom_enabled = settings.get('custom_vae_settings', False)
    
    # カスタム設定が有効な場合のみデバッグ出力
    if custom_enabled and not hasattr(vae, '_debug_printed'):
        debug_vae_properties(vae)
        vae._debug_printed = True
    
    # カスタム設定が有効な場合のみ適用
    if custom_enabled:
        # 設定値をカスタムプロパティに保存
        vae._vae_custom_settings['use_tiling'] = settings.get('use_tiling', True)
        vae._vae_custom_settings['use_slicing'] = settings.get('use_slicing', True)
        vae._vae_custom_settings['tile_size'] = settings.get('tile_size', 512)
        vae._vae_custom_settings['latent_tile_size'] = settings.get('latent_tile_size', 64)
        vae._vae_custom_settings['tile_overlap'] = settings.get('tile_overlap', 0.25)
        vae._vae_custom_settings['custom_vae_settings'] = True
        
        # タイリング設定
        if settings.get('use_tiling', True):
            vae.use_tiling = True
            if hasattr(vae, 'enable_tiling') and callable(vae.enable_tiling):
                vae.enable_tiling()
            
            # HunyuanVideoモデル用の代替設定
            if hasattr(vae, 'enable_tile') and callable(getattr(vae, 'enable_tile')):
                getattr(vae, 'enable_tile')()
                print(f"[VAE設定] HunyuanVideo用の enable_tile() メソッドを呼び出しました")
        else:
            vae.use_tiling = False
            if hasattr(vae, 'disable_tiling') and callable(vae.disable_tiling):
                vae.disable_tiling()
            
            # HunyuanVideoモデル用の代替設定
            if hasattr(vae, 'disable_tile') and callable(getattr(vae, 'disable_tile')):
                getattr(vae, 'disable_tile')()
                print(f"[VAE設定] HunyuanVideo用の disable_tile() メソッドを呼び出しました")
        
        # スライシング設定
        if settings.get('use_slicing', True):
            vae.use_slicing = True
            if hasattr(vae, 'enable_slicing') and callable(vae.enable_slicing):
                vae.enable_slicing()
            
            # HunyuanVideoモデル用の代替設定
            if hasattr(vae, 'enable_slic') and callable(getattr(vae, 'enable_slic')):
                getattr(vae, 'enable_slic')()
                print(f"[VAE設定] HunyuanVideo用の enable_slic() メソッドを呼び出しました")
        else:
            vae.use_slicing = False
            if hasattr(vae, 'disable_slicing') and callable(vae.disable_slicing):
                vae.disable_slicing()
            
            # HunyuanVideoモデル用の代替設定
            if hasattr(vae, 'disable_slic') and callable(getattr(vae, 'disable_slic')):
                getattr(vae, 'disable_slic')()
                print(f"[VAE設定] HunyuanVideo用の disable_slic() メソッドを呼び出しました")
        
        # タイルサイズパラメータの設定 - 複数の可能なプロパティ名を試す
        tile_size = settings.get('tile_size', 512)
        latent_tile_size = settings.get('latent_tile_size', 64)
        tile_overlap = settings.get('tile_overlap', 0.25)
        
        # 標準プロパティ
        tile_props = {
            'tile_sample_min_size': tile_size,
            'tile_size': tile_size,
            'sample_tile_size': tile_size,
            'min_tile_size': tile_size
        }
        
        latent_props = {
            'tile_latent_min_size': latent_tile_size,
            'latent_tile_size': latent_tile_size,
            'tile_latent_size': latent_tile_size,
            'latent_size': latent_tile_size
        }
        
        overlap_props = {
            'tile_overlap_factor': tile_overlap,
            'tile_overlap': tile_overlap,
            'overlap_factor': tile_overlap,
            'overlap': tile_overlap
        }
        
        # それぞれのプロパティを試す
        set_props = []
        for key, val in tile_props.items():
            if hasattr(vae, key):
                setattr(vae, key, val)
                set_props.append(f"{key} = {val}")
        
        for key, val in latent_props.items():
            if hasattr(vae, key):
                setattr(vae, key, val)
                set_props.append(f"{key} = {val}")
        
        for key, val in overlap_props.items():
            if hasattr(vae, key):
                setattr(vae, key, val)
                set_props.append(f"{key} = {val}")
                
        if set_props:
            print(f"[VAE設定] 設定したプロパティ: {', '.join(set_props)}")
        
        # configオブジェクトにも設定を試みる
        if hasattr(vae, 'config'):
            config_set_props = []
            for key, val in tile_props.items():
                if hasattr(vae.config, key):
                    setattr(vae.config, key, val)
                    config_set_props.append(f"{key} = {val}")
            
            for key, val in latent_props.items():
                if hasattr(vae.config, key):
                    setattr(vae.config, key, val)
                    config_set_props.append(f"{key} = {val}")
            
            for key, val in overlap_props.items():
                if hasattr(vae.config, key):
                    setattr(vae.config, key, val)
                    config_set_props.append(f"{key} = {val}")
                    
            if config_set_props:
                print(f"[VAE設定] config設定したプロパティ: {', '.join(config_set_props)}")
        
        # 標準のプロパティも設定（ない場合は新しく作成）
        vae.tile_sample_min_size = tile_size
        vae.tile_latent_min_size = latent_tile_size
        vae.tile_overlap_factor = tile_overlap
        
        # カスタム設定有効時のみ設定値を表示
        print("[VAE設定] 現在のVAE実際の設定値:")
        
        # 実際のプロパティと内部保存値を両方確認
        actual_tiling = getattr(vae, 'use_tiling', 'N/A')
        actual_slicing = getattr(vae, 'use_slicing', 'N/A')
        actual_tile_size = getattr(vae, 'tile_sample_min_size', 'N/A')
        actual_latent_tile_size = getattr(vae, 'tile_latent_min_size', 'N/A')
        actual_tile_overlap = getattr(vae, 'tile_overlap_factor', 'N/A')
        
        # カスタム保存値があれば取得
        stored_settings = getattr(vae, '_vae_custom_settings', {})
        stored_tiling = stored_settings.get('use_tiling', 'N/A')
        stored_slicing = stored_settings.get('use_slicing', 'N/A')
        stored_tile_size = stored_settings.get('tile_size', 'N/A')
        stored_latent_tile_size = stored_settings.get('latent_tile_size', 'N/A')
        stored_tile_overlap = stored_settings.get('tile_overlap', 'N/A')
        
        # 実際の値と保存値を表示
        print(f"タイリング: {actual_tiling} (保存値: {stored_tiling}), スライシング: {actual_slicing} (保存値: {stored_slicing})")
        print(f"タイルサイズ(画像空間): {actual_tile_size} (保存値: {stored_tile_size}), タイルサイズ(潜在空間): {actual_latent_tile_size} (保存値: {stored_latent_tile_size})")
        print(f"タイルオーバーラップ: {actual_tile_overlap} (保存値: {stored_tile_overlap})")
        
        if hasattr(vae, 'config'):
            print(f"モデル設定: {vae.__class__.__name__}" + (f", サンプルサイズ: {vae.config.sample_size}" if hasattr(vae.config, 'sample_size') else ""))
    else:
        # カスタム設定が無効な場合は最小限の操作
        vae._vae_custom_settings['custom_vae_settings'] = False
        
        # カスタム設定が無効でも基本的なタイリングとスライシングは有効にする
        # ただし、ログは出力しない
        vae.use_tiling = True
        if hasattr(vae, 'enable_tiling') and callable(vae.enable_tiling):
            vae.enable_tiling()
        if hasattr(vae, 'enable_tile') and callable(getattr(vae, 'enable_tile')):
            getattr(vae, 'enable_tile')()
            
        vae.use_slicing = True
        if hasattr(vae, 'enable_slicing') and callable(vae.enable_slicing):
            vae.enable_slicing()
        if hasattr(vae, 'enable_slic') and callable(getattr(vae, 'enable_slic')):
            getattr(vae, 'enable_slic')()
    
    return vae


def get_current_vae_settings_display(vae):
    """現在のVAE設定を表示用文字列として取得する

    Args:
        vae: VAEモデル

    Returns:
        str: 現在の設定を表示する文字列
    """
    if vae is None:
        return "VAEがロードされていません"
    
    # カスタム設定の有効状態を確認
    stored_settings = getattr(vae, '_vae_custom_settings', {})
    custom_enabled = stored_settings.get('custom_vae_settings', False)
    
    result = []
    
    # カスタム設定が有効な場合のみ詳細表示
    if custom_enabled:
        result.append("### Current VAE Settings (Actually Applied Values)")
        
        # 実際のプロパティと内部保存値を両方確認
        actual_tiling = getattr(vae, 'use_tiling', 'N/A')
        actual_slicing = getattr(vae, 'use_slicing', 'N/A')
        actual_tile_size = getattr(vae, 'tile_sample_min_size', 'N/A')
        actual_latent_tile_size = getattr(vae, 'tile_latent_min_size', 'N/A')
        actual_tile_overlap = getattr(vae, 'tile_overlap_factor', 'N/A')
        
        # 保存値を取得
        stored_tiling = stored_settings.get('use_tiling', 'N/A')
        stored_slicing = stored_settings.get('use_slicing', 'N/A')
        stored_tile_size = stored_settings.get('tile_size', 'N/A')
        stored_latent_tile_size = stored_settings.get('latent_tile_size', 'N/A')
        stored_tile_overlap = stored_settings.get('tile_overlap', 'N/A')
        
        # 設定値の有効状態を表示
        result.append(f"- **Custom Settings**: {'Enabled' if custom_enabled else 'Disabled'}")
        
        # 実際の値と保存値を表示
        result.append(f"- **Tiling**: {actual_tiling} (Settings: {stored_tiling})")
        result.append(f"- **Slicing**: {actual_slicing} (Settings: {stored_slicing})")
        result.append(f"- **Tile Size (Image Space)**: {actual_tile_size} (Settings: {stored_tile_size})")
        result.append(f"- **Tile Size (Latent Space)**: {actual_latent_tile_size} (Settings: {stored_latent_tile_size})")
        result.append(f"- **Tile Overlap**: {actual_tile_overlap} (Settings: {stored_tile_overlap})")
        
        if hasattr(vae, 'config'):
            result.append(f"- **Model Type**: {vae.__class__.__name__}")
            if hasattr(vae.config, 'sample_size'):
                result.append(f"- **Sample Size**: {vae.config.sample_size}")
        
        # クラス情報と利用可能なメソッドの表示
        result.append(f"\n**VAE Class**: `{vae.__class__.__module__}.{vae.__class__.__name__}`")
        
        # 利用可能なメソッドの確認
        tile_methods = []
        slice_methods = []
        
        # 標準メソッド
        for name, method_list in [
            ('enable_tiling', tile_methods),
            ('disable_tiling', tile_methods),
            ('enable_slicing', slice_methods),
            ('disable_slicing', slice_methods),
            # HunyuanVideo用の代替メソッド
            ('enable_tile', tile_methods),
            ('disable_tile', tile_methods),
            ('enable_slic', slice_methods),
            ('disable_slic', slice_methods)
        ]:
            if hasattr(vae, name) and callable(getattr(vae, name)):
                method_list.append(f"{name}()")
        
        # タイルサイズ関連プロパティの検索
        tile_properties = {}
        for name in dir(vae):
            if name.startswith('_'):
                continue
            if 'tile' in name.lower() or 'slic' in name.lower():
                attr = getattr(vae, name)
                if not callable(attr):
                    tile_properties[name] = attr
        
        if tile_methods or slice_methods:
            result.append("\n**Available Methods**:")
            if tile_methods:
                result.append(f"- Tiling: {', '.join(tile_methods)}")
            if slice_methods:
                result.append(f"- Slicing: {', '.join(slice_methods)}")
        
        if tile_properties:
            result.append("\n**Available Tile-Related Properties**:")
            for name, value in tile_properties.items():
                result.append(f"- {name} = {value}")
    else:
        # カスタム設定が無効な場合は最小限の情報だけ表示
        # Note: これらの文字列は翻訳されない固定テキスト - 翻訳はUIコンポーネント作成時に行われる
        result.append("### VAE Settings")
        result.append("- **Custom Settings**: Disabled")
        result.append("- **Tiling**: Enabled")
        result.append("- **Slicing**: Enabled")
        
        if hasattr(vae, 'config'):
            result.append(f"- **Model Type**: {vae.__class__.__name__}")
    
    return "\n".join(result)

def create_vae_settings_ui(translate_fn):
    """VAE設定用のUIコンポーネントを作成する

    Args:
        translate_fn: 翻訳関数

    Returns:
        タプル: (VAE設定アコーディオン, コントロールの辞書)
    """
    import gradio as gr
    
    # 現在の設定をロード
    print("[VAE設定] UI作成のための設定ロード")
    current_settings = load_vae_settings()
    
    with gr.Accordion(translate_fn("VAE詳細設定 (ゴースト対策)"), open=False) as vae_settings_accordion:
        # 現在の実際の設定値を表示
        current_settings_md = gr.Markdown(
            "### VAEの設定値\nモデルがロードされると、ここに現在の設定値が表示されます",
            elem_id="vae_current_settings"
        )
        
        # 再起動が必要な旨の注意
        gr.Markdown(
            translate_fn("⚠️ **注意**: 設定変更後は保存ボタンを押し、アプリケーションを**再起動**する必要があります。設定はリアルタイムに反映されません。タイリング機能をオフにするとVRAM使用量が劇的に増加するため、通常はオンのままにしてください。"),
            elem_classes=["warning-box"]
        )
        
        # カスタム設定の有効化チェックボックス
        custom_vae_settings = gr.Checkbox(
            label=translate_fn("カスタムVAE設定を有効化"),
            value=current_settings.get('custom_vae_settings', False),
            info=translate_fn("チェックすると、詳細なVAE設定が適用されます。ゴースティング（残像）問題の改善に役立つ可能性があります。")
        )
        
        # VAE設定グループ
        with gr.Group(visible=current_settings.get('custom_vae_settings', False)) as vae_settings_group:
            with gr.Row():
                use_tiling = gr.Checkbox(
                    label=translate_fn("タイリングを使用"),
                    value=current_settings.get('use_tiling', True),
                    info=translate_fn("VAEの分割処理を使用するかどうか。無効にするとVRAM使用量が劇的に増加するため、通常はオンのままにしてください。効果については現在評価中です。")
                )
                
                use_slicing = gr.Checkbox(
                    label=translate_fn("スライシングを使用"),
                    value=current_settings.get('use_slicing', True),
                    info=translate_fn("VAEのバッチスライシングを使用するかどうか。無効にするとVRAM使用量が増加します。効果については現在評価中です。")
                )
            
            with gr.Row():
                tile_size = gr.Slider(
                    label=translate_fn("タイルサイズ"),
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=current_settings.get('tile_size', 512),
                    info=translate_fn("画像空間でのタイルサイズ。大きいほどゴーストが減る可能性がありますが、効果については現在評価中です。VRAM使用量が増加します。")
                )
                
                latent_tile_size = gr.Slider(
                    label=translate_fn("潜在空間タイルサイズ"),
                    minimum=32,
                    maximum=128,
                    step=16,
                    value=current_settings.get('latent_tile_size', 64),
                    info=translate_fn("潜在空間でのタイルサイズ。大きいほどゴーストが減る可能性がありますが、効果については現在評価中です。VRAM使用量が増加します。")
                )
            
            tile_overlap = gr.Slider(
                label=translate_fn("タイルオーバーラップ"),
                minimum=0.0,
                maximum=0.5,
                step=0.05,
                value=current_settings.get('tile_overlap', 0.25),
                info=translate_fn("タイル間のオーバーラップ係数。大きいほどタイル境界が目立たなくなる可能性がありますが、効果については現在評価中です。処理時間が増加します。")
            )
        
        # カスタム設定チェックボックスの変更時にグループの表示/非表示を切り替え
        def toggle_vae_settings_group(value):
            print(f"[VAE設定] カスタム設定の有効化状態を変更: {value}")
            return gr.update(visible=value)
        
        custom_vae_settings.change(
            fn=toggle_vae_settings_group,
            inputs=[custom_vae_settings],
            outputs=[vae_settings_group]
        )
        
        # 設定保存用関数
        def save_vae_settings_from_ui(custom_enabled, use_tiling_val, use_slicing_val, 
                                    tile_size_val, latent_tile_size_val, tile_overlap_val):
            print("[VAE設定] 保存設定: カスタム設定=" + ("有効" if custom_enabled else "無効") + 
                  f", タイリング={use_tiling_val}, スライシング={use_slicing_val}, サイズ={tile_size_val}")
            
            # 設定を辞書に格納
            settings = {
                'custom_vae_settings': custom_enabled,
                'use_tiling': use_tiling_val,
                'use_slicing': use_slicing_val,
                'tile_size': tile_size_val,
                'latent_tile_size': latent_tile_size_val,
                'tile_overlap': tile_overlap_val
            }
            
            # 設定をファイルに保存
            save_result = save_vae_settings(settings)
            
            # UIに表示するメッセージを返す
            result_message = translate_fn("設定を保存しました。反映には再起動が必要です。") if save_result else translate_fn("設定の保存に失敗しました")
            return result_message
        
        with gr.Row():
            # 保存ボタン
            save_button = gr.Button(translate_fn("VAE設定を保存（再起動が必要）"), variant="primary")
            
            # デフォルトに戻すボタン
            reset_button = gr.Button(translate_fn("デフォルトに戻す（再起動が必要）"), variant="secondary")
            
        save_result = gr.Markdown("")
        
        # 保存ボタンのクリックイベント
        save_button.click(
            fn=save_vae_settings_from_ui,
            inputs=[
                custom_vae_settings,
                use_tiling,
                use_slicing,
                tile_size,
                latent_tile_size,
                tile_overlap
            ],
            outputs=[save_result]
        )
        
        # デフォルトに戻す関数
        def reset_to_defaults():
            print("[VAE設定] デフォルト設定に戻します")
            # デフォルト値から新しい設定を作成
            settings = DEFAULT_VAE_SETTINGS.copy()
            # 設定を保存
            save_result = save_vae_settings(settings)
            
            result_message = translate_fn("デフォルト設定に戻しました。反映には再起動が必要です。") if save_result else translate_fn("設定のリセットに失敗しました")
            
            # UIの値も更新
            return (
                settings['custom_vae_settings'],  # custom_vae_settings
                settings['use_tiling'],           # use_tiling
                settings['use_slicing'],          # use_slicing
                settings['tile_size'],            # tile_size
                settings['latent_tile_size'],     # latent_tile_size
                settings['tile_overlap'],         # tile_overlap
                gr.update(visible=settings['custom_vae_settings']),  # settings_group visibility
                result_message                    # save_result
            )
        
        # リセットボタンのクリックイベント
        reset_button.click(
            fn=reset_to_defaults,
            inputs=[],
            outputs=[
                custom_vae_settings,
                use_tiling,
                use_slicing,
                tile_size,
                latent_tile_size,
                tile_overlap,
                vae_settings_group,
                save_result
            ]
        )
    
    # 関連するコントロールを辞書で返す
    controls = {
        'custom_vae_settings': custom_vae_settings,
        'use_tiling': use_tiling,
        'use_slicing': use_slicing,
        'tile_size': tile_size,
        'latent_tile_size': latent_tile_size,
        'tile_overlap': tile_overlap,
        'save_button': save_button,
        'save_result': save_result,
        'settings_group': vae_settings_group,
        'current_settings_md': current_settings_md
    }
    
    print("[VAE設定] UIコンポーネント作成完了")
    return vae_settings_accordion, controls
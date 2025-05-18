# FramePack-eichi 更新履歴 / Update History / 更新历史 / История изменений

このファイルはFramePack-eichiの更新履歴を記録しています。[README](README.md)や[使用ガイド](README_userguide.md)も併せてご参照ください。

## 日本語

### 2025-05-15: バージョン1.9.3
- **プロンプトキューと画像キュー**:
  - テキストファイルから複数のプロンプトを読み込み順次処理
  - 指定フォルダ内の画像を自動的に順次処理  
  - キュー有効時は処理完了後に自動的に次の項目を処理
- **FP8最適化の改善**:
  - LoRA未使用時のメモリ使用量を大幅に削減
  - 処理速度の向上（PyTorch 2.1以上が必要）
  - 通常使用時はデフォルトで有効に
- **LoRAプリセットマネージャー**:
  - 5つのプリセットを保存・管理可能
  - UIから簡単にプリセットを切り替え
  - LoRAファイル、強度、その他の設定を一括保存
- **UIの整理と多言語対応**:
  - 関連機能をグループ化し操作性を向上
  - 各機能の説明を追加し理解しやすく
  - ロシア語UIの追加で4言語対応に拡張（日本語、英語、中国語、ロシア語）
- **動画生成時間の拡張**:
  - eichi(無印)の1秒モード(33フレーム)時に30秒、40秒を追加
  - F1に30秒、40秒、60秒、120秒の選択肢を追加
  - より多様な動画長に対応、長尺動画の生成が可能に
- **セクション管理の強化**:
  - インポート/エクスポート時のパス正規化
  - 異なる環境間でのセクション情報の互換性を向上
- **kisekaeichi機能のoichi統合**:
  - furusu氏考案・Kohya氏実装の参照画像を用いた動画生成技術をoichiに統合
  - ターゲットインデックスと履歴インデックスによる精密な制御
  - 特定領域のみを変更可能なマスク機能も実装  
- **バグ修正と最適化**:
  - プロンプト追加と削除の問題を解消
  - asyncio関連のWindowsエラーを回避
  - F1モデルの効率的なダウンロード機能を追加

### 2025-05-11: バージョン1.9.2
- **「FramePack-oichi」新機能追加**:
  - 1枚の入力画像から次の1枚の未来フレーム画像を予測生成する新機能
  - 専用の起動スクリプト（`run_oneframe_ichi.bat`他）を追加
  - 通常の動画生成より軽量で手軽に次の1フレームを確認可能
  - 多言語対応（日本語、英語、中国語）を完備
- **フレーム画像保存機能**:
  - 生成した全フレーム画像を保存するオプションを追加
  - 全セクションの全フレーム保存か最終セクションのみの保存か選択可能
  - 動画の中間過程の可視化や素材としての活用が容易に
- **セクション情報の一括管理機能強化**:
  - セクション情報のZIPファイルによる一括ダウンロード機能を追加
  - 複数プロジェクトの効率的な管理・バックアップが可能に
  - 開始画像、終了画像、セクション情報(プロンプト、画像)の一括アップロード及び内容変更後の再一括ダウンロード反映に対応
- **LoRA機能の強化**:
  - 3つのLoRAの同時使用に対応
  - ディレクトリから選択をデフォルトとし、/webui/loraフォルダに格納されたLoRAを選択可能
  - 全モード（無印版、F1版、oneframe版）でLoRA機能強化を対応
- **VAEキャッシュ機能**:
  - フレーム単位でのVAEデコードによる処理速度向上 ※furusu氏の検証に基づく実装 [詳細1](https://note.com/gcem156/n/nb93535d80c82) [詳細2](https://github.com/laksjdjf/FramePack)
  - メモリ使用量と処理速度のバランス調整が柔軟に
  - 設定画面から簡単にオン/オフを切り替え可能
  - デフォルトはOFFのため影響なし
  - フレーム間の独立性を活かした計算キャッシュにより最大30%程度の高速化を実現

### 2025-05-04: バージョン1.9.1
- **F1モデルの追加**:
  - 順生成に対応した新モデル「FramePack_F1_I2V_HY_20250503」を導入
  - 専用の起動スクリプト（`run_endframe_ichi_f1.bat`他）を追加
  - F1ではセクション（キーフレーム画像）とFinal（endframe）機能を削除
  - Image影響度の計算と設定範囲を見直し（100.0%～102.0%）
  - シンプルな操作性と活発な動きの生成を実現
- **メモリ管理機能の強化**:
  - transformer_manager.pyとtext_encoder_manager.pyによる効率的なモデル管理
  - FP8最適化機能の追加による大幅なVRAM使用量削減
  - RTX 40シリーズGPU向けのscaled_mm最適化対応
- **Docker対応の強化**:
  - コンテナ化環境での簡単なセットアップをサポート
  - CUDA 12.6、PyTorch 2.6.0環境に対応
  - 多言語対応（日本語、英語、中国語）コンテナ
- **PNGメタデータ機能**:
  - 生成画像へのプロンプト、シード、セクション情報の自動埋め込み
  - 保存された画像からの設定再取得機能
  - SD系との互換性を持つメタデータ形式
- **最適解像度バケットシステム**:
  - アスペクト比に応じた最適な解像度を自動選択
  - 安全な標準解像度（512, 640, 768）への自動調整
  - 様々な出力サイズでの一貫した品質確保
- **クリップボード対応の拡充**:
  - 無印版のImageおよびFinalとF1版のImageのクリップボード対応
  - 外部ツールとの連携強化

### 2025-04-30: バージョン1.9
- **kohya-ss/FramePack-LoRAReadyの導入**:
  - kohya-ss氏許諾の下、LoRA機能の性能と安定性を大幅に向上
  - 高VRAMモードと低VRAMモードのLoRA適用方式を直接適用方式に統一
  - load_and_apply_lora関数を共通で使用することでコードの複雑さが軽減
- **FP8最適化機能の導入**:
  - 8ビット浮動小数点形式を使用した量子化によるメモリ使用量と処理速度の最適化
  - 基本的にはOFFでの使用を推奨（一部環境では警告が出てエラーとなる場合あり）
  - RTX 40シリーズGPUでの高速化オプションもサポート
- **ディレクトリ構造の変更**:
  - diffusers_helper: モデルのメモリ管理改善用ユーティリティを追加
  - 既存の本家ツールのソースを差し替えるため、必要に応じてバックアップを推奨
- **HunyuanVideo形式への統一**:
  - LoRAフォーマットをHunyuanVideo形式に統一し互換性を向上

### 2025-04-29: バージョン1.8.1
- **多言語対応（i18n）の実装**:
  - 日本語、英語、繁体字中国語の3言語に対応
  - 言語別実行ファイル（`run_endframe_ichi_en.bat`、`run_endframe_ichi_zh-tw.bat`）の追加
  - UIテキスト、ログメッセージ、エラーメッセージを含むすべての要素の多言語化
  - JSONファイルベースの翻訳システムによる拡張性の向上
  - ※1.7.1時点の機能を中心に翻訳対応。1.8の機能は今後追加予定

### 2025-04-28: バージョン1.8
- **テンソルデータの結合機能を追加**:
  - 動画の潜在表現を.safetensors形式で保存する機能を追加
  - 保存したテンソルデータを新規生成する動画の後方に結合する機能を追加
  - チャンク処理方式による大規模テンソルデータの効率的な処理を実装
  - メモリ効率の向上とデバイス/型の互換性強化による安定性向上
  - 中間ファイルの自動削除機能とフィードバック表示の追加

### 2025-04-28: バージョン1.7.1
- **内部計算の最適化**:
  - 0.5秒モードのセクション数計算が正確になり、動画生成の安定性が向上
  - フレーム計算パラメータ（latent_window_size）が5から4.5に変更
- **UIの改善**:
  - 動画モード名の簡素化: 括弧付き表記を削除（例: "10(5x2)秒" → "10秒"）
  - レイアウトの整理: UI全体が整理され、より使いやすくなりました

### 2025-04-28: バージョン1.7.0
- **キーフレーム画像コピー機能の大幅改良**:
  - 赤枠/青枠による視覚的な区別を導入
  - **赤枠(セクション0)** ⇒ すべての偶数番号セクション(0,2,4,6...)に自動コピー
  - **青枠(セクション1)** ⇒ すべての奇数番号セクション(1,3,5,7...)に自動コピー
  - 動的セクション数計算の精度向上により、選択した動画長に応じて正確にコピー範囲を調整
  - ユーザーが必要に応じてコピー機能のオン/オフを切り替え可能に
- **コピー処理の安全性向上**:
  - セクション境界チェックを強化し、無効な位置へのコピーを防止
  - 詳細なログ出力によりコピー動作の追跡が容易に

### 2025-04-27: バージョン1.6.2
- **MP4圧縮設定の追加**: ※本家からマージ
  - 動画のファイルサイズと品質のバランスを調整可能に
  - 0〜100の範囲で設定可能（0=無圧縮、16=デフォルト、高い値=高圧縮・低品質）
  - 黒画面問題対策として最適な設定値を提供
- **コード品質の向上**:
  - UIスタイル定義を`ui_styles.py`として分離し、保守性と可読性を改善
  - CSSスタイルの一元管理によりUI一貫性を向上
- **フレームサイズ計算の微調整**:
  - 0.5秒モードの計算精度を向上（`latent_window_size=4.5`を使用）
  - セクション数と動画長の計算精度が向上し、より安定した動画生成が可能に

### 2025-04-26: バージョン1.6.1
- **短時間動画モードの拡充**:
  - 2秒モード: 60フレーム（2秒×30FPS）、2セクション、コピー不要
  - 3秒モード: 90フレーム（3秒×30FPS）、3セクション、キーフレーム0→1にコピー
  - 4秒モード: 120フレーム（4秒×30FPS）、4セクション、キーフレーム0→1,2にコピー
- **v1.5.1の基本構造に回帰**:
  - オリジナルのモード名表記（カッコ付き）を維持
  - キーフレームガイドHTMLを復活
  - 元々の関数構造・処理方法を維持

### 2025-04-25: バージョン1.5.1
- **短い動画生成向けの「1秒」モードを追加**:
  - 1秒（約30フレーム @ 30fps）に対応
- **デフォルト動画長を変更**:
  - 「6秒」から「1秒」にデフォルト値を変更
- **入力画像のコピー動作の最適化**:
  - 通常モードでの入力画像からのコピー処理を停止
  - ループモードでのみLastにコピーする処理を有効化
- **キーフレーム自動コピー機能のデフォルト値を変更**:
  - デフォルトでオフに設定し、より細かな制御を可能に
  - 必要な場合はオンにすることで自動コピー可能
- **画像処理の安定性向上**:
  - Startボタンを押した時に画像が正しく再取得されるよう改善
  - プレビュー画像の明示的なクリア処理を追加

### 2025-04-24: バージョン1.5.0
- **フレームサイズ設定の追加**:
  - 0.5秒モードと1秒モードを切り替え可能
  - フレームサイズによってlatent_window_sizeを動的に調整

### 2025-04-24: バージョン1.4.0
- **オールパディング機能の追加**:
  - すべてのセクションで同じパディング値を使用
  - この値が小さいほど1セッションでの動きが激しくなる

### 2025-04-24: バージョン1.3.2
- **LoRA適用機能の統一**:
  - 低VRAMモード（DynamicSwap）でも高VRAMモードと同様に直接LoRAを適用する方式に統一
  - フック方式を廃止し、より安定した直接適用方式のみをサポート
  - 互換性のためにインターフェースは維持しつつ、内部実装を改善
- **デバッグ・検証機能の強化**:
  - LoRAの適用状況を確認するための専用ユーティリティを追加（lora_check_helper.py）
  - 詳細なログ出力とデバッグ情報の提供

### 2025-04-24: バージョン1.3.1
- **コードベースのリファクタリング**: 保守性と拡張性向上のため、コードを複数のモジュールに整理
  - `eichi_utils`: キーフレーム処理、設定管理、プリセット管理、ビデオモード設定を管理
  - `lora_utils`: LoRA関連の機能を集約

### 2025-04-23: バージョン1.3
- **【調査中】Hunyuan LoRAサポートの追加**: モデルをカスタマイズして独自の表現を追加
- **【試験実装】EndFrame影響度設定の追加**: 最終フレームの影響力を0.01〜1.00の範囲で調整可能に（nirvash氏の知見）

### 2025-04-23: バージョン1.2
- **「20(4x5)秒」モードを追加**: 5パート構成の長時間動画生成に対応
- **セクションごとのプロンプト機能を追加**: 各セクションに個別のプロンプトを設定可能に【試験実装】
- **出力フォルダ管理機能を強化**: 出力先フォルダの指定とOSに依存しない開き方をサポート
- **設定ファイル管理の改善**: JSONベースの設定ファイルで設定を永続化
- **クロスプラットフォーム対応の強化**: Windows以外の環境での動作を改善

### 2025-04-22: バージョン1.1
- **「16(4x4)秒」モードを追加**: 4パート構成の長時間動画生成に対応
- **生成処理中の進捗表示を改善**: セクション情報（例：「セクション: 3/15」）が表示されるようになり、特に長い動画生成時の進捗が把握しやすくなりました
- **設定ファイルの構造化**: 動画モード設定が別ファイルに分離され、拡張性が向上

### 2025-04-21: 初版リリース
- プロンプト管理機能の最適化
- キーフレームガイド機能の追加

## English

### 2025-05-15: Version 1.9.3
- **Prompt Queue and Image Queue**:
  - Load multiple prompts from text file and process sequentially
  - Automatically process images in specified folder sequentially
  - Auto-process next item after completion when queue is enabled
- **FP8 Optimization Improvements**:
  - Significant memory usage reduction when not using LoRA
  - Improved processing speed (requires PyTorch 2.1 or higher)
  - Enabled by default for normal use
- **LoRA Preset Manager**:
  - Save and manage up to 5 presets
  - Easy preset switching from UI
  - Batch save of LoRA files, strengths, and other settings
- **UI Organization and Multilingual Support**:
  - Grouped related functions for improved usability
  - Added explanations for each function for better understanding
  - Russian UI added for 4-language support (Japanese, English, Chinese, Russian)
- **Video Generation Duration Extension**:
  - Added 30s, 40s options to eichi (standard) in 1s mode (33 frames)
  - Added 30s, 40s, 60s, 120s options to F1
  - Support for more diverse video lengths, enabling long video generation
- **Section Management Enhancement**:
  - Path normalization for import/export
  - Improved compatibility of section information between different environments
- **kisekaeichi Function Integration to oichi**:
  - Integration of reference image-based video generation technology devised by furusu and implemented by Kohya
  - Precise control through target index and history index
  - Mask function implemented for changing specific areas only
- **Bug Fixes and Optimization**:
  - Fixed prompt addition and deletion issues
  - Avoided Windows errors related to asyncio
  - Added efficient download function for F1 model

### 2025-05-11: Version 1.9.2
- **"FramePack-oichi" New Feature Added**:
  - New function to predict and generate the next future frame image from a single input image
  - Added dedicated startup scripts (`run_oneframe_ichi.bat` and others)
  - Lighter and easier way to check the next frame compared to regular video generation
  - Complete multilingual support (Japanese, English, Chinese)
- **Frame Image Save Function**:
  - Added option to save all generated frame images
  - Choice between saving all frames from all sections or only from the final section
  - Easier visualization of intermediate video process and use as material resources
- **Section Information Batch Management Enhancement**:
  - Added batch download function of section information via ZIP file
  - Efficient management and backup of multiple projects
  - Support for bulk upload of start images, end images, section information (prompts, images) and re-download after content changes
- **LoRA Function Enhancement**:
  - Support for simultaneous use of three LoRAs
  - Default selection from directory, ability to select LoRAs stored in the /webui/lora folder
  - LoRA function enhancement supported in all modes (Standard, F1, oneframe)
- **VAE Cache Function**:
  - Processing speed improvement through frame-by-frame VAE decoding - Based on research by furusu [Details1](https://note.com/gcem156/n/nb93535d80c82) [Details2](https://github.com/laksjdjf/FramePack)
  - Flexible adjustment of balance between memory usage and processing speed
  - Easy on/off switching from settings screen
  - No impact as default is OFF
  - Achieves up to 30% speed improvement through computational caching leveraging frame independence

### 2025-05-04: Version 1.9.1
- **F1 Model Added**:
  - Introduction of the new "FramePack_F1_I2V_HY_20250503" model supporting forward generation
  - Added dedicated startup scripts (`run_endframe_ichi_f1.bat` and others)
  - Removed Section (keyframe images) and Final (endframe) features in F1 version
  - Revised Image influence calculation and setting range (100.0% to 102.0%)
  - Achieved simple operation and dynamic motion generation
- **Memory Management Enhancement**:
  - Efficient model management with transformer_manager.py and text_encoder_manager.py
  - Significant VRAM usage reduction with FP8 optimization
  - Support for scaled_mm optimization for RTX 40 series GPUs
- **Docker Support Enhancement**:
  - Easy setup in containerized environments
  - Compatibility with CUDA 12.6 and PyTorch 2.6.0
  - Multilingual container support (Japanese, English, Chinese)
- **PNG Metadata Feature**:
  - Automatic embedding of prompts, seeds, and section information in generated images
  - Settings retrieval from saved images
  - Compatible metadata format with SD-based tools
- **Optimal Resolution Bucket System**:
  - Automatic selection of optimal resolution based on aspect ratio
  - Automatic adjustment to safe standard resolutions (512, 640, 768)
  - Consistent quality across various output sizes
- **Clipboard Support Enhancement**:
  - Clipboard support for Images and Final frames in standard version and Images in F1 version
  - Enhanced integration with external tools

### 2025-04-30: Version 1.9
- **Implementation of kohya-ss/FramePack-LoRAReady**:
  - Significantly improved LoRA functionality performance and stability with permission from Kohya Tech
  - Unified the LoRA application method for both high-VRAM and low-VRAM modes
  - Reduced code complexity by using a common load_and_apply_lora function
- **Introduction of FP8 Optimization**:
  - Memory usage and processing speed optimization through 8-bit floating-point quantization
  - Recommended to be OFF by default (may cause warnings or errors in some environments)
  - Support for acceleration options on RTX 40 series GPUs
- **Directory Structure Changes**:
  - Added diffusers_helper: utilities for improving model memory management
  - Recommend backups as needed since it replaces original tool sources
- **Unification to HunyuanVideo Format**:
  - Improved compatibility by standardizing LoRA format to HunyuanVideo format

### 2025-04-29: Version 1.8.1
- **Multilingual Support (i18n) Implementation**:
  - Support for three languages: Japanese, English, and Traditional Chinese
  - Added language-specific execution files (`run_endframe_ichi_en.bat`, `run_endframe_ichi_zh-tw.bat`)
  - Multilingual UI text, log messages, and error messages
  - Extensible translation system based on JSON files
  - Note: Current translations primarily cover v1.7.1 features; v1.8 features will be added later

### 2025-04-28: Version 1.8
- **Added Tensor Data Concatenation Feature**:
  - Added feature to save video latent representations in .safetensors format
  - Added feature to combine saved tensor data with newly generated videos
  - Implemented efficient processing of large tensor data using chunk processing
  - Improved stability through enhanced memory efficiency and device/type compatibility
  - Added automatic deletion of intermediate files and feedback display

### 2025-04-28: Version 1.7.1
- **Internal Calculation Optimization**:
  - Improved video generation stability with more accurate section count calculation in 0.5-second mode
  - Changed frame calculation parameter (latent_window_size) from 5 to 4.5
- **UI Improvements**:
  - Simplified video mode names: removed parenthetical notation (e.g., "10(5x2) seconds" → "10 seconds")
  - Reorganized layout: entire UI has been reorganized for improved usability

### 2025-04-28: Version 1.7.0
- **Greatly Improved Keyframe Image Copy Functionality**:
  - Introduced visual distinction with red/blue frames
  - **Red frames (Section 0)** ⇒ Automatically copies to all even-numbered sections (0,2,4,6...)
  - **Blue frames (Section 1)** ⇒ Automatically copies to all odd-numbered sections (1,3,5,7...)
  - Improved dynamic section count calculation accuracy for precise copy range adjustment based on selected video length
  - Users can toggle the copy function on/off as needed
- **Improved Copy Processing Safety**:
  - Enhanced section boundary checking to prevent copying to invalid positions
  - Easier tracking of copy operations through detailed log output

### 2025-04-27: Version 1.6.2
- **Added MP4 Compression Settings**: ※Merged from original project
  - Ability to adjust the balance between video file size and quality
  - Configurable in the range of 0-100 (0=uncompressed, 16=default, higher values=higher compression/lower quality)
  - Provides optimal settings to address black screen issues
- **Improved Code Quality**:
  - Separated UI style definitions into `ui_styles.py` for improved maintainability and readability
  - Enhanced UI consistency through centralized CSS style management
- **Fine-tuned Frame Size Calculation**:
  - Improved calculation precision for 0.5-second mode (using `latent_window_size=4.5`)
  - Enhanced section count and video length calculation accuracy for more stable video generation

### 2025-04-26: Version 1.6.1
- **Expanded Short Video Modes**:
  - 2-second mode: 60 frames (2 seconds × 30FPS), 2 sections, no copying needed
  - 3-second mode: 90 frames (3 seconds × 30FPS), 3 sections, keyframe 0→1 copying
  - 4-second mode: 120 frames (4 seconds × 30FPS), 4 sections, keyframe 0→1,2 copying
- **Reverting to v1.5.1 Basic Structure**:
  - Maintained original mode name notation (with parentheses)
  - Restored keyframe guide HTML
  - Maintained original function structure and processing methods

### 2025-04-25: Version 1.5.1
- **Added "1 Second" Mode for Short Video Generation**:
  - Support for 1 second (approximately 30 frames @ 30fps)
- **Changed Default Video Length**:
  - Changed default value from "6 seconds" to "1 second"
- **Optimized Input Image Copy Behavior**:
  - Stopped copying from input image in normal mode
  - Enabled copying to Last only in loop mode
- **Changed Default Value for Keyframe Auto-Copy Function**:
  - Set to off by default for more fine-grained control
  - Can be turned on when needed for automatic copying
- **Improved Image Processing Stability**:
  - Enhanced image retrieval when pressing the Start button
  - Added explicit clearing of preview images

### 2025-04-24: Version 1.5.0
- **Added Frame Size Setting**:
  - Ability to switch between 0.5-second and 1-second modes
  - Dynamic adjustment of latent_window_size based on frame size

### 2025-04-24: Version 1.4.0
- **Added All-Padding Function**:
  - Use the same padding value for all sections
  - Smaller values result in more intense movement per session

### 2025-04-24: Version 1.3.2
- **Unified LoRA Application Function**:
  - Unified the method to directly apply LoRA in low-VRAM mode (DynamicSwap) similar to high-VRAM mode
  - Eliminated hook method in favor of a more stable direct application method
  - Maintained interface for compatibility while improving internal implementation
- **Enhanced Debugging and Verification Functions**:
  - Added a dedicated utility to check LoRA application status (lora_check_helper.py)
  - Provided detailed log output and debugging information

### 2025-04-24: Version 1.3.1
- **Code Base Refactoring**: Reorganized code into multiple modules for improved maintainability and extensibility
  - `eichi_utils`: Manages keyframe processing, settings management, preset management, and video mode settings
  - `lora_utils`: Consolidates LoRA-related functionality

### 2025-04-23: Version 1.3
- **[Under Investigation] Added Hunyuan LoRA Support**: Customize models to add unique expressions
- **[Experimental Implementation] Added EndFrame Influence Setting**: Ability to adjust the influence of the final frame in the range of 0.01-1.00 (based on Nirvash's insights)

### 2025-04-23: Version 1.2
- **Added "20(4x5) Second" Mode**: Support for long video generation with 5-part structure
- **Added Section-Specific Prompt Feature**: Ability to set individual prompts for each section [Experimental Implementation]
- **Enhanced Output Folder Management**: Support for specifying output destination folder and OS-independent opening methods
- **Improved Settings File Management**: Persistent settings using JSON-based configuration files
- **Strengthened Cross-Platform Support**: Improved operation in non-Windows environments

### 2025-04-22: Version 1.1
- **Added "16(4x4) Second" Mode**: Support for long video generation with 4-part structure
- **Improved Progress Display During Generation**: Section information (e.g., "Section: 3/15") is now displayed, making it easier to track progress, especially for long video generation
- **Structured Configuration Files**: Video mode settings are separated into a separate file, improving extensibility

### 2025-04-21: Initial Release
- Optimized prompt management functionality
- Added keyframe guide functionality

## 简体中文

### 2025-05-15: 版本1.9.3
- **提示词队列和图像队列**：
  - 从文本文件加载多个提示词并按顺序处理
  - 自动按顺序处理指定文件夹中的图像
  - 启用队列时完成后自动处理下一项
- **FP8优化改进**：
  - 不使用LoRA时显著减少内存使用
  - 提高处理速度（需要PyTorch 2.1或更高版本）
  - 正常使用时默认启用
- **LoRA预设管理器**：
  - 保存和管理最多5个预设
  - 从UI轻松切换预设
  - 批量保存LoRA文件、强度和其他设置
- **UI组织和多语言支持**：
  - 将相关功能分组以提高可用性
  - 为每个功能添加说明以便更好理解
  - 添加俄语UI，支持4种语言（日语、英语、中文、俄语）
- **视频生成时长扩展**：
  - 在eichi（标准版）的1秒模式（33帧）下添加30秒、40秒选项
  - F1添加30秒、40秒、60秒、120秒选项
  - 支持更多样的视频长度，实现长视频生成
- **分段管理增强**：
  - 导入/导出的路径规范化
  - 改善不同环境之间的分段信息兼容性
- **kisekaeichi功能集成到oichi**：
  - 集成由furusu设计、Kohya实现的基于参考图像的视频生成技术
  - 通过目标索引和历史索引进行精确控制
  - 实现了仅更改特定区域的遮罩功能
- **错误修复和优化**：
  - 修复提示词添加和删除问题
  - 避免与asyncio相关的Windows错误
  - 添加F1模型的高效下载功能

### 2025-05-11: 版本1.9.2
- **"FramePack-oichi"新功能添加**:
  - 新功能可从单张输入图像预测生成下一张未来帧图像
  - 添加专用启动脚本（`run_oneframe_ichi.bat`等）
  - 与常规视频生成相比，更轻便简单地检查下一帧
  - 完整的多语言支持（日语、英语、中文）
- **帧图像保存功能**:
  - 添加保存所有生成帧图像的选项
  - 可选择保存所有段落的所有帧或仅最终段落的帧
  - 更容易可视化视频中间过程并作为素材资源使用
- **分段信息批量管理功能增强**:
  - 添加通过ZIP文件批量下载分段信息的功能
  - 高效管理和备份多个项目
  - 支持批量上传开始图像、结束图像、分段信息（提示词、图像）以及内容更改后的重新下载
- **LoRA功能增强**:
  - 支持同时使用三个LoRA
  - 默认从目录中选择，能够选择存储在/webui/lora文件夹中的LoRA
  - 所有模式（标准版、F1版、oneframe版）都支持LoRA功能增强
- **VAE缓存功能**:
  - 通过逐帧VAE解码提高处理速度 - 基于furusu研究的实现 [详情1](https://note.com/gcem156/n/nb93535d80c82) [详情2](https://github.com/laksjdjf/FramePack)
  - 灵活调整内存使用量和处理速度之间的平衡
  - 从设置屏幕轻松切换开/关
  - 默认为关闭，因此没有影响
  - 利用帧之间的独立性实现计算缓存，实现高达30%的速度提升

### 2025-05-04: 版本1.9.1
- **F1模型添加**:
  - 引入支持正向生成的新模型"FramePack_F1_I2V_HY_20250503"
  - 添加专用启动脚本（`run_endframe_ichi_f1.bat`等）
  - 在F1版本中移除了部分（关键帧图像）和Final（结束帧）功能
  - 修订了Image影响度的计算和设置范围（100.0%～102.0%）
  - 实现了简单操作和动态动作生成
- **内存管理增强**:
  - 通过transformer_manager.py和text_encoder_manager.py实现高效模型管理
  - 通过FP8优化大幅减少VRAM使用量
  - 支持RTX 40系列GPU的scaled_mm优化
- **Docker支持增强**:
  - 容器化环境中的简便设置
  - 兼容CUDA 12.6和PyTorch 2.6.0
  - 多语言容器支持（日语、英语、中文）
- **PNG元数据功能**:
  - 在生成的图像中自动嵌入提示词、种子和部分信息
  - 从保存的图像中检索设置
  - 与SD系工具兼容的元数据格式
- **最佳分辨率桶系统**:
  - 根据宽高比自动选择最佳分辨率
  - 自动调整为安全标准分辨率（512、640、768）
  - 各种输出尺寸的一致质量
- **剪贴板支持增强**:
  - 标准版中Images和Final帧以及F1版中Images的剪贴板支持
  - 与外部工具的增强集成

### 2025-04-30: 版本1.9
- **实施kohya-ss/FramePack-LoRAReady**:
  - 在Kohya Tech许可下，大幅提高LoRA功能性能和稳定性
  - 统一了高VRAM和低VRAM模式的LoRA应用方法
  - 通过使用通用load_and_apply_lora函数减少了代码复杂性
- **引入FP8优化**:
  - 通过8位浮点量化优化内存使用和处理速度
  - 默认建议关闭（在某些环境中可能会导致警告或错误）
  - 支持RTX 40系列GPU的加速选项
- **目录结构变更**:
  - 添加了diffusers_helper：用于改善模型内存管理的实用程序
  - 建议在需要时进行备份，因为它会替换原始工具源
- **统一为HunyuanVideo格式**:
  - 通过将LoRA格式标准化为HunyuanVideo格式来提高兼容性

### 2025-04-29: 版本1.8.1
- **多语言支持（i18n）实施**:
  - 支持三种语言：日语、英语和繁体中文
  - 添加了特定语言的执行文件（`run_endframe_ichi_en.bat`、`run_endframe_ichi_zh-tw.bat`）
  - 多语言UI文本、日志消息和错误消息
  - 基于JSON文件的可扩展翻译系统
  - 注意：当前翻译主要涵盖v1.7.1功能；v1.8功能将在后续添加

### 2025-04-28: 版本1.8
- **添加张量数据连接功能**:
  - 添加以.safetensors格式保存视频潜在表示的功能
  - 添加将保存的张量数据与新生成的视频结合的功能
  - 使用块处理实现大型张量数据的高效处理
  - 通过增强内存效率和设备/类型兼容性提高稳定性
  - 添加中间文件的自动删除和反馈显示

### 2025-04-28: 版本1.7.1
- **内部计算优化**:
  - 通过更准确的0.5秒模式部分计数计算提高视频生成稳定性
  - 将帧计算参数（latent_window_size）从5更改为4.5
- **UI改进**:
  - 简化视频模式名称：删除括号表示法（例如："10(5x2)秒" → "10秒"）
  - 重组布局：整个UI已重组以提高可用性

### 2025-04-28: 版本1.7.0
- **大幅改进关键帧图像复制功能**:
  - 引入红色/蓝色框架的视觉区分
  - **红色框架（部分0）** ⇒ 自动复制到所有偶数部分(0,2,4,6...)
  - **蓝色框架（部分1）** ⇒ 自动复制到所有奇数部分(1,3,5,7...)
  - 改进动态部分计数计算精度，根据所选视频长度精确调整复制范围
  - 用户可以根据需要切换复制功能的开关
- **改进复制处理安全性**:
  - 增强部分边界检查以防止复制到无效位置
  - 通过详细的日志输出更容易跟踪复制操作

### 2025-04-27: 版本1.6.2
- **添加MP4压缩设置**: ※从原始项目合并
  - 能够调整视频文件大小和质量之间的平衡
  - 可在0-100范围内配置（0=未压缩，16=默认，更高值=更高压缩/更低质量）
  - 提供最佳设置以解决黑屏问题
- **改进代码质量**:
  - 将UI样式定义分离到`ui_styles.py`以提高可维护性和可读性
  - 通过集中化CSS样式管理增强UI一致性
- **微调帧大小计算**:
  - 改进0.5秒模式的计算精度（使用`latent_window_size=4.5`）
  - 增强部分计数和视频长度计算精度以获得更稳定的视频生成

### 2025-04-26: 版本1.6.1
- **扩展短视频模式**:
  - 2秒模式：60帧（2秒×30FPS），2个部分，无需复制
  - 3秒模式：90帧（3秒×30FPS），3个部分，关键帧0→1复制
  - 4秒模式：120帧（4秒×30FPS），4个部分，关键帧0→1,2复制
- **恢复到v1.5.1基本结构**:
  - 维持原始模式名称表示法（带括号）
  - 恢复关键帧指南HTML
  - 维持原始函数结构和处理方法

### 2025-04-25: 版本1.5.1
- **为短视频生成添加"1秒"模式**:
  - 支持1秒（约30帧@30fps）
- **更改默认视频长度**:
  - 将默认值从"6秒"更改为"1秒"
- **优化输入图像复制行为**:
  - 在正常模式下停止从输入图像复制
  - 仅在循环模式下启用复制到Last
- **更改关键帧自动复制功能的默认值**:
  - 默认设置为关闭以获得更精细的控制
  - 需要时可以打开以进行自动复制
- **改进图像处理稳定性**:
  - 按下开始按钮时增强图像检索
  - 添加明确清除预览图像

### 2025-04-24: 版本1.5.0
- **添加帧大小设置**:
  - 能够在0.5秒和1秒模式之间切换
  - 基于帧大小动态调整latent_window_size

### 2025-04-24: 版本1.4.0
- **添加全填充功能**:
  - 对所有部分使用相同的填充值
  - 较小的值会导致每个会话中的移动更加剧烈

### 2025-04-24: 版本1.3.2
- **统一LoRA应用功能**:
  - 统一在低VRAM模式（DynamicSwap）中直接应用LoRA的方法，类似于高VRAM模式
  - 取消钩子方法，支持更稳定的直接应用方法
  - 保持接口兼容性同时改进内部实现
- **增强调试和验证功能**:
  - 添加专用实用程序来检查LoRA应用状态（lora_check_helper.py）
  - 提供详细的日志输出和调试信息

### 2025-04-24: 版本1.3.1
- **代码库重构**: 将代码重组为多个模块以提高可维护性和可扩展性
  - `eichi_utils`：管理关键帧处理、设置管理、预设管理和视频模式设置
  - `lora_utils`：整合LoRA相关功能

### 2025-04-23: 版本1.3
- **[调查中] 添加Hunyuan LoRA支持**: 自定义模型以添加独特表达
- **[实验性实现] 添加EndFrame影响度设置**: 能够在0.01-1.00范围内调整最终帧的影响（基于Nirvash的见解）

### 2025-04-23: 版本1.2
- **添加"20(4x5)秒"模式**: 支持具有5部分结构的长视频生成
- **添加部分特定提示功能**: 能够为每个部分设置单独的提示 [实验性实现]
- **增强输出文件夹管理**: 支持指定输出目标文件夹和与操作系统无关的打开方法
- **改进设置文件管理**: 使用基于JSON的配置文件持久化设置
- **加强跨平台支持**: 改进在非Windows环境中的操作

### 2025-04-22: 版本1.1
- **添加"16(4x4)秒"模式**: 支持具有4部分结构的长视频生成
- **改进生成期间的进度显示**: 现在显示部分信息（例如"部分：3/15"），使得跟踪进度更容易，特别是对于长视频生成
- **结构化配置文件**: 视频模式设置分离到单独的文件中，提高了可扩展性

### 2025-04-21: 初始发布
- 优化提示管理功能
- 添加关键帧指南功能

## Русский

### 2025-05-15: Версия 1.9.3
- **Очередь промптов и очередь изображений**:
  - Загрузка нескольких промптов из текстового файла и последовательная обработка
  - Автоматическая последовательная обработка изображений в указанной папке
  - Автоматическая обработка следующего элемента после завершения при включенной очереди
- **Улучшения оптимизации FP8**:
  - Значительное снижение использования памяти при неиспользовании LoRA
  - Повышение скорости обработки (требуется PyTorch 2.1 или выше)
  - Включено по умолчанию для обычного использования
- **Менеджер пресетов LoRA**:
  - Сохранение и управление до 5 пресетов
  - Легкое переключение пресетов из интерфейса
  - Пакетное сохранение файлов LoRA, силы и других настроек
- **Организация интерфейса и многоязычная поддержка**:
  - Группировка связанных функций для улучшения удобства использования
  - Добавлены объяснения для каждой функции для лучшего понимания
  - Добавлен русский интерфейс для поддержки 4 языков (японский, английский, китайский, русский)
- **Расширение продолжительности генерации видео**:
  - Добавлены варианты 30с, 40с для eichi (стандартная версия) в режиме 1с (33 кадра)
  - Добавлены варианты 30с, 40с, 60с, 120с для F1
  - Поддержка более разнообразной длины видео, включая генерацию длинных видео
- **Улучшение управления секциями**:
  - Нормализация путей для импорта/экспорта
  - Улучшенная совместимость информации о секциях между различными средами
- **Интеграция функции kisekaeichi в oichi**:
  - Интеграция технологии генерации видео на основе эталонного изображения, разработанной furusu и реализованной Kohya
  - Точный контроль через целевой индекс и индекс истории
  - Реализована функция маски для изменения только определенных областей
- **Исправления ошибок и оптимизация**:
  - Исправлены проблемы с добавлением и удалением промптов
  - Избежание ошибок Windows, связанных с asyncio
  - Добавлена эффективная функция загрузки для модели F1

### 2025-05-11: Версия 1.9.2
- **Добавление новой функции «FramePack-oichi»**:
  - Новая функция для предсказания и генерации следующего кадра из одного входного изображения
  - Добавлены специальные скрипты запуска (`run_oneframe_ichi.bat` и другие)
  - Более легкий и удобный способ проверки следующего кадра по сравнению с обычной генерацией видео
  - Полная многоязычная поддержка (японский, английский, китайский, русский)
- **Функция сохранения кадров**:
  - Добавлена опция сохранения всех сгенерированных кадров
  - Выбор между сохранением всех кадров из всех секций или только из последней секции
  - Облегчена визуализация промежуточного процесса видео и использование в качестве ресурсов
- **Улучшение пакетного управления информацией о секциях**:
  - Добавлена функция пакетной загрузки информации о секциях через ZIP-файл
  - Эффективное управление и резервное копирование нескольких проектов
  - Поддержка массовой загрузки начальных изображений, конечных изображений, информации о секциях (промпты, изображения) и повторной загрузки после изменения содержимого
- **Улучшение функций LoRA**:
  - Поддержка одновременного использования трех LoRA
  - Выбор по умолчанию из каталога, возможность выбора LoRA из папки /webui/lora
  - Улучшение функции LoRA поддерживается во всех режимах (стандартная версия, версия F1, версия oneframe)
- **Функция кэширования VAE**:
  - Повышение скорости обработки за счет покадрового декодирования VAE - На основе исследования furusu [Подробности1](https://note.com/gcem156/n/nb93535d80c82) [Подробности2](https://github.com/laksjdjf/FramePack)
  - Гибкая настройка баланса между использованием памяти и скоростью обработки
  - Легкое включение/выключение с экрана настроек
  - По умолчанию выключено, поэтому без влияния
  - Достигает ускорения до 30% за счет вычислительного кэширования с использованием независимости между кадрами

### 2025-05-04: Версия 1.9.1
- **Добавлена модель F1**:
  - Внедрение новой модели «FramePack_F1_I2V_HY_20250503» с поддержкой прямой генерации
  - Добавлены специальные скрипты запуска (`run_endframe_ichi_f1.bat` и другие)
  - Удалены функции секций (ключевые кадры) и Final (конечный кадр) в версии F1
  - Пересмотрен расчет и диапазон настройки влияния изображения (от 100.0% до 102.0%)
  - Достигнута простота управления и генерация динамичного движения
- **Улучшение управления памятью**:
  - Эффективное управление моделями с помощью transformer_manager.py и text_encoder_manager.py
  - Значительное сокращение использования VRAM с оптимизацией FP8
  - Поддержка оптимизации scaled_mm для GPU RTX 40 серии
- **Улучшение поддержки Docker**:
  - Простая настройка в контейнеризованных средах
  - Совместимость с CUDA 12.6 и PyTorch 2.6.0
  - Поддержка многоязычных контейнеров (японский, английский, китайский, русский)
- **Функция метаданных PNG**:
  - Автоматическое внедрение промптов, сидов и информации о секциях в сгенерированные изображения
  - Извлечение настроек из сохраненных изображений
  - Совместимый формат метаданных с инструментами на основе SD
- **Система оптимальных бакетов разрешения**:
  - Автоматический выбор оптимального разрешения на основе соотношения сторон
  - Автоматическая настройка на безопасные стандартные разрешения (512, 640, 768)
  - Последовательное качество при различных размерах выходных данных
- **Улучшение поддержки буфера обмена**:
  - Поддержка буфера обмена для изображений и конечных кадров в стандартной версии и изображений в версии F1
  - Улучшенная интеграция с внешними инструментами

### 2025-04-30: Версия 1.9
- **Внедрение kohya-ss/FramePack-LoRAReady**:
  - Значительно улучшена производительность и стабильность функции LoRA с разрешения Kohya Tech
  - Унифицирован метод применения LoRA для режимов с высоким и низким VRAM
  - Снижена сложность кода за счет использования общей функции load_and_apply_lora
- **Введение оптимизации FP8**:
  - Оптимизация использования памяти и скорости обработки с помощью 8-битной квантизации с плавающей точкой
  - Рекомендуется быть выключенным по умолчанию (может вызывать предупреждения или ошибки в некоторых средах)
  - Поддержка опций ускорения на GPU RTX 40 серии
- **Изменения структуры каталога**:
  - Добавлен diffusers_helper: утилиты для улучшения управления памятью модели
  - Рекомендуется делать резервные копии при необходимости, так как заменяются исходные исходники инструмента
- **Унификация формата HunyuanVideo**:
  - Улучшена совместимость за счет стандартизации формата LoRA до формата HunyuanVideo

### 2025-04-29: Версия 1.8.1
- **Реализация многоязычной поддержки (i18n)**:
  - Поддержка трех языков: японский, английский и традиционный китайский
  - Добавлены исполняемые файлы для конкретных языков (`run_endframe_ichi_en.bat`, `run_endframe_ichi_zh-tw.bat`)
  - Многоязычный текст UI, сообщения журнала и сообщения об ошибках
  - Расширяемая система перевода на основе JSON-файлов
  - Примечание: текущие переводы в основном охватывают функции v1.7.1; функции v1.8 будут добавлены позже

### 2025-04-28: Версия 1.8
- **Добавлена функция объединения тензорных данных**:
  - Добавлена функция сохранения скрытых представлений видео в формате .safetensors
  - Добавлена функция объединения сохраненных тензорных данных с вновь сгенерированными видео
  - Реализована эффективная обработка больших тензорных данных с использованием чанковой обработки
  - Улучшена стабильность за счет повышения эффективности использования памяти и совместимости устройств/типов
  - Добавлено автоматическое удаление промежуточных файлов и отображение обратной связи

### 2025-04-28: Версия 1.7.1
- **Оптимизация внутренних расчетов**:
  - Улучшена стабильность генерации видео благодаря более точному расчету количества секций в режиме 0.5 секунды
  - Изменен параметр расчета кадров (latent_window_size) с 5 на 4.5
- **Улучшения UI**:
  - Упрощены названия режимов видео: удалено обозначение в скобках (например, «10(5x2) секунд» → «10 секунд»)
  - Реорганизованный макет: весь UI был реорганизован для улучшения удобства использования

### 2025-04-28: Версия 1.7.0
- **Значительно улучшена функциональность копирования изображений ключевых кадров**:
  - Введено визуальное различие с красными/синими рамками
  - **Красные рамки (Секция 0)** ⇒ Автоматически копируются во все секции с четными номерами (0,2,4,6...)
  - **Синие рамки (Секция 1)** ⇒ Автоматически копируются во все секции с нечетными номерами (1,3,5,7...)
  - Улучшена точность расчета динамического количества секций для точной настройки диапазона копирования на основе выбранной длины видео
  - Пользователи могут включать/выключать функцию копирования по мере необходимости
- **Улучшена безопасность процесса копирования**:
  - Усилена проверка границ секций для предотвращения копирования в недопустимые позиции
  - Более легкое отслеживание операций копирования через подробный журнал вывода

### 2025-04-27: Версия 1.6.2
- **Добавлены настройки сжатия MP4**: ※Объединено из оригинального проекта
  - Возможность регулировки баланса между размером видеофайла и качеством
  - Настраиваемый в диапазоне 0-100 (0=без сжатия, 16=по умолчанию, более высокие значения=более высокое сжатие/низкое качество)
  - Предоставляет оптимальные настройки для устранения проблем с черным экраном
- **Улучшено качество кода**:
  - Отделены определения стилей UI в `ui_styles.py` для улучшения поддерживаемости и читаемости
  - Улучшена согласованность UI через централизованное управление стилями CSS
- **Точная настройка расчета размера кадра**:
  - Улучшена точность расчетов для режима 0.5 секунды (с использованием `latent_window_size=4.5`)
  - Повышена точность расчета количества секций и длины видео для более стабильной генерации видео

### 2025-04-26: Версия 1.6.1
- **Расширены режимы коротких видео**:
  - Режим 2 секунды: 60 кадров (2 секунды × 30FPS), 2 секции, копирование не требуется
  - Режим 3 секунды: 90 кадров (3 секунды × 30FPS), 3 секции, копирование ключевого кадра 0→1
  - Режим 4 секунды: 120 кадров (4 секунды × 30FPS), 4 секции, копирование ключевого кадра 0→1,2
- **Возврат к базовой структуре v1.5.1**:
  - Сохранено исходное обозначение режима (со скобками)
  - Восстановлен HTML-руководство по ключевым кадрам
  - Сохранены исходная структура функций и методы обработки

### 2025-04-25: Версия 1.5.1
- **Добавлен режим «1 секунда» для генерации коротких видео**:
  - Поддержка 1 секунды (примерно 30 кадров @ 30fps)
- **Изменена длина видео по умолчанию**:
  - Изменено значение по умолчанию с «6 секунд» на «1 секунду»
- **Оптимизировано поведение копирования входного изображения**:
  - Прекращено копирование из входного изображения в обычном режиме
  - Разрешено копирование только в Last в режиме цикла
- **Изменено значение по умолчанию для функции автокопирования ключевого кадра**:
  - Установлено выключенным по умолчанию для более точного контроля
  - Можно включить при необходимости для автоматического копирования
- **Улучшена стабильность обработки изображений**:
  - Улучшено получение изображений при нажатии кнопки Start
  - Добавлена явная очистка предварительных изображений

### 2025-04-24: Версия 1.5.0
- **Добавлена настройка размера кадра**:
  - Возможность переключения между режимами 0.5 секунды и 1 секунды
  - Динамическая настройка latent_window_size в зависимости от размера кадра

### 2025-04-24: Версия 1.4.0
- **Добавлена функция общего отступа**:
  - Использование одинакового значения отступа для всех секций
  - Меньшие значения приводят к более интенсивному движению за сеанс

### 2025-04-24: Версия 1.3.2
- **Унифицирована функция применения LoRA**:
  - Унифицирован метод прямого применения LoRA в режиме низкого VRAM (DynamicSwap), аналогично режиму высокого VRAM
  - Устранен метод hook в пользу более стабильного метода прямого применения
  - Сохранен интерфейс для совместимости при улучшении внутренней реализации
- **Улучшены функции отладки и проверки**:
  - Добавлена специальная утилита для проверки статуса применения LoRA (lora_check_helper.py)
  - Предоставлен подробный вывод журнала и отладочная информация

### 2025-04-24: Версия 1.3.1
- **Рефакторинг кодовой базы**: Реорганизация кода в несколько модулей для улучшения поддерживаемости и расширяемости
  - `eichi_utils`: Управление обработкой ключевых кадров, управление настройками, управление пресетами и настройки режима видео
  - `lora_utils`: Консолидация функциональности, связанной с LoRA

### 2025-04-23: Версия 1.3
- **[В процессе исследования] Добавлена поддержка Hunyuan LoRA**: Настройка моделей для добавления уникальных выражений
- **[Экспериментальная реализация] Добавлена настройка влияния EndFrame**: Возможность регулировки влияния конечного кадра в диапазоне 0.01-1.00 (на основе знаний Nirvash)

### 2025-04-23: Версия 1.2
- **Добавлен режим «20(4x5) секунд»**: Поддержка генерации длинных видео с 5-частной структурой
- **Добавлена функция специфичных для секций промптов**: Возможность установки индивидуальных промптов для каждой секции [Экспериментальная реализация]
- **Улучшено управление выходной папкой**: Поддержка указания выходной папки назначения и независимых от ОС методов открытия
- **Улучшено управление файлами настроек**: Постоянные настройки с использованием конфигурационных файлов на основе JSON
- **Усилена кроссплатформенная поддержка**: Улучшена работа в средах, отличных от Windows

### 2025-04-22: Версия 1.1
- **Добавлен режим «16(4x4) секунд»**: Поддержка генерации длинных видео с 4-частной структурой
- **Улучшено отображение прогресса во время генерации**: Теперь отображается информация о секциях (например, «Секция: 3/15»), что облегчает отслеживание прогресса, особенно для генерации длинных видео
- **Структурированные конфигурационные файлы**: Настройки режима видео разделены в отдельный файл, улучшая расширяемость

### 2025-04-21: Первоначальный выпуск
- Оптимизирована функциональность управления промптами
- Добавлена функциональность руководства по ключевым кадрам
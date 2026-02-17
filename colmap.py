import os
import subprocess
import shutil
import sys

# ================= 設定エリア =================
# 画像が入っているフォルダ
IMAGES_PATH = "images"
# データベースファイル名
DB_PATH = "database.db"
# Mapperの一時出力先（作業用）
SPARSE_TEMP_PATH = "sparse_temp"
# 最終出力先（3DGS学習用）
OUTPUT_PATH = "output_undistorted"

# COLMAPの実行ファイル名（パスが通っていない場合はフルパスで指定してください）
# Windows例: r"C:\COLMAP\colmap.bat"
COLMAP_BIN = "colmap" 
# ============================================

def run_command(cmd, description):
    """コマンド実行用ヘルパー関数"""
    print(f"\n>>> {description} を開始します...")
    try:
        subprocess.run(cmd, check=True)
        print(f">>> {description} 完了。")
    except subprocess.CalledProcessError as e:
        print(f"!!! エラー発生: {description} に失敗しました。")
        sys.exit(1)

def main():
    # 1. 特徴点抽出 (Feature Extractor)
    # DBが存在すれば、COLMAPは自動的に「まだ計算していない画像」だけを処理します。
    run_command([
        COLMAP_BIN, "feature_extractor",
        "--database_path", DB_PATH,
        "--image_path", IMAGES_PATH,
        "--ImageReader.camera_model", "SIMPLE_PINHOLE", # 必要に応じて変更
        "--ImageReader.single_camera", "1" # 全画像で同じカメラなら1
    ], "特徴点抽出 (Feature Extractor)")

    # 2. マッチング (Exhaustive Matcher)
    # 画像を追加した場合は、関係性を再構築するために実行が必要です。
    run_command([
        COLMAP_BIN, "exhaustive_matcher",
        "--database_path", DB_PATH
    ], "特徴点マッチング (Exhaustive Matcher)")

    # 3. 疎な再構成 (Mapper)
    # 前回の計算結果が残っていると邪魔になることがあるため、一時フォルダをクリア
    if os.path.exists(SPARSE_TEMP_PATH):
        shutil.rmtree(SPARSE_TEMP_PATH)
    os.makedirs(SPARSE_TEMP_PATH)

    run_command([
        COLMAP_BIN, "mapper",
        "--database_path", DB_PATH,
        "--image_path", IMAGES_PATH,
        "--output_path", SPARSE_TEMP_PATH
    ], "3次元再構成 (Mapper)")

    # 4. モデルの統合チェック
    # sparse_temp の中に "0" フォルダがあり、かつ "1" がないことを確認
    if not os.path.exists(os.path.join(SPARSE_TEMP_PATH, "0")):
        print("!!! 失敗: モデルが生成されませんでした。")
        sys.exit(1)
    
    if os.path.exists(os.path.join(SPARSE_TEMP_PATH, "1")):
        print("!!! 警告: モデルが複数に分かれています (sparse/0, sparse/1...)。")
        print("!!! 画像のオーバーラップが不足しています。Undistortは実行しません。")
        sys.exit(1)

    print(">>> 成功: 単一のモデル (sparse/0) が生成されました。")

    # 5. 画像の歪み補正と出力 (Image Undistorter)
    # ここで output_undistorted フォルダが作られます
    if os.path.exists(OUTPUT_PATH):
        print(f">>> 既存の {OUTPUT_PATH} を削除して再作成します...")
        shutil.rmtree(OUTPUT_PATH)
    
    run_command([
        COLMAP_BIN, "image_undistorter",
        "--image_path", IMAGES_PATH,
        "--input_path", os.path.join(SPARSE_TEMP_PATH, "0"),
        "--output_path", OUTPUT_PATH,
        "--output_type", "COLMAP"
    ], "歪み補正と出力 (Image Undistorter)")

    print("\n============================================")
    print(f"全工程が完了しました！")
    print(f"出力先: {os.path.abspath(OUTPUT_PATH)}")
    print("このフォルダを使って 3DGS の train.py を実行してください。")
    print("============================================")

if __name__ == "__main__":
    main()
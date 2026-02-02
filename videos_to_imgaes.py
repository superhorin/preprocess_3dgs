import cv2
import os
import glob
from pathlib import Path

# ==========================================
# 設定エリア
# ==========================================
INPUT_FOLDER = "videos"
OUTPUT_ROOT = "images"

# "SEPARATE" or "MERGE"
MODE = "MERGE"

# 間引き設定
SAVE_EVERY_N_FRAME = 1

# 【画質設定】
# "png" : 画質劣化ゼロ（最高画質）。ファイルサイズ大。3DGSに最適。
# "jpg" : 画質良。ファイルサイズ小。
IMAGE_FORMAT = "png"
# ==========================================

def process_videos():
    if not os.path.exists(INPUT_FOLDER):
        print(f"エラー: '{INPUT_FOLDER}' がありません。")
        os.makedirs(INPUT_FOLDER)
        return

    # 対応動画拡張子
    extensions = ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI"]
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    video_files.sort()

    if not video_files:
        print(f"'{INPUT_FOLDER}' に動画ファイルが見つかりません。")
        return

    print(f"計 {len(video_files)} 本の動画を処理します。形式: {IMAGE_FORMAT.upper()}")

    total_global_count = 0

    for video_path in video_files:
        video_name = Path(video_path).stem
        print(f"\n処理中: {os.path.basename(video_path)} ...")

        if MODE == "SEPARATE":
            save_dir = os.path.join(OUTPUT_ROOT, video_name)
        else:
            save_dir = OUTPUT_ROOT

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % SAVE_EVERY_N_FRAME == 0:
                if MODE == "SEPARATE":
                    filename = f"{saved_count + 1:05d}.{IMAGE_FORMAT}"
                else:
                    filename = f"{video_name}_{saved_count + 1:05d}.{IMAGE_FORMAT}"

                save_path = os.path.join(save_dir, filename)
                
                # 【ここが変更点】
                if IMAGE_FORMAT == "png":
                    # PNG保存（圧縮レベル0〜9を指定可能だが、デフォルトで画質は変わらない）
                    # 劣化なしで保存される
                    cv2.imwrite(save_path, frame)
                else:
                    # JPEG保存（最高画質）
                    cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                
                saved_count += 1
                total_global_count += 1

            frame_count += 1

        cap.release()
        print(f"  -> {saved_count} 枚保存しました。")

    print(f"\n完了しました。合計 {total_global_count} 枚。")
    print(f"保存先: {OUTPUT_ROOT}")

if __name__ == "__main__":
    process_videos()
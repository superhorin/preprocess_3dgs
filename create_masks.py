import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
from tqdm import tqdm

# ==========================================
# 設定エリア
# ==========================================
INPUT_FOLDER = "images"
OUTPUT_FOLDER = "masks"
MODEL_NAME = 'yolov8m-seg.pt'
TARGET_CLASSES = [0, 2, 3, 5, 7]
# ==========================================

def create_mask_for_image(model, img_path, output_folder):
    filename = os.path.basename(img_path)
    # 出力は常にpngにする
    # 拡張子を除いたファイル名を取得
    name_without_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, name_without_ext + ".png")

    img = cv2.imread(img_path)
    if img is None:
        print(f"スキップ: 画像が読み込めませんでした (非対応フォーマットの可能性): {filename}")
        return
    height, width = img.shape[:2]

    # 背景＝白(255)
    mask_img = np.full((height, width), 255, dtype=np.uint8)

    results = model.predict(
        source=img,
        classes=TARGET_CLASSES,
        conf=0.25,
        retina_masks=True,
        verbose=False
    )
    result = results[0]

    if result.masks is not None:
        contours = result.masks.xy
        for contour in contours:
            if len(contour) > 0:
                contour = contour.astype(np.int32)
                # 対象＝黒(0)
                cv2.fillPoly(mask_img, pts=[contour], color=(0))

    cv2.imwrite(output_path, mask_img)

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"出力フォルダを作成しました: {OUTPUT_FOLDER}")

    # 【修正点】大文字・小文字の両方の拡張子を探すように変更
    # HEICはOpenCVで読めないためここには含めません（変換してください）
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    
    for ext in extensions:
        # フォルダ内のパスを結合して検索
        files = glob.glob(os.path.join(INPUT_FOLDER, ext))
        image_files.extend(files)
    
    # 重複を除去（念のため）
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"エラー: {INPUT_FOLDER} に jpg, jpeg, png ファイルが見つかりません。")
        print("ヒント: HEICファイルがある場合は、jpgに変換してください。")
        return

    print(f"{len(image_files)} 枚の画像を処理します...")
    
    model = YOLO(MODEL_NAME)

    for img_path in tqdm(image_files):
        create_mask_for_image(model, img_path, OUTPUT_FOLDER)

    print(f"\n完了しました。マスク画像は {OUTPUT_FOLDER} に保存されました。")

if __name__ == "__main__":
    main()
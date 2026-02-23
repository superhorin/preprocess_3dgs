# python resize_images.py --image_dir my_3dgs_dataset/images

import os
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse

def resize_dataset(input_dir, factor):
    input_path = Path(input_dir)
    # 出力先フォルダ名 (例: images_4, images_8)
    output_path = input_path.parent / f"{input_path.name}_{factor}"
    
    print(f"\n{output_path.name} を作成中... (縮小率: 1/{factor})")
    
    # フォルダ内のすべてのjpg/pngを取得（サブフォルダも含む）
    image_files = list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.png"))
    
    if not image_files:
        print("画像が見つかりません。パスを確認してください。")
        return

    for img_path in tqdm(image_files):
        # サブフォルダを含む相対パスを取得 (例: pano_camera0/xxx.jpg)
        rel_path = img_path.relative_to(input_path)
        out_img_path = output_path / rel_path
        
        # 保存先のサブフォルダがなければ作成
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 画像の読み込みと縮小
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        new_h, new_w = h // factor, w // factor
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 保存
        cv2.imwrite(str(out_img_path), resized)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="my_3dgs_dataset/images", help="元のimagesフォルダのパス")
    args = parser.parse_args()
    
    # gsplatでよく使われる 1/4 と 1/8 サイズを作成
    resize_dataset(args.image_dir, factor=4)
    resize_dataset(args.image_dir, factor=8)
    
    print("\nすべての縮小処理が完了しました！")
import cv2
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm

def get_perspective_map(img_w, img_h, out_w, out_h, fov, yaw, pitch, roll):
    """
    Equirectangular画像からPerspective画像への変換マップを作成する関数
    一度計算すれば、同じ画角・方向なら使い回せるため高速です。
    """
    # カメラ内部パラメータ行列 (K)
    f = 0.5 * out_w / np.tan(0.5 * fov * np.pi / 180)
    cx = (out_w - 1) / 2
    cy = (out_h - 1) / 2
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ])
    K_inv = np.linalg.inv(K)

    # 回転行列 (R) - ヨー、ピッチ、ロールから計算
    # 座標系: X:右, Y:下, Z:奥
    
    # Radians
    y_rad = np.radians(yaw)
    p_rad = np.radians(pitch)
    r_rad = np.radians(roll)

    R_y = np.array([
        [np.cos(y_rad), 0, np.sin(y_rad)],
        [0, 1, 0],
        [-np.sin(y_rad), 0, np.cos(y_rad)]
    ])
    R_p = np.array([
        [1, 0, 0],
        [0, np.cos(p_rad), -np.sin(p_rad)],
        [0, np.sin(p_rad), np.cos(p_rad)]
    ])
    R_r = np.array([
        [np.cos(r_rad), -np.sin(r_rad), 0],
        [np.sin(r_rad), np.cos(r_rad), 0],
        [0, 0, 1]
    ])
    
    R = R_r @ R_p @ R_y

    # グリッド作成 (出力画像の各ピクセル座標)
    x_range = np.arange(out_w)
    y_range = np.arange(out_h)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    
    # 同次座標系へ (z=1)
    # 形状を (3, H*W) に変形
    ones = np.ones_like(x_grid)
    xyz_cam = np.stack([x_grid, y_grid, ones], axis=2).reshape(-1, 3).T
    
    # カメラ座標系から世界座標系（球の中心）へ逆投影
    xyz_world = R @ (K_inv @ xyz_cam)
    
    # 直交座標(x,y,z) -> 球面座標(theta, phi)
    x = xyz_world[0, :]
    y = xyz_world[1, :]
    z = xyz_world[2, :]
    
    # 経度 (phi) と 緯度 (theta)
    # theta: -pi/2 to pi/2 (緯度), phi: -pi to pi (経度)
    # ここではY軸が下向き、Z軸が正面の座標系を想定して調整
    phi = np.arctan2(x, z) 
    hyp = np.sqrt(x**2 + z**2)
    theta = np.arctan2(y, hyp)

    # Equirectangular画像上の座標 (u, v) に変換
    # img_w, img_h は入力画像のサイズ
    u = (phi + np.pi) / (2 * np.pi) * img_w
    v = (theta + np.pi / 2) / np.pi * img_h
    
    # cv2.remap用にfloat32型に変形
    map_x = u.reshape(out_h, out_w).astype(np.float32)
    map_y = v.reshape(out_h, out_w).astype(np.float32)

    return map_x, map_y

def main():
    parser = argparse.ArgumentParser(description="360度動画をCOLMAP用に分割・抽出するスクリプト")
    parser.add_argument("video_path", help="入力動画ファイルのパス")
    parser.add_argument("--output_dir", default="images", help="出力先ディレクトリ (デフォルト: images)")
    parser.add_argument("--interval", type=int, default=10, help="何フレームごとに抽出するか (デフォルト: 10)")
    parser.add_argument("--width", type=int, default=1024, help="出力画像の幅 (デフォルト: 1024)")
    parser.add_argument("--fov", type=int, default=90, help="出力画像の視野角 (デフォルト: 90度)")
    
    args = parser.parse_args()

    # 動画読み込み
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルが開けません: {args.video_path}")
        sys.exit(1)

    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"動画情報: {input_w}x{input_h}, 全{total_frames}フレーム")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 切り出す方向の設定 (Yaw, Pitch)
    # 前, 右, 後, 左, 上, 下
    directions = {
        "F": (0, 0),    # Front
        "R": (90, 0),   # Right
        "B": (180, 0),  # Back
        "L": (-90, 0),  # Left
        "U": (0, -90),  # Up
        "D": (0, 90)    # Down
    }

    # マップの事前計算 (これが高速化のキモです)
    print("変換マップを計算中...")
    maps = {}
    for key, (yaw, pitch) in directions.items():
        maps[key] = get_perspective_map(
            input_w, input_h, 
            args.width, args.width, # 正方形出力
            args.fov, 
            yaw, pitch, 0
        )

    print(f"抽出開始: {args.interval}フレームごとに保存します。")
    
    frame_idx = 0
    extracted_count = 0
    
    pbar = tqdm(total=total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 指定インターバルごとのみ処理
        if frame_idx % args.interval == 0:
            base_filename = f"{frame_idx:06d}"
            
            for key, (map_x, map_y) in maps.items():
                # 歪み補正 (切り出し)
                perspective_img = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
                
                # 保存
                out_path = os.path.join(args.output_dir, f"{base_filename}_{key}.jpg")
                cv2.imwrite(out_path, perspective_img)
            
            extracted_count += 1
        
        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f"\n完了しました！ 合計 {extracted_count * 6} 枚の画像を {args.output_dir} に保存しました。")
    print("COLMAPの feature_extractor を実行してください。")

if __name__ == "__main__":
    main()

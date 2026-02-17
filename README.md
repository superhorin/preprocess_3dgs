# preprocess

動画ファイルから画像（フレーム）を切り出し、物体検出・セグメンテーション学習用のマスク画像を生成するためのツールキットです。

## 主な機能
動画から画像への変換: videos フォルダ内の動画からフレームを抽出します。

自動マスク生成: Ultralytics (YOLO) 等を利用して、画像内のオブジェクトに対するマスク画像を生成します。

前処理: COLMAPで特徴点抽出とマッチング

## セットアップ
### 1. リポジトリのクローン
```
git clone https://github.com/superhorin/preprocess_3dgs
cd preprocess_3dgs
```
### 2. 仮想環境の作成（推奨）
Pythonの依存関係を分離するために仮想環境を作成します。
```
python3 -m venv venv
source venv/bin/activate  # Windowsの場合は venv\Scripts\activate
```
### 3. 依存ライブラリのインストール
PyTorchおよび画像処理に必要なライブラリをインストールします。
```
pip install torch torchvision torchaudio ultralytics opencv-python numpy tqdm
```

## 使い方
### ステップ1: 動画の準備
videos フォルダを作成し、処理したい動画ファイルを格納してください。
### ステップ2: 動画から画像を抽出
動画をフレーム単位の画像に変換します。
```
python video_to_images.py
```
### ステップ3: マスク画像の生成
抽出した画像から、学習用のマスク画像を生成します。
```
python create_masks.py
```
### ステップ4: 前処理
抽出した画像から、特徴点抽出とマッチング
```
python colmap.py
```
### ステップ0: 360 to images
360度映像を処理可能に変換
```
python 360_to_colmap.py input_video.mp4
```
```
python 360_to_colmap.py input_video.mp4 --interval 5 --width 512 --output_dir my_images
```

## ディレクトリ構造
```
 ├── videos/             # 元動画の格納場所
 ├── images/             # 抽出された画像（自動生成）
 ├── masks/              # 生成されたマスク画像（自動生成）
 ├── video_to_images.py  # 動画 -> 画像変換スクリプト
 ├── create_masks.py     # マスク生成スクリプト
 └── README.md           # 本ファイル
```
### 技術スタック
```
Python 3
PyTorch / Torchvision: 学習・推論プラットフォーム
Ultralytics (YOLO): セグメンテーション・物体検出
OpenCV: 画像処理
tqdm: プログレスバー表示
```

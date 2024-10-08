# -*- coding: utf-8 -*-
"""chocolate.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gYBYHPEPrPZ6ih9UcuwZubJQqEfOA-4t

## 「きのこの山」か「たけのこ里」かを判定する
RezNet-18をファインチューニングして「きのこの山」か「たけのこ里」かを判定するモデルを作成します

【注意事項】
* ファイルにchocolate.zipをアップロードしてから実行してください<br>
* ランタイムはGPUを使用してください
* このファイルをコピーして使ってください

### 1. zipファイルを読み込む
"""

# ! unzip /content/chocolate.zip

"""### 2.モデルを作成する"""

import warnings

warnings.filterwarnings('ignore')

# pandasをインポートする
import pandas as pd

# PyTorchと関連ライブラリをインポートする
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

# torchvisionと関連ライブラリをインポートする
from torchvision import datasets, transforms, models

# 画像の前処理を定義するためのトランスフォームを設定
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomGrayscale(p=0.1),  # 10%の確率でグレースケールに変換
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ImageFolderを使用してディレクトリから画像データセットを読み込み、定義した前処理を適用
full_dataset = datasets.ImageFolder('data/train', transform=transform)

# 全データセットを訓練データセット（80％）と検証データセットに（20％）分割
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

len(train_dataset), len(val_dataset)

# 訓練データローダーと検証データローダーを設定
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 事前に訓練されたResNet-18モデルをロードし、最後の全結合層を2クラス分類用に置き換える
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# num_features

# 損失関数とオプティマイザーを定義
criterion = nn.CrossEntropyLoss()  # クロスエントロピー損失を使用
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adamオプティマイザーを使用し、学習率を0.0001に設定

# 訓練と検証のループ
num_epochs = 15  # 訓練を行うエポック数を設定

for epoch in range(num_epochs):  # エポック数だけ訓練と検証のループを回す
    model.train()  # モデルを訓練モードに設定
    running_loss = 0.0  # 損失の合計を初期化
    train_correct = 0  # 訓練データの正解数
    train_total = 0  # 訓練データの総数

    # 訓練データのバッチごとに処理
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # オプティマイザーの勾配をリセット
        outputs = model(inputs)  # モデルに入力データを与えて予測
        loss = criterion(outputs, labels)  # 損失を計算
        loss.backward()  # バックプロパゲーションを行い勾配を計算
        optimizer.step()  # 勾配を使用してパラメータを更新
        running_loss += loss.item()  # このエポックでの損失を加算
        _, predicted = torch.max(outputs.data, 1)  # 最も確率の高いクラスを予測
        train_total += labels.size(0)  # バッチのデータ数を加算
        train_correct += (predicted == labels).sum().item()  # 正解数を更新

    train_accuracy = train_correct / train_total  # 訓練データの正解率を計算

    # モデルを評価モードに設定
    model.eval()
    val_correct = 0  # 検証データの正解数
    val_total = 0  # 検証データの総数

    # 検証データの処理
    with torch.no_grad():  # 勾配計算を無効化
        for inputs, labels in val_loader:
            outputs = model(inputs)  # モデルに入力データを与えて予測
            _, predicted = torch.max(outputs.data, 1)  # 最も確率の高いクラスを予測
            val_total += labels.size(0)  # バッチのデータ数を加算
            val_correct += (predicted == labels).sum().item()  # 正解数を更新

    val_accuracy = val_correct / val_total  # 検証データの正解率を計算

    # エポックごとの結果を表示
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, '
          f'Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')

"""### 3.テストデータで検証する"""

# テストデータセットのロード
test_dataset = datasets.ImageFolder('data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=18, shuffle=False)

# テストデータでの予測
model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())

# 予測結果をCSVファイルに出力
test_filenames = [x[0].split('/')[-1] for x in test_dataset.imgs]
submission = pd.DataFrame({'filename': test_filenames, 'class': predictions})
submission.to_csv('submission.csv', index=False)

# モデルを保存
# model_save_path = '/content/model.pth'
model_save_path = 'models/model.pth'
torch.save(model.state_dict(), model_save_path)

# きのこ度とたけのこ度を出力する

# テストデータセットのロード
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
test_dataset = datasets.ImageFolder('chocolate/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# モデルを評価モードに設定
model.eval()

# テストデータでの予測
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        # ソフトマックス関数を適用して確率を計算
        probabilities = F.softmax(outputs, dim=1)

        # 各画像のファイル名を取得
        filenames = [test_dataset.imgs[i][0] for i in range(inputs.size(0))]

        for filename, prob in zip(filenames, probabilities):
            kinoko_prob = prob[0].item() * 100  # きのこの確率
            takenoko_prob = prob[1].item() * 100  # たけのこの確率
            print(f"ファイル名: {filename}, きのこ度: {kinoko_prob:.2f}%, たけのこ度: {takenoko_prob:.2f}%")

"""### 4.任意の画像ファイルを判定する（おまけ✨）"""

from PIL import Image
# from google.colab import files
import io

# テストデータセットのロード
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
test_dataset = datasets.ImageFolder('chocolate/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()


def predict_image(image):
    image = Image.open(image).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        kinoko_prob = probabilities[0][0].item() * 100
        takenoko_prob = probabilities[0][1].item() * 100

        print(f"きのこ度: {kinoko_prob:.2f}%, たけのこ度: {takenoko_prob:.2f}%")


# 画像をアップロードして予測
# uploaded = files.upload()

# for fn in uploaded.keys():
#     predict_image(io.BytesIO(uploaded[fn]))

"""
きのこの山とたけのこの里を分類するための機械学習モデルを訓練し、テストするスクリプト。

このスクリプトは以下の主要な機能を提供します：
1. データの前処理と読み込み
2. モデルの作成と訓練
3. 訓練済みモデルの保存
4. テスト画像の予測
5. 予測結果のCSVファイル出力

使用方法:
1. 'data' ディレクトリに訓練データとテストデータを配置します。
2. このスクリプトを実行します: python script_name.py

注意:
- PyTorch、torchvision、Pillowがインストールされている必要があります。
- GPU使用時は、適宜コードを修正してGPUを使用するようにしてください。
"""

import os
import warnings
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image

warnings.filterwarnings('ignore')


def create_transforms():
    """
    画像の前処理用の変換を作成する。

    変換内容:
    1. リサイズ (256x256)
    2. 中央部分の切り取り (224x224)
    3. ランダムな色調整
    4. 10%の確率でグレースケールに変換
    5. テンソルに変換
    6. 正規化

    Returns:
        transforms.Compose: 一連の変換をまとめたオブジェクト
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_and_split_data(data_dir, transform):
    """
    指定されたディレクトリから画像データを読み込み、訓練用と検証用に分割する。

    Args:
        data_dir (str): 画像データが格納されているディレクトリのパス
        transform (transforms.Compose): 画像に適用する前処理

    Returns:
        tuple: (訓練用データセット, 検証用データセット)
    """
    # ImageFolderを使用してデータを読み込む
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    # データを8:2の割合で分割
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    return random_split(full_dataset, [train_size, val_size])


def create_data_loaders(train_dataset, val_dataset, batch_size=16):
    """
    訓練用と検証用のDataLoaderを作成する。

    Args:
        train_dataset (Dataset): 訓練用データセット
        val_dataset (Dataset): 検証用データセット
        batch_size (int): バッチサイズ（デフォルト: 16）

    Returns:
        tuple: (訓練用DataLoader, 検証用DataLoader)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def create_model():
    """
    事前学習済みのResNet-18モデルをロードし、最後の全結合層を2クラス分類用に変更する。

    Returns:
        nn.Module: 修正されたResNet-18モデル
    """
    # 事前学習済みのResNet-18をロード
    model = models.resnet18(pretrained=True)

    # 最後の全結合層を2クラス分類用に変更
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model


import torch
import torch.nn as nn
import torch.optim as optim


def train_epoch(model, train_loader, criterion, optimizer):
    """
    1エポックの訓練を行う

    Args:
        model (nn.Module): 訓練するモデル
        train_loader (DataLoader): 訓練データのDataLoader
        criterion: 損失関数
        optimizer: オプティマイザ

    Returns:
        float: 平均損失
        float: 訓練精度
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader):
    """
    モデルの検証を行う

    Args:
        model (nn.Module): 検証するモデル
        val_loader (DataLoader): 検証データのDataLoader

    Returns:
        float: 検証精度
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def train_model(model, train_loader, val_loader, num_epochs=15):
    """
    モデルを訓練し、各エポックごとの損失と精度を表示する

    Args:
        model (nn.Module): 訓練するモデル
        train_loader (DataLoader): 訓練データのDataLoader
        val_loader (DataLoader): 検証データのDataLoader
        num_epochs (int): 訓練エポック数（デフォルト: 15）
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer)
        val_accuracy = validate(model, val_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')


def save_model(model, save_path):
    """
    モデルを指定されたパスに保存する。

    Args:
        model (nn.Module): 保存するモデル
        save_path (str): モデルを保存するパス
    """
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def predict_image(model, image_path, transform):
    """
    単一の画像に対して予測を行う。

    Args:
        model (nn.Module): 予測に使用するモデル
        image_path (str): 予測する画像のパス
        transform (transforms.Compose): 画像に適用する前処理

    Returns:
        tuple: (きのこの確率, たけのこの確率)
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # バッチ次元を追加

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        kinoko_prob = probabilities[0][0].item() * 100
        takenoko_prob = probabilities[0][1].item() * 100

    return kinoko_prob, takenoko_prob


def predict_test_images(model, test_dir, transform):
    """
    テストディレクトリ内のすべての画像に対して予測を行い、結果を表示する。

    Args:
        model (nn.Module): 予測に使用するモデル
        test_dir (str): テスト画像が格納されているディレクトリのパス
        transform (transforms.Compose): 画像に適用する前処理
    """
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(test_dir, filename)
            kinoko_prob, takenoko_prob = predict_image(model, image_path, transform)
            print(f"ファイル名: {filename}")
            print(f"きのこ度: {kinoko_prob:.2f}%, たけのこ度: {takenoko_prob:.2f}%")
            print()


def create_submission(model, test_dir, transform):
    """
    テストデータに対する予測結果をCSVファイルとして出力する。

    Args:
        model (nn.Module): 予測に使用するモデル
        test_dir (str): テスト画像が格納されているディレクトリのパス
        transform (transforms.Compose): 画像に適用する前処理
    """
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=18, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())

    test_filenames = [os.path.basename(x[0]) for x in test_dataset.imgs]
    submission = pd.DataFrame({'filename': test_filenames, 'class': predictions})
    submission.to_csv('results/submission.csv', index=False)
    print("Submission file created: submission.csv")


def main():
    """
    メイン関数。スクリプトの全体の流れを制御する。
    """
    # 前処理の設定
    transform = create_transforms()

    # データの読み込みと分割
    train_dataset, val_dataset = load_and_split_data('data/train', transform)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)

    # モデルの作成と訓練
    model = create_model()
    train_model(model, train_loader, val_loader)

    # モデルの保存
    save_model(model, 'models/model.pth')

    # テスト画像の予測
    predict_test_images(model, 'data/test', transform)

    # 提出用CSVファイルの作成
    create_submission(model, 'data/test', transform)


if __name__ == "__main__":
    main()

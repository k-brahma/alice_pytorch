"""
きのこの山とたけのこの里を分類するための機械学習モデルを訓練し、テストするスクリプト。 temperature scalingを追加

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
import copy
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

warnings.filterwarnings('ignore')


class TemperatureScaling(nn.Module):
    def __init__(self, model):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return logits / self.temperature


def create_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_and_split_data(data_dir, transform):
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    return random_split(full_dataset, [train_size, val_size, test_size])


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def create_model():
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model


def train_model(model, train_loader, val_loader, num_epochs=15, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    no_improve_epochs = 0

    for epoch in range(num_epochs):
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

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        val_accuracy = validate(model, val_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_weights = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f'Early stopping triggered after epoch {epoch + 1}')
            break

    model.load_state_dict(best_model_weights)
    print(f'Best validation accuracy: {best_accuracy:.4f}')
    return model


def validate(model, val_loader):
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


def calibrate_temperature(model, val_loader):
    temperature_model = TemperatureScaling(model)
    optimizer = optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def eval():
        temperature_model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = temperature_model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def closure():
        optimizer.zero_grad()
        loss = torch.tensor(eval(), requires_grad=True)
        loss.backward()
        return loss

    optimizer.step(closure)

    return temperature_model


def evaluate_calibration(model, data_loader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Brier Score
    brier_score = brier_score_loss(all_labels, all_probs[:, 1])

    # Expected Calibration Error (ECE)
    confidences = np.max(all_probs, axis=1)
    predictions = np.argmax(all_probs, axis=1)
    accuracies = predictions == all_labels

    ece = 0
    for bin in range(10):
        bin_start = bin / 10
        bin_end = (bin + 1) / 10
        bin_indices = np.where((confidences >= bin_start) & (confidences < bin_end))[0]
        if len(bin_indices) > 0:
            bin_accuracy = np.mean(accuracies[bin_indices])
            bin_confidence = np.mean(confidences[bin_indices])
            ece += len(bin_indices) * np.abs(bin_accuracy - bin_confidence)
    ece /= len(all_labels)

    return brier_score, ece


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def predict_image(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # バッチ次元を追加

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        kinoko_prob = probabilities[0][0].item() * 100
        takenoko_prob = probabilities[0][1].item() * 100

    return kinoko_prob, takenoko_prob


def predict_test_images(model, test_dir, transform):
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(test_dir, filename)
            kinoko_prob, takenoko_prob = predict_image(model, image_path, transform)
            print(f"ファイル名: {filename}")
            print(f"きのこ度: {kinoko_prob:.2f}%, たけのこ度: {takenoko_prob:.2f}%")
            print()


def create_submission(model, test_dir, transform):
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())

    test_filenames = [os.path.basename(x[0]) for x in test_dataset.imgs]
    submission = pd.DataFrame({'filename': test_filenames, 'class': predictions})
    submission.to_csv('results/submission_temperature_scaled.csv', index=False)
    print("Submission file created: submission_temperature_scaled.csv")


def main():
    transform = create_transforms()

    train_dataset, val_dataset, test_dataset = load_and_split_data('data/train', transform)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

    model = create_model()
    model = train_model(model, train_loader, val_loader)

    save_model(model, 'models/model_avoidance.pth')

    print("Evaluating original model...")
    original_brier, original_ece = evaluate_calibration(model, test_loader)

    print("Calibrating temperature...")
    temperature_model = calibrate_temperature(model, val_loader)

    print("Evaluating temperature-scaled model...")
    scaled_brier, scaled_ece = evaluate_calibration(temperature_model, test_loader)

    print(f"Original model - Brier Score: {original_brier:.4f}, ECE: {original_ece:.4f}")
    print(f"Scaled model - Brier Score: {scaled_brier:.4f}, ECE: {scaled_ece:.4f}")
    print(f"Optimal Temperature: {temperature_model.temperature.item():.4f}")

    save_model(temperature_model, 'models/model_avoidance_temperature_scaled.pth')

    predict_test_images(temperature_model, 'data/test', transform)

    create_submission(temperature_model, 'data/test', transform)


if __name__ == "__main__":
    main()

# import os
# from PIL import Image
# import torch
# import torch.nn.functional as F
# from torchvision import transforms, datasets
# from torch.utils.data import DataLoader
#
# # テストデータセットのロード
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor()
# ])
# test_dataset = datasets.ImageFolder('chocolate/test', transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#
# model.eval()
#
# def predict_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0)
#
#     with torch.no_grad():
#         outputs = model(image)
#         probabilities = F.softmax(outputs, dim=1)
#         kinoko_prob = probabilities[0][0].item() * 100
#         takenoko_prob = probabilities[0][1].item() * 100
#
#         print(f"ファイル名: {os.path.basename(image_path)}")
#         print(f"きのこ度: {kinoko_prob:.2f}%, たけのこ度: {takenoko_prob:.2f}%")
#
# # 画像ファイルのディレクトリを指定
# image_dir = 'data/test'
#
# # ディレクトリ内の全ての画像ファイルに対して予測を実行
# for filename in os.listdir(image_dir):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#         image_path = os.path.join(image_dir, filename)
#         predict_image(image_path)
#         print()  # 空行を出力して結果を見やすくする
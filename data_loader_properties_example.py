import torch
from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = list(range(size))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.tensor([self.data[idx]])


# データセットとDataLoaderの作成
dataset = SimpleDataset(10)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

print("1. イテラブルの性質:")
for epoch in range(2):
    print(f"Epoch {epoch + 1}:")
    for batch in dataloader:
        print(f"  Batch: {batch.squeeze().tolist()}")
    print()

print("2-1. 反復可能な性質:")
iterator = iter(dataloader)
print("First iteration:", next(iterator).squeeze().tolist())
print("Second iteration:", next(iterator).squeeze().tolist())

print("2-2. 反復可能な性質-より詳細に型を調べてみる:")
for batch in dataloader:
    print(type(batch), batch)

    squeezed_batch = batch.squeeze()
    print(type(squeezed_batch), squeezed_batch)

    list_result = squeezed_batch.tolist()
    print(type(list_result), list_result)

print("\n3. ランダムアクセス不可:")
try:
    print(dataloader[0])
except TypeError as e:
    print(f"Error: {e}")

print("\n4. 遅延評価（実際のデータローディングはイテレーション時）")

print("\n5. マルチプロセス対応（この例では示されていません）")

print("\n6. 自動バッチ処理:")
print("Batch size:", dataloader.batch_size)
print("Number of batches:", len(dataloader))

print("\n7. オプションのシャッフル（この例ではシャッフルが有効）")

print("\n8. カスタマイズ可能（この例では基本的な設定を使用）")

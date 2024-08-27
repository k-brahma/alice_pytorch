from torch.utils.data import Dataset, DataLoader


# 簡単なデータセットの定義
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# サンプルデータ
data = list(range(20))  # 0から19までの数字

# データセットの作成
dataset = SimpleDataset(data)

# DataLoaderの作成
batch_size = 4
shuffle = True
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# DataLoaderの動作をシミュレート
for epoch in range(2):  # 2エポック分をシミュレート
    print(f"Epoch {epoch + 1}")
    for batch in dataloader:
        print(f"  Batch: {batch.tolist()}")
    print()

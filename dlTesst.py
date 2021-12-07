import torch
from torch.utils.data import DataLoader, Dataset

class MyCustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    # 创建一些模拟数据
X = torch.linspace(0, 100, 100)
y = X * 2 + 5

# 创建自定义Dataset实例
dataset = MyCustomDataset(X, y)

# 创建DataLoader
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

# 遍历DataLoader并打印数据
for batch_X, batch_y in dataloader:
    print(batch_X, batch_y)
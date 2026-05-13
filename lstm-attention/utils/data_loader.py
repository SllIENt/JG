import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class SMDDataset(Dataset):
    """SMD 数据集加载器"""

    def __init__(self, data_path, win_size=100, mode='train'):
        self.win_size = win_size
        self.mode = mode
        self.scaler = StandardScaler()

        # 加载数据
        train_data = np.load(f'{data_path}/SMD_train.npy')
        test_data = np.load(f'{data_path}/SMD_test.npy')
        test_labels = np.load(f'{data_path}/SMD_test_label.npy')

        # 标准化
        self.scaler.fit(train_data)
        self.train = self.scaler.transform(train_data)
        self.test = self.scaler.transform(test_data)
        self.test_labels = test_labels

        # 验证集使用训练集的后 20%
        train_len = len(self.train)
        self.val = self.train[int(train_len * 0.8):]

    def __len__(self):
        if self.mode == 'train':
            return len(self.train) - self.win_size + 1
        elif self.mode == 'val':
            return len(self.val) - self.win_size + 1
        else:
            return len(self.test) - self.win_size + 1

    def __getitem__(self, idx):
        if self.mode == 'train':
            data = self.train[idx:idx + self.win_size]
            # 重建目标：输入本身
            return torch.FloatTensor(data), torch.FloatTensor(data)
        elif self.mode == 'val':
            data = self.val[idx:idx + self.win_size]
            return torch.FloatTensor(data), torch.FloatTensor(data)
        else:
            data = self.test[idx:idx + self.win_size]
            labels = self.test_labels[idx:idx + self.win_size]
            return torch.FloatTensor(data), torch.FloatTensor(data), torch.LongTensor(labels)


def get_dataloader(data_path, batch_size, win_size=100, mode='train'):
    """获取数据加载器"""
    dataset = SMDDataset(data_path, win_size, mode)

    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

    return dataloader

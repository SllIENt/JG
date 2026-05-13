import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class SMDClassifierDataset(Dataset):
    """SMD 分类数据集加载器"""

    def __init__(self, data_path, win_size=100, step=100, mode='train'):
        self.win_size = win_size
        self.step = step
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

        # 训练集标签（全部为 0，因为训练集是正常数据）
        self.train_labels = np.zeros(len(self.train))

        # 验证集使用训练集的后 20%
        train_len = len(self.train)
        self.val = self.train[int(train_len * 0.8):]
        self.val_labels = self.train_labels[int(train_len * 0.8):]

    def __len__(self):
        if self.mode == 'train':
            return (len(self.train) - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (len(self.val) - self.win_size) // self.step + 1
        else:
            return (len(self.test) - self.win_size) // self.step + 1

    def __getitem__(self, idx):
        idx = idx * self.step
        if self.mode == 'train':
            data = self.train[idx:idx + self.win_size]
            # 如果窗口中包含任何异常点，则标签为 1
            label = 1 if np.any(self.train_labels[idx:idx + self.win_size] == 1) else 0
            return torch.FloatTensor(data), torch.FloatTensor([label])
        elif self.mode == 'val':
            data = self.val[idx:idx + self.win_size]
            label = 1 if np.any(self.val_labels[idx:idx + self.win_size] == 1) else 0
            return torch.FloatTensor(data), torch.FloatTensor([label])
        else:
            data = self.test[idx:idx + self.win_size]
            # 测试集使用真实标签
            label = 1 if np.any(self.test_labels[idx:idx + self.win_size] == 1) else 0
            return torch.FloatTensor(data), torch.FloatTensor([label])


def get_classifier_dataloader(data_path, batch_size, win_size=100, step=100, mode='train'):
    """获取分类数据加载器"""
    dataset = SMDClassifierDataset(data_path, win_size, step, mode)

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

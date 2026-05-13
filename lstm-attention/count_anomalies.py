import numpy as np

data_path = '../Anomaly-Transformer/dataset/SMD'

# 加载标签
test_labels = np.load(f'{data_path}/SMD_test_label.npy')

print(f'测试集标签形状: {test_labels.shape}')
print(f'总样本数: {len(test_labels)}')

# 统计正常和异常样本
normal_count = np.sum(test_labels == 0)
anomaly_count = np.sum(test_labels == 1)

print(f'正常样本数: {normal_count}')
print(f'异常样本数: {anomaly_count}')
print(f'异常比例: {anomaly_count / len(test_labels) * 100:.2f}%')

# 查看标签分布
print(f'\n标签值分布:')
unique, counts = np.unique(test_labels, return_counts=True)
for val, count in zip(unique, counts):
    print(f'  标签 {val}: {count} 个样本')

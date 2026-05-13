import numpy as np

data_path = '../Anomaly-Transformer/dataset/SMD'

# 加载标签
test_labels = np.load(f'{data_path}/SMD_test_label.npy')

print('='*50)
print('原始数据集统计')
print('='*50)
print(f'测试集标签形状: {test_labels.shape}')
print(f'总时间步数: {len(test_labels)}')

# 统计正常和异常样本
normal_count = np.sum(test_labels == 0)
anomaly_count = np.sum(test_labels == 1)

print(f'正常时间步数: {normal_count}')
print(f'异常时间步数: {anomaly_count}')
print(f'异常比例: {anomaly_count / len(test_labels) * 100:.2f}%')

# 滑动窗口统计
win_size = 100
step = 100

print('\n' + '='*50)
print(f'滑动窗口统计 (win_size={win_size}, step={step})')
print('='*50)

# 计算窗口数量
num_windows = (len(test_labels) - win_size) // step + 1
print(f'总窗口数: {num_windows}')

# 统计包含异常的窗口数
anomaly_windows = 0
normal_windows = 0

for i in range(0, len(test_labels) - win_size + 1, step):
    window_labels = test_labels[i:i + win_size]
    if np.any(window_labels == 1):
        anomaly_windows += 1
    else:
        normal_windows += 1

print(f'正常窗口数: {normal_windows}')
print(f'异常窗口数: {anomaly_windows}')
print(f'异常窗口比例: {anomaly_windows / num_windows * 100:.2f}%')

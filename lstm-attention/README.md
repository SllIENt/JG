# LSTM + 注意力机制 时间序列异常检测

## 项目概述

使用 LSTM 作为基线模型，添加注意力机制作为改进，进行时间序列异常检测对比实验。

## 数据集

**SMD (Server Machine Dataset)**
- 38 维特征
- 训练集：正常数据
- 测试集：包含正常和异常数据

## 模型

### 基线模型：LSTM
- 2 层 LSTM
- 128 维隐藏层
- 全连接输出层

### 改进模型：LSTM + 注意力
- 2 层 LSTM
- 4 头自注意力机制
- 残差连接 + 层归一化

## 使用方法

### 1. 准备数据
```bash
# 将 SMD 数据集放入 data/dataset/SMD/ 目录
cp -r /path/to/SMD data/dataset/
```

### 2. 训练模型
```bash
# 训练 LSTM 基线模型
python train.py --model lstm --epochs 50

# 训练 LSTM + 注意力模型
python train.py --model lstm_attention --epochs 50
```

### 3. 测试模型
```bash
# 测试 LSTM 基线模型
python test.py --model lstm

# 测试 LSTM + 注意力模型
python test.py --model lstm_attention
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | lstm | 模型类型 (lstm/lstm_attention) |
| `--hidden_dim` | 128 | LSTM 隐藏层维度 |
| `--num_layers` | 2 | LSTM 层数 |
| `--n_heads` | 4 | 注意力头数 |
| `--win_size` | 100 | 滑动窗口大小 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--epochs` | 50 | 训练轮数 |

## 评估指标

- **F1-score** (主指标)
- Precision
- Recall
- Accuracy

## 文件结构

```
anomaly-lstm-attention/
├── data/
│   └── dataset/
│       └── SMD/           # SMD 数据集
├── models/
│   ├── lstm.py           # LSTM 基线模型
│   └── lstm_attention.py # LSTM + 注意力模型
├── utils/
│   └── data_loader.py    # 数据加载器
├── train.py              # 训练脚本
├── test.py               # 测试脚本
└── README.md
```

#!/bin/bash

# 运行实验脚本

echo "=========================================="
echo "时间序列异常检测对比实验"
echo "=========================================="

# 创建必要目录
mkdir -p data/dataset
mkdir -p checkpoints

# 检查数据集
if [ ! -d "data/dataset/SMD" ]; then
    echo "错误：请先将 SMD 数据集放入 data/dataset/SMD/ 目录"
    exit 1
fi

echo ""
echo "Step 1: 训练 LSTM 基线模型"
echo "------------------------------------------"
python train.py --model lstm --epochs 50 --batch_size 32

echo ""
echo "Step 2: 训练 LSTM + 注意力模型"
echo "------------------------------------------"
python train.py --model lstm_attention --epochs 50 --batch_size 32 --n_heads 4

echo ""
echo "Step 3: 测试 LSTM 基线模型"
echo "------------------------------------------"
python test.py --model lstm

echo ""
echo "Step 4: 测试 LSTM + 注意力模型"
echo "------------------------------------------"
python test.py --model lstm_attention

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo ""
echo "结果文件："
echo "  - checkpoints/lstm_results.txt"
echo "  - checkpoints/lstm_attention_results.txt"

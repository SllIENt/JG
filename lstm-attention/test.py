import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models.lstm import LSTMAnomalyDetector
from models.lstm_attention import LSTMAttentionAnomalyDetector
from utils.data_loader import get_dataloader


def calculate_anomaly_scores(model, test_loader, device):
    """计算异常分数"""
    model.eval()
    all_errors = []
    all_labels = []

    with torch.no_grad():
        for data, target, labels in test_loader:
            data = data.to(device)
            target = target.to(device)

            if isinstance(model, LSTMAttentionAnomalyDetector):
                output, _ = model(data)
            else:
                output = model(data)

            # 计算重建误差
            error = torch.mean((output - target) ** 2, dim=-1)  # (batch, seq_len)

            all_errors.append(error.cpu().numpy())
            all_labels.append(labels.numpy())

    all_errors = np.concatenate(all_errors, axis=0).reshape(-1)
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1)

    return all_errors, all_labels


def find_threshold(errors, labels, percentile=95):
    """寻找最优阈值"""
    # 使用训练集误差的百分位数作为阈值
    threshold = np.percentile(errors, percentile)
    return threshold


def evaluate(errors, labels, threshold):
    """评估模型"""
    predictions = (errors > threshold).astype(int)
    labels = labels.astype(int)

    # 计算指标
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )

    return accuracy, precision, recall, f_score


def main():
    parser = argparse.ArgumentParser(description='Test LSTM Anomaly Detector')
    parser.add_argument('--data_path', type=str, default='data/dataset/SMD')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'lstm_attention'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint', type=str, default='checkpoints')
    parser.add_argument('--percentile', type=float, default=95)

    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据加载
    train_loader = get_dataloader(args.data_path, args.batch_size, args.win_size, 'train')
    test_loader = get_dataloader(args.data_path, args.batch_size, args.win_size, 'test')

    # 模型
    input_dim = 38
    output_dim = input_dim

    if args.model == 'lstm':
        model = LSTMAnomalyDetector(input_dim, args.hidden_dim, args.num_layers, output_dim)
    else:
        model = LSTMAttentionAnomalyDetector(input_dim, args.hidden_dim, args.num_layers, output_dim, args.n_heads)

    # 加载模型
    checkpoint_path = os.path.join(args.checkpoint, f'{args.model}_best.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f'Loaded model from {checkpoint_path}')
    else:
        print(f'No checkpoint found at {checkpoint_path}')
        return

    model = model.to(device)

    # 计算训练集误差（用于确定阈值）
    print('Calculating training errors...')
    train_errors, _ = calculate_anomaly_scores(model, train_loader, device)
    threshold = find_threshold(train_errors, None, args.percentile)
    print(f'Threshold (percentile={args.percentile}): {threshold:.6f}')

    # 测试
    print('Testing...')
    test_errors, test_labels = calculate_anomaly_scores(model, test_loader, device)

    # 评估
    accuracy, precision, recall, f_score = evaluate(test_errors, test_labels, threshold)

    print('\n' + '='*50)
    print('Results:')
    print('='*50)
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1-score:  {f_score:.4f}')
    print('='*50)

    # 保存结果
    results = {
        'model': args.model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f_score,
        'threshold': threshold
    }

    results_file = os.path.join(args.checkpoint, f'{args.model}_results.txt')
    with open(results_file, 'w') as f:
        for key, value in results.items():
            f.write(f'{key}: {value}\n')
    print(f'Results saved to {results_file}')


if __name__ == '__main__':
    main()

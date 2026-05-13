import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

from models.lstm_classifier import LSTMClassifier, LSTMAttentionClassifier
from utils.classifier_data_loader import get_classifier_dataloader


def evaluate(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)

            if isinstance(model, LSTMAttentionClassifier):
                output, _ = model(data)
            else:
                output = model(data)

            # 预测
            predicted = (output > 0.5).float().cpu().numpy()
            all_predictions.append(predicted)
            all_labels.append(labels.numpy())

    all_predictions = np.concatenate(all_predictions, axis=0).reshape(-1)
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )

    return accuracy, precision, recall, f_score, all_predictions, all_labels


def main():
    parser = argparse.ArgumentParser(description='Test LSTM Classifier for Anomaly Detection')
    parser.add_argument('--data_path', type=str, default='../Anomaly-Transformer/dataset/SMD')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'lstm_attention'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint', type=str, default='checkpoints')

    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据加载
    print('Loading test data...')
    test_loader = get_classifier_dataloader(args.data_path, args.batch_size, args.win_size, args.step, 'test')
    print(f'Test samples: {len(test_loader.dataset)}')

    # 模型
    input_dim = 38

    if args.model == 'lstm':
        model = LSTMClassifier(input_dim, args.hidden_dim, args.num_layers)
    else:
        model = LSTMAttentionClassifier(input_dim, args.hidden_dim, args.num_layers, args.n_heads)

    # 加载模型
    checkpoint_path = os.path.join(args.checkpoint, f'{args.model}_classifier_best.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f'Loaded model from {checkpoint_path}')
    else:
        print(f'No checkpoint found at {checkpoint_path}')
        return

    model = model.to(device)

    # 测试
    print('Testing...')
    accuracy, precision, recall, f_score, predictions, labels = evaluate(model, test_loader, device)

    print('\n' + '='*50)
    print('Results:')
    print('='*50)
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1-score:  {f_score:.4f}')
    print('='*50)

    # 详细分类报告
    print('\nClassification Report:')
    print(classification_report(labels, predictions, target_names=['Normal', 'Anomaly']))

    # 保存结果
    results = {
        'model': args.model,
        'method': 'classification',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f_score
    }

    results_file = os.path.join(args.checkpoint, f'{args.model}_classifier_results.txt')
    with open(results_file, 'w') as f:
        for key, value in results.items():
            f.write(f'{key}: {value}\n')
    print(f'Results saved to {results_file}')


if __name__ == '__main__':
    main()

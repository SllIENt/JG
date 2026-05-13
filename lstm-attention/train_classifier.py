import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from models.lstm_classifier import LSTMClassifier, LSTMAttentionClassifier
from utils.classifier_data_loader import get_classifier_dataloader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        if isinstance(model, LSTMAttentionClassifier):
            output, _ = model(data)
        else:
            output = model(data)

        # 计算损失
        loss = criterion(output, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算准确率
        predicted = (output > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 每 100 个 batch 打印进度
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.6f}')

    accuracy = correct / total
    return total_loss / len(train_loader), accuracy


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)

            if isinstance(model, LSTMAttentionClassifier):
                output, _ = model(data)
            else:
                output = model(data)

            loss = criterion(output, labels)
            total_loss += loss.item()

            predicted = (output > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return total_loss / len(val_loader), accuracy


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Classifier for Anomaly Detection')
    parser.add_argument('--data_path', type=str, default='../Anomaly-Transformer/dataset/SMD')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'lstm_attention'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_path', type=str, default='checkpoints')

    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据加载
    print(f'Loading data from: {args.data_path}')
    print('Loading train data...')
    train_loader = get_classifier_dataloader(args.data_path, args.batch_size, args.win_size, args.step, 'train')
    print(f'Train samples: {len(train_loader.dataset)}')
    print('Loading val data...')
    val_loader = get_classifier_dataloader(args.data_path, args.batch_size, args.win_size, args.step, 'val')
    print(f'Val samples: {len(val_loader.dataset)}')
    print('Data loaded successfully!')

    # 模型
    input_dim = 38

    if args.model == 'lstm':
        model = LSTMClassifier(input_dim, args.hidden_dim, args.num_layers)
    else:
        model = LSTMAttentionClassifier(input_dim, args.hidden_dim, args.num_layers, args.n_heads)

    model = model.to(device)
    print(f'Model: {args.model}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # 优化器和损失函数
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()  # 二分类交叉熵损失

    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)

    # 训练
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f} | '
              f'Time: {elapsed:.1f}s')

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            save_file = os.path.join(args.save_path, f'{args.model}_classifier_best.pth')
            torch.save(model.state_dict(), save_file)
            print(f'  -> Saved best model to {save_file}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'Training completed. Best val loss: {best_val_loss:.6f}')


if __name__ == '__main__':
    main()

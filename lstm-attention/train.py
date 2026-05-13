import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from models.lstm import LSTMAnomalyDetector
from models.lstm_attention import LSTMAttentionAnomalyDetector
from utils.data_loader import get_dataloader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # 前向传播
        if isinstance(model, LSTMAttentionAnomalyDetector):
            output, _ = model(data)
        else:
            output = model(data)

        # 计算损失
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            if isinstance(model, LSTMAttentionAnomalyDetector):
                output, _ = model(data)
            else:
                output = model(data)

            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Anomaly Detector')
    parser.add_argument('--data_path', type=str, default='data/dataset/SMD')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'lstm_attention'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_path', type=str, default='checkpoints')

    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据加载
    train_loader = get_dataloader(args.data_path, args.batch_size, args.win_size, 'train')
    val_loader = get_dataloader(args.data_path, args.batch_size, args.win_size, 'val')

    # 模型
    input_dim = 38  # SMD 数据集特征维度
    output_dim = input_dim

    if args.model == 'lstm':
        model = LSTMAnomalyDetector(input_dim, args.hidden_dim, args.num_layers, output_dim)
    else:
        model = LSTMAttentionAnomalyDetector(input_dim, args.hidden_dim, args.num_layers, output_dim, args.n_heads)

    model = model.to(device)
    print(f'Model: {args.model}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # 优化器和损失函数
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)

    # 训练
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Train Loss: {train_loss:.6f} | '
              f'Val Loss: {val_loss:.6f} | '
              f'Time: {elapsed:.1f}s')

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            save_file = os.path.join(args.save_path, f'{args.model}_best.pth')
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

import torch
import torch.nn as nn


class LSTMAnomalyDetector(nn.Module):
    """LSTM 基线模型用于时间序列异常检测"""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        output: (batch, seq_len, output_dim)
        """
        # LSTM 前向传播
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # 全连接层
        output = self.fc(lstm_out)  # (batch, seq_len, output_dim)

        return output

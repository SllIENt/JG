import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM 分类器用于时间序列异常检测"""

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
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

        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        output: (batch, 1) - 异常概率 0-1
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # 只取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # 分类
        out = self.fc(last_out)  # (batch, 1)
        out = self.sigmoid(out)  # (batch, 1)

        return out


class LSTMAttentionClassifier(nn.Module):
    """LSTM + 注意力分类器用于时间序列异常检测"""

    def __init__(self, input_dim, hidden_dim, num_layers, n_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_heads = n_heads

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        output: (batch, 1) - 异常概率 0-1
        attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # 自注意力
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )  # (batch, seq_len, hidden_dim)

        # 残差连接 + 层归一化
        out = self.norm(lstm_out + attn_out)

        # 取最后一个时间步
        last_out = out[:, -1, :]  # (batch, hidden_dim)

        # 分类
        out = self.fc(last_out)  # (batch, 1)
        out = self.sigmoid(out)  # (batch, 1)

        return out, attn_weights

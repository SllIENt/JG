import torch
import torch.nn as nn


class LSTMAttentionAnomalyDetector(nn.Module):
    """LSTM + 自注意力机制用于时间序列异常检测"""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_heads = n_heads

        # LSTM 层
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 自注意力层
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        output: (batch, seq_len, output_dim)
        attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        # LSTM 前向传播
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # 自注意力
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )  # (batch, seq_len, hidden_dim)

        # 残差连接 + 层归一化
        out = self.norm(lstm_out + attn_out)

        # 全连接层
        output = self.fc(out)  # (batch, seq_len, output_dim)

        return output, attn_weights

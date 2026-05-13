import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyLayer(nn.Module):
    """频域注意力层：FFT -> 注意力 -> 逆变换 -> 融合"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(FrequencyLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 频域内的注意力
        self.freq_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # 融合层：时域特征 + 频域增强特征 -> 输出
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        B, L, D = x.shape

        # Step 1: FFT 变换到频域
        # rfft 输出 (batch, seq_len//2+1, d_model) 的复数
        x_freq = torch.fft.rfft(x, dim=1)

        # Step 2: 分离实部虚部，各自做注意力
        x_real = x_freq.real  # (B, L//2+1, D)
        x_imag = x_freq.imag  # (B, L//2+1, D)

        attn_real, _ = self.freq_attention(x_real, x_real, x_real)
        attn_imag, _ = self.freq_attention(x_imag, x_imag, x_imag)

        # Step 3: 逆变换回时域
        x_freq_enhanced = torch.complex(attn_real, attn_imag)
        x_time_from_freq = torch.fft.irfft(x_freq_enhanced, n=L, dim=1)  # (B, L, D)

        # Step 4: 和原始时域特征融合
        concat = torch.cat([x, x_time_from_freq], dim=-1)  # (B, L, 2*D)
        output = self.fusion(concat)  # (B, L, D)
        output = self.dropout(output)
        output = self.norm(output + x)  # 残差连接

        return output

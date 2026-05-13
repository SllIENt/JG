# 代码改动记录

## 概述
在原始 Anomaly Transformer 基础上加入频域注意力分支，用于提升时间序列异常检测效果。

---

## 新增文件（完全由我编写）

### 1. `model/freq_layer.py`
- **作用**：频域注意力层
- **内容**：`FrequencyLayer` 类，实现 FFT -> 频域注意力 -> 逆变换 -> 融合
- **核心逻辑**：
  - `torch.fft.rfft()` 将时域输入变换到频域
  - 对实部和虚部分别做 `nn.MultiheadAttention`
  - `torch.fft.irfft()` 逆变换回时域
  - 拼接时域原始特征和频域增强特征，通过线性层融合

### 2. `scripts/SMD_freq.sh`
- **作用**：SMD 数据集的训练/测试脚本（频域增强版本）
- **内容**：调用 `main.py` 的命令行参数，包含 `--use_freq` 和 `--freq_loss_weight`

### 3. `CHANGES.md`（本文件）
- **作用**：记录所有代码改动

---

## 修改的文件

### 1. `model/AnomalyTransformer.py`

**改动位置 1：导入（第 4 行）**
```python
# 新增
from .freq_layer import FrequencyLayer
```

**改动位置 2：Encoder 类（第 34-54 行）**
```python
# 原始：
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        ...
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            ...

# 修改后：
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None, freq_layers=None):
        ...
        self.attn_layers = nn.ModuleList(attn_layers)
        self.freq_layers = nn.ModuleList(freq_layers) if freq_layers else None  # 新增
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for i, attn_layer in enumerate(self.attn_layers):
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            if self.freq_layers is not None and i < len(self.freq_layers):  # 新增
                x = self.freq_layers[i](x)                                  # 新增
            ...
```

**改动位置 3：AnomalyTransformer 类（第 57-82 行）**
```python
# 原始：
def __init__(self, ..., output_attention=True):
    ...
    self.encoder = Encoder([...], norm_layer=...)

# 修改后：
def __init__(self, ..., output_attention=True, use_freq=True):  # 新增参数
    ...
    self.use_freq = use_freq
    freq_layers = [FrequencyLayer(d_model, n_heads, dropout) for _ in range(e_layers)] if use_freq else None  # 新增
    self.encoder = Encoder([...], norm_layer=..., freq_layers=freq_layers)  # 传入频域层
```

---

### 2. `solver.py`

**改动位置 1：__init__ 方法（新增 freq_loss_weight 属性）**
```python
# 新增
self.freq_loss_weight = config.get('freq_loss_weight', 0.1)

# 新增方法
def freq_recon_loss(self, output, target):
    """频域重建损失"""
    output_freq = torch.fft.rfft(output, dim=1)
    target_freq = torch.fft.rfft(target, dim=1)
    loss = F.mse_loss(output_freq.abs(), target_freq.abs())
    return loss
```

**改动位置 2：build_model 方法**
```python
# 原始：
self.model = AnomalyTransformer(...)

# 修改后：
use_freq = getattr(self, 'use_freq', True)
self.model = AnomalyTransformer(..., use_freq=use_freq)
```

**改动位置 3：train 方法（损失计算部分）**
```python
# 原始：
rec_loss = self.criterion(output, input)
loss1 = rec_loss - self.k * series_loss
loss2 = rec_loss + self.k * prior_loss

# 修改后：
rec_loss = self.criterion(output, input)
f_loss = self.freq_recon_loss(output, input)  # 新增
rec_loss_total = rec_loss + self.freq_loss_weight * f_loss  # 新增
loss1 = rec_loss_total - self.k * series_loss
loss2 = rec_loss_total + self.k * prior_loss
```

**改动位置 4：vali 方法（同上逻辑）**
```python
# 同 train 方法的改动，加入频域损失
```

---

### 3. `main.py`

**改动位置：参数解析（新增两个参数）**
```python
# 新增
parser.add_argument('--use_freq', type=str2bool, default=True)
parser.add_argument('--freq_loss_weight', type=float, default=0.1)
```

---

## 改动总结

| 文件 | 改动类型 | 改动行数 | 说明 |
|------|----------|----------|------|
| `model/freq_layer.py` | 新增 | ~40 行 | 频域注意力层 |
| `model/AnomalyTransformer.py` | 修改 | ~15 行 | 加入频域分支 |
| `solver.py` | 修改 | ~20 行 | 加入频域损失 |
| `main.py` | 修改 | 2 行 | 新增命令行参数 |
| `scripts/SMD_freq.sh` | 新增 | ~8 行 | 训练脚本 |

**总新增/修改代码：约 85 行**

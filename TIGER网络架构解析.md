# TIGER 网络架构详解

## 1. 项目概述

TIGER (Time-frequency Interleaved Gain Extraction and Reconstruction) 是一个轻量级语音分离模型，由清华大学的研究团队提出，发表于 ICLR 2025。

### 核心亮点
- **参数量减少 94.3%**：相比 SOTA 模型 TF-GridNet
- **计算量减少 95.3%**：MACs 大幅降低
- **首个百万参数以下**：达到接近 SOTA 性能的语音分离模型

### 应用场景
1. **语音分离**：将混合语音分离成独立说话人（2人）
2. **影视音频分离 (DnR)**：分离对话(Dialog)、音效(Effect)、音乐(Music)

---

## 2. 整体网络架构

```
输入音频 → STFT → 频带分割 → 特征提取(BN) → 分离器(Recurrent) → 掩码生成 → iSTFT → 输出音频
```

### 2.1 数据流维度变化

| 阶段 | 维度 | 说明 |
|------|------|------|
| 输入 | `[B, C, T]` | 批次、通道、时间采样点 |
| STFT后 | `[B*C, 2, F, T']` | 实部+虚部、频率bins、时间帧 |
| 子带特征 | `[B, nband, N, T']` | 频带数、特征维度、时间帧 |
| 分离输出 | `[B, nband, N, T']` | 同上 |
| 掩码应用后 | `[B*C*K, F, T']` | K为说话人数量 |
| 最终输出 | `[B*C, K, T]` | 分离后的K个音频 |

---

## 3. 核心模块详解

### 3.1 频带分割策略 (Band-Split)

TIGER 的关键创新之一是基于先验知识的非均匀频带分割：

```python
# 16kHz 采样率下的频带划分 (tiger.py)
# 0-1kHz:   25Hz 带宽 × 40个子带  (低频精细划分，语音基频区域)
# 1-2kHz:   100Hz 带宽 × 10个子带
# 2-4kHz:   250Hz 带宽 × 8个子带
# 4-8kHz:   500Hz 带宽 × 8个子带  (高频粗略划分)
```

**设计原理**：
- 低频区域（0-1kHz）包含语音的基频和重要谐波，需要更精细的频率分辨率
- 高频区域对语音可懂度贡献较小，可以使用更粗的频率分辨率
- 这种非均匀划分大幅减少了计算量，同时保持了分离质量

### 3.2 特征提取层 (BN - Bottleneck)

```python
self.BN = nn.ModuleList([])
for i in range(self.nband):
    self.BN.append(nn.Sequential(
        nn.GroupNorm(1, self.band_width[i]*2, self.eps),  # 全局层归一化
        nn.Conv1d(self.band_width[i]*2, self.feature_dim, 1)  # 1x1卷积降维
    ))
```

**作用**：
- 对每个子带的实部+虚部进行归一化
- 通过 1×1 卷积将不同宽度的子带统一映射到相同的特征维度 `feature_dim`

### 3.3 Recurrent 分离器（核心）

这是 TIGER 的核心分离模块，采用**时频交替处理**的策略：

```python
class Recurrent(nn.Module):
    def __init__(self, ...):
        # 频率路径：处理频率维度的上下文
        self.freq_path = nn.ModuleList([
            UConvBlock(...),           # MSA模块
            MultiHeadSelfAttention2D(...),  # F³A模块
            LayerNorm(...)
        ])

        # 时间路径：处理时间维度的上下文
        self.frame_path = nn.ModuleList([
            UConvBlock(...),           # MSA模块
            MultiHeadSelfAttention2D(...),  # F³A模块
            LayerNorm(...)
        ])
```

**迭代处理流程**：
```
for i in range(num_blocks):
    if i == 0:
        x = freq_time_process(x)
    else:
        x = freq_time_process(mixture + x)  # 残差连接
```

每次迭代包含：
1. **频率路径处理**：在每个时间帧内，跨频带建模
2. **时间路径处理**：在每个频带内，跨时间帧建模

### 3.4 UConvBlock - 多尺度选择性注意力 (MSA)

```python
class UConvBlock(nn.Module):
    """
    多分辨率特征提取模块
    通过连续下采样和上采样分析不同尺度的特征
    """
```

**处理流程**：

```
输入 x
    ↓
1×1 卷积投影 (升维)
    ↓
多尺度下采样 (spp_dw)
    ├── scale 0: stride=1 (原始分辨率)
    ├── scale 1: stride=2 (1/2 分辨率)
    ├── scale 2: stride=2 (1/4 分辨率)
    └── ...
    ↓
全局特征聚合 (adaptive_avg_pool + Mlp)
    ↓
局部-全局特征融合 (InjectionMultiSum)
    ↓
逐级上采样融合 (last_layer)
    ↓
1×1 卷积投影 (降维) + 残差连接
    ↓
输出
```

**InjectionMultiSum 融合机制**：
```python
out = local_feat * sigmoid(global_act) + global_feat
```
- 全局特征通过 Sigmoid 门控调制局部特征
- 同时加上全局特征的直接贡献

### 3.5 MultiHeadSelfAttention2D - 全频帧注意力 (F³A)

```python
class MultiHeadSelfAttention2D(nn.Module):
    """
    2D 多头自注意力模块
    捕获时间和频率维度的全局上下文信息
    """
```

**计算流程**：
```
输入: [B, C, T, F]
    ↓
Q, K, V 投影 (每个头独立)
    ↓
展平频率维度: Q, K → [B', T, E*F]
    ↓
注意力计算: softmax(Q @ K^T / sqrt(d))
    ↓
加权求和: attn @ V
    ↓
多头拼接 + 投影
    ↓
残差连接
    ↓
输出: [B, C, T, F]
```

**作用**：捕获全局的时间-频率依赖关系，帮助模型理解长距离的上下文信息。

### 3.6 掩码生成与应用

```python
# 生成复数掩码
this_output = self.mask[i](sep_output[:,i]).view(B, 2, 2, K, BW, T)
# 维度: [batch, real/imag, mask/gate, num_speakers, bandwidth, time]

# 门控机制
this_mask = this_output[:,0] * torch.sigmoid(this_output[:,1])

# 强制掩码和为1（保证能量守恒）
this_mask_real = this_mask_real - (this_mask_real_sum - 1) / num_output
this_mask_imag = this_mask_imag - this_mask_imag_sum / num_output

# 复数掩码应用
est_spec_real = spec.real * mask_real - spec.imag * mask_imag
est_spec_imag = spec.real * mask_imag + spec.imag * mask_real
```

**复数掩码的优势**：
- 不仅调整幅度，还能调整相位
- 更好地处理混叠和干扰

---

## 4. TIGERDNR - 影视音频分离

TIGERDNR 是 TIGER 的扩展版本，用于分离影视音频中的三个成分：

```python
class TIGERDNR(BaseModel):
    def __init__(self, ...):
        self.dialog = TIGER(...)  # 对话分离器
        self.effect = TIGER(...)  # 音效分离器
        self.music = TIGER(...)   # 音乐分离器
```

**特点**：
- 使用三个独立的 TIGER 网络
- 采用滑动窗口推理 (`wav_chunk_inference`) 处理长音频
- 44.1kHz 采样率，更宽的频带划分

---

## 5. 关键设计总结

| 设计 | 作用 | 效果 |
|------|------|------|
| 非均匀频带分割 | 利用语音先验知识 | 减少计算量，保持性能 |
| 时频交替处理 | 分别建模时间和频率上下文 | 避免联合建模的高复杂度 |
| 多尺度选择性注意力 (MSA) | 多分辨率特征提取 | 捕获不同尺度的模式 |
| 全频帧注意力 (F³A) | 全局上下文建模 | 长距离依赖捕获 |
| 复数掩码 + 门控 | 幅度和相位联合估计 | 更精确的信号重建 |
| 迭代精化 | 多次处理逐步改善 | 提升分离质量 |

---

## 6. 配置参数说明

```yaml
audionet:
  audionet_name: TIGER
  audionet_config:
    out_channels: 128      # 特征维度
    in_channels: 256       # UConvBlock 内部维度
    num_blocks: 8          # 迭代次数
    upsampling_depth: 5    # 多尺度深度
    win: 640               # STFT 窗口大小
    stride: 160            # STFT 步长
    num_sources: 2         # 分离的说话人数量
```

---

## 7. 参考资料

- 论文: [TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation](https://arxiv.org/abs/2410.01469)
- 代码: https://github.com/JusperLee/TIGER
- 预训练模型: https://huggingface.co/JusperLee/TIGER-speech
- 数据集 EchoSet: https://huggingface.co/datasets/JusperLee/EchoSet

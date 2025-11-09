# Transformer Language Modeling Project

一个从头开始实现的Transformer模型，用于小规模文本建模任务。本项目完整实现了Transformer架构，包括multi-head self-attention、position-wise FFN、残差连接与LayerNorm、位置编码等核心组件，并在WikiText-2数据集上进行了训练和消融实验。

## 项目概述

本项目手工搭建了一个完整的Transformer模型，支持以下特性：

- **核心组件**：Multi-head Self-Attention、Position-wise Feed Forward Networks、残差连接 + LayerNorm、位置编码
- **高级特性**：相对位置编码、线性注意力、Decoder支持、训练稳定性技巧
- **实验功能**：参数统计、模型保存/加载、训练曲线可视化、超参数敏感性分析
- **数据集**：在WikiText-2数据集上进行训练和评估

## 项目结构

```
transformer-project/
├── dataset/                    # 数据集目录
│   ├── wikitext-2/
│   │   ├── wiki.test.tokens
│   │   ├── wiki.train.tokens
│   │   └── wiki.valid.tokens
├── src/                       # 源代码
│   ├── data_loader.py        # 数据加载和处理
│   ├── model.py              # Transformer模型实现
│   ├── train.py              # 训练循环和实验
│   └── utils.py              # 工具函数（学习率调度、日志记录等）
├── scripts/
│   └── run.sh                # 实验运行脚本
├── requirements.txt          # Python依赖包
├── results/                  # 实验结果（训练曲线、性能表格）
└── README.md                # 项目说明文档
```

## 环境要求

### 硬件要求
- **GPU**: 至少8GB显存（推荐NVIDIA RTX 3080或更高）
- **内存**: 16GB RAM或更高
- **存储**: 至少5GB可用空间

### 软件依赖
- Python 3.8+
- PyTorch 1.9+
- 其他依赖见 `requirements.txt`

## 安装步骤

1. 克隆项目并安装依赖：
```bash
git clone https://github.com/wujiaqi2333/manualTransformer.git
cd manualTransformer
pip install -r requirements.txt
```

2. 数据集：

可以自行手动下载WikiText-2数据集并放置三个文件到 dataset/wikitext-2/ 目录 。我的项目已经上传并给出数据集，可以直接使用


## 数据集介绍

### WikiText-2
WikiText-2是一个包含Wikipedia优质文章的语言建模数据集，特点包括：
- **规模**: 约2,500万token
- **词汇量**: 约33,000个单词
- **分割**: 训练集(~800篇文章)、验证集(~1.8万句子)、测试集(~1.8万句子)
- **特点**: 保留原始大小写、标点和数字，包含罕见词汇

**数据集链接**: [WikiText Long Term Dependency Language Modeling Dataset](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/)

## 使用方法

### 快速开始

运行基础模型训练：
```bash
cd scripts
./run.sh
```

### 详细训练命令

#### 基础模型训练（重现实验的exact命令）
```bash
python src/train.py  --batch-size 32   --seq-len 128   --epochs 20    --d-model 512    --n-heads 8    --num-layers 6  --d-ff 2048  --dropout 0.1   --lr 0.0001   --clip-grad 1.0    --warmup-steps 4000     --seed 42  --use-decoder   --save-dir checkpoints/base_model
```

#### 使用相对位置编码
```bash
python src/train.py  --batch-size 32  --seq-len 128   --epochs 20   --d-model 512    --n-heads 8     --num-layers 6     --use-relative-pos     --seed 42     --save-dir checkpoints/relative_pos     --data-dir dataset/wikitext-2
```

#### 使用线性注意力
```bash
python src/train.py  --batch-size 32   --seq-len 128     --epochs 20     --d-model 512     --n-heads 8    --attention-type linear     --seed 42    --save-dir checkpoints/linear_attention     --data-dir dataset/wikitext-2
```

#### 消融实验 - 更小的模型
```bash
python src/train.py  --batch-size 32    --seq-len 128     --epochs 20     --d-model 256     --n-heads 4     --num-layers 4     --d-ff 1024     --seed 43     --save-dir checkpoints/small_model     --data-dir dataset/wikitext-2
```

#### 消融实验 - 更大的模型
```bash
python src/train.py    --batch-size 16    --seq-len 128     --epochs 15     --d-model 768     --n-heads 12     --num-layers 8    --d-ff 3072   --seed 44   --save-dir checkpoints/large_model    --data-dir dataset/wikitext-2
```

## 核心实现

### 模型架构

#### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
```

#### 线性注意力（优化版本）

```python
class LinearAttention(nn.Module):
   def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
      # 使用特征映射和关联属性优化计算
      Q = F.elu(Q) + 1
      K = F.elu(K) + 1

      KV = torch.einsum("bhnd,bhnk->bhdk", K, V)
      Z = 1.0 / (torch.einsum("bhnd,bhd->bhn", Q, K.sum(dim=2)) + 1e-6)
      V = torch.einsum("bhnd,bhdk,bhn->bhnd", Q, KV, Z)
```

#### 位置编码
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
```

### 训练稳定性技巧

#### 学习率调度
```python
class LearningRateScheduler:
    def step(self):
        self.current_step += 1
        lr = self.d_model ** -0.5 * min(
            self.current_step ** -0.5,
            self.current_step * self.warmup_steps ** -1.5
        )
```

#### 梯度裁剪
```python
# 在train.py中
if clip_grad > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
```

## 实验结果

### 性能指标

| 模型配置 | 参数量 | 训练损失   | 测试损失   | 测试困惑度 | 训练时间 |
|---------|--------|--------|--------|-------|------|
| Base (6L, 512D) | ~45M | 0.0471 | 0.0903 | 1.09  | 1.8h |
| + Relative Pos | ~46M | 0.0455 | 0.0901 | 1.07  | 2.1h |
| + Linear Attention | ~45M | 0.0481 | 0.908  | 1.13  | 1.7h |
| Small (4L, 256D) | ~18M | 0.0492 | 0.0898 | 1.09  | 0.4h |
| Large (8L, 768D) | ~125M | 0.1023 | 0.1235 | 1.13  | 1.1h |
ps:除了Large训练了15个epoch之外，其他都是训练了20个epoch

### 训练曲线

训练过程中会自动生成以下可视化结果：
- 训练和验证损失曲线
- 学习率调度曲线
- 困惑度下降趋势

结果保存在 `results/training_curves/` 目录中。

## 关键特性

### 1. 完整的Transformer实现
- Multi-head self-attention机制
- Position-wise Feed Forward Networks
- 残差连接和LayerNorm
- 正弦位置编码和相对位置编码

### 2. 训练优化
- AdamW优化器
- 学习率warmup和调度
- 梯度裁剪
- 检查点保存和恢复

### 3. 实验功能
- 参数统计和模型分析
- 多种注意力机制支持
- 消融实验框架
- 训练过程可视化

### 4. 扩展功能
- Decoder支持
- 相对位置编码
- 线性注意力优化
- 超参数敏感性分析

## 故障排除

### 常见问题

1. **内存不足错误**
   - 减小 `batch-size` 或 `seq-len`
   - 使用 `--attention-type linear` 减少内存使用

2. **数据集加载失败**
   - 检查 `dataset/wikitext-2/` 目录是否存在且包含三个token文件
   - 验证文件路径权限

3. **训练不收敛**
   - 调整学习率 `--lr`
   - 增加 `--warmup-steps`
   - 检查梯度裁剪参数 `--clip-grad`

### 调试模式

启用详细日志：
```bash
python src/train.py --batch-size 16 --seq-len 64 --epochs 5 --d-model 256 --debug
```



## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{transformer_lm,
  title = {Transformer Language Modeling Implementation},
  author = {wujiaqi},
  year = {2025},
  url = {https://github.com/wujiaqi2333/manualTransformer.git}
}
```


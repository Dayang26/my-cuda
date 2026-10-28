"""
Transformer 模型 - 嵌入层模块

该模块实现了 Transformer 中的词嵌入层（Embedding Layer），
用于将离散的词汇索引映射为连续的向量表示。
"""

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedding(nn.Module):
    """
    词嵌入层

    将词汇表中的每个词映射到一个 d_model 维的稠密向量空间。
    这是 Transformer 编码器和解码器的第一层，负责将输入的 token ID
    转换为可学习的连续向量表示。

    Args:
        d_model (int): 嵌入向量的维度，决定每个词的向量长度。
                       同时也是 Transformer 模型的隐藏层维度。
        vocab (int):  词汇表的大小，即词嵌入矩阵的行数。
                      决定了模型能够处理的不同 token 的数量。
    """

    def __init__(self, d_model, vocab):
        """
        初始化嵌入层

        创建一个 vocab x d_model 的可学习嵌入矩阵。

        Args:
            d_model (int): 嵌入向量的维度
            vocab (int): 词汇表大小
        """
        # 调用父类 nn.Module 的构造函数，确保模块正确初始化
        super().__init__()

        # nn.Embedding 是 PyTorch 提供的嵌入层查找表（Lookup Table）
        # - 第一个参数: num_embeddings，表示词汇表大小（嵌入矩阵的行数）
        # - 第二个参数: embedding_dim，表示每个嵌入向量的维度（嵌入矩阵的列数）
        # - 权重矩阵形状: (vocab, d_model)
        # - 输入: (batch_size, sequence_length) 的 LongTensor，表示 token 的索引
        # - 输出: (batch_size, sequence_length, d_model) 的张量
        self.lut = nn.Embedding(vocab, d_model)

        # 保存 d_model 属性，方便在其他地方引用（如计算位置编码时）
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PrositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        position = torch.arange(0.0, max_len, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(
            torch.range(0.0, d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


## attention function implement
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)  # 获取query的最后一个维度
    scores = torch.matmul(query, key.transpose(-2, -1)) // math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, value=-1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """克隆 N 个完全相同的独立子模块，并封装进 nn.ModuleList 中"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


## 多头注意力机制
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super().__init__()

        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        pass

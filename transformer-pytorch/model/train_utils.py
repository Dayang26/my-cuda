import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam


class MultiGPULossCompute:
    def __init__(self):
        super().__init__()
        pass


class NoamOpt:
    def __init__(self, model_size, warmup, factor, optimizer):
        """
        初始化优化器参数
        """
        self.optimizer = optimizer
        self._step = 0  # 当前训练步数
        self.model_size = model_size
        self.warmup = warmup  # 预热步数
        self.factor = factor  # 学习率因子
        self._rate = 0  # 当前学习率

    def step(self):
        """更新优化器参数和学习率"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate()
        self.optimizer.step()

    def rate(self, step=None):
        """根据当前步数计算学习率"""
        if step is None:
            step = self._step

        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def get_std_opt(model):
        # 创建并且返回一个NoamOpt优化器,包含Adam优化器做基础。
        return NoamOpt(
            model_size=model.src_embed[0].d_model,
            factor=1,
            warmup=10000,
            optimizer=torch.optim.Adam(
                model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
            ),
        )

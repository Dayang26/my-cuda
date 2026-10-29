import torch

"""
模型超参数
"""
# 维度
d_model = 512
# 头数
n_head = 8
# 层数
n_learyer = 6
# k v 维度
q_k = 64
q_v = 64
# feed-forward维度
d_ff = 2048
# 随机丢弃10的神经元
dropout = 0.1

"""
词表超参数
"""
# 源语言词表大小
src_vocab_size = 32000
# 目标词表大小
tgt_vocab_size = 32000
# 填充索引为0
padding_idx = 0
# 句子开始符索引位置
bos_idx = 2
# 句子结束符索引位置
eos_idx = 3


"""
训练配置参数
"""
# 训练批次大小
batch_size = 32
# 训练轮次
epoch_num = 3
# 学习率
lr = 3e-4


"""
路径配置
"""
data_dir = "./data"
train_data_path = "./data/dataset/train.json"
dev_data_path = "./data/dataset/dev.json"
test_data_path = "./data/dataset/test.json"
# 预训练权重 ：目前不存在
model_path = "./weights/transformer_model.pth"
test_model_path = "./weights/transformer_model.pth"


"""
设备配置
"""
gpu_id = "0"
device_id = [0]
# set device
if gpu_id != "":
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device("cpu")

print(device)

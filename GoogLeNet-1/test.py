import os

# 1. 强制开启离线模式（不再向远端发送任何请求，直接读本地缓存）
os.environ["HF_HUB_OFFLINE"] = "1"

from datasets import load_dataset

# 2. 读取本地缓存（注意：不要传 local_files_only 参数）
dataset = load_dataset(
    "microsoft/cats_vs_dogs", 
    split="train", 
    cache_dir="./data"
)

# 3. 划分 80% 训练集与 20% 验证集
split_data = dataset.train_test_split(test_size=0.2, seed=42)
train_set = split_data["train"]
val_set = split_data["test"]

print(f"训练集数量: {len(train_set)}，验证集数量: {len(val_set)}")
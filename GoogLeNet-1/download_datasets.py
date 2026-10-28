import os

# 1. 必须在导入 datasets 之前设置环境变量！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 之后再导入 datasets
from datasets import load_dataset

# 3. 开始下载并加载
dataset = load_dataset("microsoft/cats_vs_dogs", split="train", cache_dir="./cats_vs_dogs_cache")

print("数据集加载成功：", dataset)
sample = dataset[0]
print("第一张图标签:", sample["labels"])  # 0 为 cat, 1 为 dog
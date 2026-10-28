import copy
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"


import pandas as pd
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from model import GoogLeNet, Inception
from torch.utils import data
from torchvision import transforms

# 1. 顶层定义预处理 Transforms
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# 2. 必须定义在文件顶层（不要写在函数内部或用 lambda），这样多进程才能正常 pickle
def train_transform_batch(batch):
    batch["pixel_values"] = [
        train_transforms(img.convert("RGB")) for img in batch["image"]
    ]
    return batch


def val_transform_batch(batch):
    batch["pixel_values"] = [
        val_transforms(img.convert("RGB")) for img in batch["image"]
    ]
    return batch


def collate_fn(batch):
    images = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return images, labels


# 3. 业务函数内部直接传入顶层函数名
def train_val_data_process(batch_size=64, num_workers=4):
    dataset = load_dataset(
        "microsoft/cats_vs_dogs",
        split="train",
        cache_dir="./data",
    )

    split_data = dataset.train_test_split(test_size=0.2, seed=42)

    # 直接传顶层函数名，不使用 lambda
    train_data = split_data["train"].with_transform(train_transform_batch)
    val_data = split_data["test"].with_transform(val_transform_batch)

    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = data.DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


train_val_data_process()


def model_train(model, train_loader, val_loader, num_epochs=10):

    # 设置设备为 GPU（如果可用）或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 损失函数 # 交叉熵用于分类 # 均方差用于回归
    criterion = torch.nn.CrossEntropyLoss()

    # Aadm优化器 梯度下降用于反向传播更新模型参数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # LR学习率

    model = model.to(device)

    # 复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高精准度
    best_acc = 0.0

    # 训练集损失
    train_loss_all = []
    # 验证集损失
    val_loss_all = []

    # 训练集准确率
    train_acc_all = []
    # 验证集准确率
    val_acc_all = []

    # time
    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # 初始化参数
        # 训练损失值
        train_loss = 0.0
        # 训练准确率
        train_corrects = 0

        # 验证损失值
        val_loss = 0.0
        # 验证准确率
        val_corrects = 0

        # 训练样本数
        train_num = 0
        # 验证样本数
        val_num = 0

        # 对数据的每个批次丢进模型神经网络中
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train()  # 设置模型为训练模式

            output = model(b_x)  # 前向传播

            pre_label = torch.argmax(output, dim=1)  # 预测标签 argmax取的是索引

            loss = criterion(output, b_y)  # 计算损失值

            optimizer.zero_grad()  # 梯度清零

            loss.backward()  # 反向传播

            optimizer.step()  # 更新参数

            # 统计训练数据
            # 累加当前批次的训练损失值
            train_loss += loss.item() * b_x.size(0)  # 累加损失值

            # 累加当前批次的训练准确率
            train_corrects += torch.sum(
                pre_label == b_y.data
            )  # 拿预测下标跟真实标签对比，统计正确的个数

            train_num += b_x.size(0)

        # 验证过程
        for step, (b_x, b_y) in enumerate(val_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()  # 设置模型为验证模式

            output = model(b_x)  # 前向传播

            pre_label = torch.argmax(output, dim=1)  # 预测标签 argmax取的是索引

            loss = criterion(output, b_y)  # 计算损失值

            # 统计验证数据
            # 累加当前批次的验证损失值
            val_loss += loss.item() * b_x.size(0)  # 累加损失值

            # 累加当前批次的验证准确率
            val_corrects += torch.sum(
                pre_label == b_y.data
            )  # 拿预测下标跟真实标签对比，统计正确的个数

            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)

        train_acc_all.append(train_corrects.double().item() / train_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print(
            f"Epoch: {epoch + 1} Train Loss :{train_loss_all[-1]:.4f}  Train Acc: {train_acc_all[-1]:.4f}"
        )
        print(
            f"Epoch: {epoch + 1} Val Loss :{val_loss_all[-1]:.4f}  Val Acc: {val_acc_all[-1]:.4f}"
        )

        ## 寻找最高准确度
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]  # 复制当前最高准确度

            best_model_wts = copy.deepcopy(model.state_dict())  # 复制当前模型参数

        ## 训练耗时
        time_elapsed = time.time() - since
        print(f"训练和验证整体耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    ## 选择最优参数保存
    torch.save(best_model_wts, "./GoogLeNet_model.pth")

    train_process = pd.DataFrame(
        data={
            "epoch": list(range(1, num_epochs + 1)),
            "train_loss": train_loss_all,
            "val_loss": val_loss_all,
            "train_acc": train_acc_all,
            "val_acc": val_acc_all,
        }
    )

    return train_process


def plot_acc_loss(train_process):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        train_process["epoch"],
        train_process["train_loss"],
        "ro-",
        label="Train Loss",
    )
    plt.plot(
        train_process["epoch"],
        train_process["val_loss"],
        "bs-",
        label="Val Loss",
    )

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(
        train_process["epoch"],
        train_process["train_acc"],
        "ro-",
        label="Train Acc",
    )
    plt.plot(
        train_process["epoch"],
        train_process["val_acc"],
        "bs-",
        label="Val Acc",
    )

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    GoogLeNet = GoogLeNet(Inception)

    train_loader, val_loader = train_val_data_process()

    train_process = model_train(GoogLeNet, train_loader, val_loader, num_epochs=13)

    plot_acc_loss(train_process)

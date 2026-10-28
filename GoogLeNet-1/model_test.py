import os

# 1. 开启离线模式，避免重复连接远端
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from datasets import load_dataset
from model import GoogLeNet, Inception
from PIL import Image
from torch.utils import data
from torchvision import transforms

# 2. 顶层定义测试预处理（仅做缩放和归一化，严禁添加随机水平翻转等训练增强）
test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def test_transform_batch(batch):
    batch["pixel_values"] = [
        test_transforms(img.convert("RGB")) for img in batch["image"]
    ]
    return batch


def collate_fn(batch):
    images = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return images, labels


# 3. 测试集数据加载函数：提取固定 seed=42 切分出的 20% 测试集
def test_data_process(batch_size=64, num_workers=4):
    dataset = load_dataset(
        "microsoft/cats_vs_dogs",
        split="train",
        cache_dir="./data",
    )

    # 必须与训练时使用相同的 seed=42，确保取出的 test 部分完全一致
    split_data = dataset.train_test_split(test_size=0.2, seed=42)
    test_data = split_data["test"].with_transform(test_transform_batch)

    test_loader = data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,  # 测试阶段保持确定性顺序，无需打乱
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return test_loader


# 4. 全量测试评估函数
def test_model_process(model, test_loader, device):
    model.eval()
    test_correct = 0
    test_num = 0

    with torch.no_grad():
        for b_x, b_y in test_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)
            pred_index = torch.argmax(output, dim=1)

            test_correct += torch.sum(pred_index == b_y)
            test_num += b_x.size(0)

    test_acc = test_correct.double().item() / test_num
    print(f"Total Test Samples: {test_num}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_correct}/{test_num})")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型（注意：如果是二分类，检查你的模型入参是否为 num_classes=2）
    # 若你的 GoogLeNet 默认就是 2 分类，则直接传入 Inception 即可
    net = GoogLeNet(Inception).to(device)

    # 加载训练好的权重文件
    checkpoint_path = "./GoogLeNet_model.pth"
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded weights from {checkpoint_path}")

    # 1. 批量加载测试集评估全量准确率
    # test_loader = test_data_process(batch_size=64, num_workers=4)
    # test_model_process(net, test_loader, device)

    # # 2. 类别映射（0 为猫，1 为狗）
    classes = ["Cat", "Dog"]

    # # 3. 抽样展示前 100 张测试图片的预测结果
    # print("\n--- 抽样预测展示 ---")
    # net.eval()
    # count = 0
    # with torch.no_grad():
    #     for b_x, b_y in test_loader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)

    #         outputs = net(b_x)
    #         preds = torch.argmax(outputs, dim=1)

    #         for i in range(b_x.size(0)):
    #             pred_label = classes[preds[i].item()]
    #             true_label = classes[b_y[i].item()]
    #             mark = "✓" if pred_label == true_label else "✗"
    #             print(
    #                 f"Sample {count + 1:02d} [{mark}] -> Predicted: {pred_label:<4} | Actual: {true_label}"
    #             )
    #             count += 1
    #             if count >= 100:
    #                 break
    #         if count >= 100:
    #             break

    # image = Image.open("./imgs/28864261_235825728127_2.jpg")
    # image = Image.open("./imgs/32372279_115418394107_2.jpg")
    image = Image.open("./imgs/v2-93aba9c44bf90fbae5d983b9528b5f93_r.jpg")

    image = test_transforms(img=image)
    print(image.shape)
    image = image.unsqueeze(0)
    image = image.to(device)
    print(image.shape)

    with torch.no_grad():
        net.eval()
        output = net(image)
        pred_index = torch.argmax(output, dim=1)

        print("result: ", pred_index.item())

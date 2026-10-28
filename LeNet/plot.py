from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

train_data = FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
)

train_loader = data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)

for step, (b_x, b_y) in enumerate(train_loader):
    print(b_x.shape, b_y.shape)
    if step > 0:
        break

batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()
print(batch_x.shape, batch_y.shape)
class_labels = train_data.classes
print(class_labels)

# ------------------ Matplotlib 绘图部分 ------------------
plt.figure(figsize=(24, 10))


for ii in range(len(batch_y)):
    # 4 行 16 列，索引从 1 开始
    plt.subplot(4, 16, ii + 1)

    # 渲染灰度图像
    plt.imshow(batch_x[ii], cmap="gray")

    # 类别文字标题（字号设为 7~8 防止横向遮挡）
    plt.title(class_labels[batch_y[ii]], fontsize=7)

    # 关闭坐标刻度
    plt.axis("off")

# 调整子图间距，避免文字与图像贴得太紧
plt.subplots_adjust(wspace=0.1, hspace=0.35)
plt.show()

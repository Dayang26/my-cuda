import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Relu = nn.ReLU(inplace=True)

        self.c1 = nn.Conv2d(
            in_channels=1,  # 原始通道数
            out_channels=96,  # 输出通道数
            kernel_size=11,  # 卷积核大小5*5
            stride=4,  # 步幅
            padding=0,  # 填充
        )

        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.c2 = nn.Conv2d(
            in_channels=96,  # 原始通道数
            out_channels=256,  # 输出通道数
            kernel_size=5,  # 卷积核大小5*5
            stride=1,  # 步幅
            padding=2,  # 填充
        )

        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.c3 = nn.Conv2d(
            in_channels=256,  # 原始通道数
            out_channels=384,  # 输出通道数
            kernel_size=3,  # 卷积核大小3*3
            stride=1,  # 步幅
            padding=1,  # 填充
        )

        self.c4 = nn.Conv2d(
            in_channels=384,  # 原始通道数
            out_channels=384,  # 输出通道数
            kernel_size=3,  # 卷积核大小3*3
            stride=1,  # 步幅
            padding=1,  # 填充
        )
        self.c5 = nn.Conv2d(
            in_channels=384,  # 原始通道数
            out_channels=256,  # 输出通道数
            kernel_size=3,  # 卷积核大小3*3
            stride=1,  # 步幅
            padding=1,  # 填充
        )

        self.s3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.f1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.f2 = nn.Linear(in_features=4096, out_features=4096)
        self.f3 = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        x = self.Relu(self.c1(x))
        x = self.s1(x)

        x = self.Relu(self.c2(x))
        x = self.s2(x)

        x = self.Relu(self.c3(x))
        x = self.Relu(self.c4(x))
        x = self.Relu(self.c5(x))
        x = self.s3(x)
        x = self.flatten(x)

        x = F.dropout(self.Relu(self.f1(x)), p=0.5)
        x = F.dropout(self.Relu(self.f2(x)), p=0.5)
        x = self.f3(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNet().to(device)
    print(summary(model, (1, 227, 227)))

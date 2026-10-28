import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(
            in_channels=1,  # 原始通道数
            out_channels=6,  # 输出通道数
            kernel_size=5,  # 卷积核大小5*5
            stride=1,  # 步幅
            padding=2,  # 填充
        )
        self.sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.c2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        )
        self.s3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.f5 = nn.Linear(in_features=5 * 5 * 16, out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, xx):
        x = self.sigmoid(self.c1(xx))
        x1 = self.s2(x)

        x2 = self.sigmoid(self.c2(x1))
        x3 = self.s3(x2)

        x4 = self.flatten(x3)
        x5 = self.f5(x4)
        x6 = self.f6(x5)
        x7 = self.f7(x6)

        return x7


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))

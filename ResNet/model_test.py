import torch
from model import Residual, ResNet18
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import FashionMNIST


def test_data_process():
    test_data = FashionMNIST(
        root="../data",
        train=False,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(size=224), transforms.ToTensor()]
        ),
    )

    
    test_loader = data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=True,
        num_workers=16,
    )

    return test_loader


def test_model_process(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
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
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    ResNet18 = ResNet18(Residual)
    ResNet18.load_state_dict(torch.load("./ResNet18_model.pth"))

    test_loader = test_data_process()
    test_model_process(ResNet18, test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18.to(device)
    model.eval()

    # classes = [
    #     "T-shirt/top",
    #     "Trouser",
    #     "Pullover",
    #     "Dress",
    #     "Coat",
    #     "Sandal",
    #     "Shirt",
    #     "Sneaker",
    #     "Bag",
    #     "Ankle boot",
    # ]
    # with torch.no_grad():
    #     for step, (b_x, b_y) in enumerate(test_loader):
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)

    #         output = model(b_x)
    #         pred_index = torch.argmax(output, dim=1)

    #         print(
    #             f"Step {step}-> Predicted: {classes[pred_index.item()]}, Actual: {classes[b_y.item()]}"
    #         )

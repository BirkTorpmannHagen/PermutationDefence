import torch

from  StegoSafeModel import StegoSafeModel
import torchvision
from torchvision.models.resnet import ResNet34_Weights
from torchvision.datasets import ImageNet, ImageFolder
from copy import deepcopy
from torch.utils.data import DataLoader
from cdma import hide_secret_in_model, recover_secret_in_model


def check_functional_equality():
    device = "cuda"
    dataset = ImageFolder(root='~/Datasets/imagenette2/val', transform=torchvision.transforms.ToTensor())
    model1 = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval()
    model2 = deepcopy(model1).eval()
    model1 = StegoSafeModel(model1)

    for x,y in DataLoader(dataset, batch_size=1):
        x = x
        y = y
        out1 = model1(x)
        out2 =model2(x)
        assert torch.allclose(out1, out2, atol=1e-5), "Outputs are different!"
    print("Outputs are the same!")

def check_payload_integrity():
    device = "cuda"
    dataset = ImageFolder(root='~/Datasets/imagenette2/val', transform=torchvision.transforms.ToTensor())
    model = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval()
    with torch.no_grad():
        ld = hide_secret_in_model(model, "hello world")
        model2 = deepcopy(model).eval()
        safe_model = StegoSafeModel(model).eval()
        print(recover_secret_in_model(model2, ld))
        print(recover_secret_in_model(safe_model, ld))



if __name__ == '__main__':
    # check_functional_equality()
    check_payload_integrity()


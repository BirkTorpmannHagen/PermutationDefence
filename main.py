import torch

from  StegoSafeModel import StegoSafeModel
import torchvision
from torchvision.models.resnet import ResNet34_Weights
from torchvision.datasets import ImageNet, ImageFolder
from copy import deepcopy
from torch.utils.data import DataLoader


def eval_models():
    dataset = ImageFolder(root='~/Datasets/imagenette2/val', transform=torchvision.transforms.ToTensor())
    model1 = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval().to("cuda")
    model2 = deepcopy(model1).eval().to('cuda')
    model1 = StegoSafeModel(model1).to('cuda')
    for x,y in DataLoader(dataset, batch_size=1):
        x = x.to("cuda")
        y = y.to("cuda")
        out1 = model1(x)
        out2 =model2(x)
        print(torch.allclose(out1, out2, atol=1e-5), torch.argmax(out1), torch.argmax(out2))

eval_models()

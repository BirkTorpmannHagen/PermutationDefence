import torch

from  StegoSafeModel import StegoSafeModel
import torchvision
from torchvision.models.resnet import ResNet34_Weights
from torchvision.datasets import ImageNet, ImageFolder
from copy import deepcopy
from torch.utils.data import DataLoader
from maleficnet.injector import Injector
from maleficnet.extractor import Extractor
from maleficnet.models.densenet import DenseNet

def check_functional_equality(num_samples=10):
    device = "cuda"
    dataset = ImageFolder(root='~/Datasets/imagenette2/val', transform=torchvision.transforms.ToTensor())
    model1 = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval()
    model2 = deepcopy(model1).eval()
    model1 = StegoSafeModel(model1)

    for i, (x,y) in enumerate(DataLoader(dataset, batch_size=1)):
        if i>num_samples:
            break
        x = x
        y = y
        out1 = model1(x)
        out2 =model2(x)
        assert torch.allclose(out1, out2, atol=1e-5), "Outputs are different!"
    print("Outputs are the same!")

def check_payload_integrity(message):

    injector = Injector(seed=42,
                        device="cpu",
                        malware_payload=message,
                        chunk_factor=6)

    # Infect the system 🦠
    extractor = Extractor(seed=42,
                          device="cpu",
                          malware_length=len(injector.payload),
                          hash_length=len(injector.hash),
                          chunk_factor=6)
    model = DenseNet(32, 10, only_pretrained=True)
    new_model_sd, message_length, _, _ = injector.inject(model, 0.0009)
    model.load_state_dict(new_model_sd)
    safe_model = StegoSafeModel(deepcopy(model))
    hashequality_vanilla = extractor.extract(model, message_length)
    hashequaulity_safe = extractor.extract(safe_model, message_length)
    print(f"Vanilla Payload Recovered: {hashequality_vanilla}")
    print(f"Permuted model Payload Recovered: {hashequaulity_safe}")



if __name__ == '__main__':
    # check_functional_equality()
    check_payload_integrity("Hello, World! This could be malware! But it is not :)")


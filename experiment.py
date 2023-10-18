import math
from typing import List, Tuple, Dict
from pathlib import Path
from copy import deepcopy
import hashlib
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
import torch
from  StegoSafeModel import StegoSafeModel
import torchvision
from torchvision.models.resnet import ResNet34_Weights
from torchvision.datasets import ImageNet, ImageFolder
from copy import deepcopy
from torch.utils.data import DataLoader
from stego import FloatBinary, str_to_bits, bits_to_str, dummy_data_generator
from utils import check_model_equality, PermuteIterator

# How many bits (LSB) to use from the fraction (mantissa) of the float values
BITS_TO_USE = 16
assert BITS_TO_USE <= 23, "Can't be bigger then 23 bits"
DATA_FOLDER = "../../Datasets/imagenette2/val"
dataset = ImageFolder(root='~/Datasets/imagenette2/val', transform=torchvision.transforms.ToTensor())
model1 = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval().to("cuda")
model2 = deepcopy(model1).eval().to('cuda')
model1 = StegoSafeModel(model1).to('cuda')

IMAGES_TO_TEST_ON = list(map(str, Path(DATA_FOLDER).glob("**/*.jpg")))
layers_storage_capacity_mb: Dict[str, int] = {}

for i,l in enumerate(PermuteIterator(model1)):
    nb_params = np.prod(l.weight[0].shape)
    capacity_in_bytes = np.floor((nb_params * BITS_TO_USE) / 8).astype(int)
    layers_storage_capacity_mb[str(l)+str(i)] = capacity_in_bytes / float(1<<20)

print(layers_storage_capacity_mb)
selected_layers_weights = []
layer_names = list(layers_storage_capacity_mb.keys())

for n in layer_names:
    v = model.get_layer(n).weights[0].numpy().ravel()
    selected_layers_weights.extend(v)
selected_layers_weights = np.array(selected_layers_weights)
nb_values = len(selected_layers_weights)
overall_storage_capacity_bytes = nb_values * BITS_TO_USE / 8
overall_storage_capacity_mb = overall_storage_capacity_bytes // float(1<<20)


secret_to_hide = dummy_data_generator.generate_dummy_data(overall_storage_capacity_bytes)
secret_bits = str_to_bits(secret_to_hide)
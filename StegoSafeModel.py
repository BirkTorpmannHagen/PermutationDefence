import torch 
import torch.nn as nn
from copy import deepcopy
import torchvision
import matplotlib.pyplot as plt
from torchvision.models.resnet import BasicBlock, ResNet34_Weights
from utils import check_model_equality, PermuteIterator
from math import factorial as fac
class StegoSafeModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.permute_model()

    def permute(self, module, perm=None):
        with torch.no_grad():
            if perm is None:  # previous layer returns unpermuted activations
                perm = torch.randperm(module.weight.data.size()[0])
                if (perm == torch.arange(module.weight.data.size()[
                                             0])).all():  # if the permutation is the identity, we need to make sure it's not
                    return self.permute(module=module, perm=perm)

            perm_full = torch.arange(module.weight.data.size()[0])
            perm_full[:len(perm)] = perm
            # perm_full = perm

            module.weight.data = module.weight.data[perm_full]
            if module.bias is not None:
                module.bias.data = module.bias.data[perm_full]
            inv_perm = torch.argsort(perm)  # potential issue here?
            return inv_perm
    def permute_model(self):
        def create_permute_hook(pinv):
            # wild closure magic, chatgpt helped me with this
            def permute_output(module, input, output):
                permuted_channels = torch.index_select(output, 1, pinv)
                remaining_channels = output[:, len(pinv):]
                permuted_output = torch.cat((permuted_channels, remaining_channels), 1)
                out_set = set(output.unique().tolist())
                new_out_set = set(permuted_output.unique().tolist())
                return permuted_output

            return permute_output

        for module in PermuteIterator(self.model):  # only iterates over conv2d layers with stride 1
            module.register_forward_hook(create_permute_hook(self.permute(module)))


    def num_permutations(self):
        return sum([fac(m.weight.shape[1]) for m in PermuteIterator(self.model)])

    def forward(self, x):
        return self.model(x)




if __name__ == '__main__':
    # model1 = nn.Sequential(nn.Conv2d(3, 3 , (3,3))).eval()
    # model1 = nn.Sequential(nn.Conv2d(3, 6 , (3,3)), nn.BatchNorm2d(6), nn.ReLU(), nn.Conv2d(6, 6 , (3,3)), nn.BatchNorm2d(6), nn.ReLU()).eval()
    model1 = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval()
    # model1 = nn.Sequential(nn.Conv2d(3, 3 , (3,3)), nn.Conv2d(3, 6 , (3,3))).eval()
    model2 = deepcopy(model1).eval()
    model1 = StegoSafeModel(model1)
    import numpy as np
    print(model1.num_permutations())
    dummy = torch.rand(1, 3, 64,64)
    out = model1(dummy)
    # permute_model(model = model1)

    print("model equality:", check_model_equality(model1, model2))
    new_out = model1(dummy)
    print("output true equality: ", torch.allclose(out, new_out, atol=1e-5))

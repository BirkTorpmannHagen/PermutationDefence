import torch 
import torch.nn as nn
from copy import deepcopy
import torchvision
import matplotlib.pyplot as plt
from torchvision.models.resnet import BasicBlock, ResNet34_Weights
from utils import check_model_equality, Conv2DIterator, LinearIterator
from math import factorial as fac
class StegoSafeModel(nn.Module):
    """
    Permutes linear and convolutional layers. Note that linear layers are the building blocks in attention layers,
    a
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.permute_model()

    def permute_conv2d(self, module, perm=None):
        with torch.no_grad():
            if perm is None:  # can define permutations if needed
                perm = torch.randperm(module.weight.data.size()[0])
                if (perm == torch.arange(module.weight.data.size()[
                                             0])).all():  # if the permutation is the identity, we need to make sure it's not
                    return self.permute_conv2d(module=module, perm=perm)

            perm_full = torch.arange(module.weight.data.size()[0])
            perm_full[:len(perm)] = perm
            # perm_full = perm

            module.weight.data = module.weight.data[perm_full]
            if module.bias is not None:
                module.bias.data = module.bias.data[perm_full]
            inv_perm = torch.argsort(perm)  # potential issue here?
            return inv_perm.cuda() #returns the inverse permutation

    def permute_linear(self, module, perm=None):
        print(f"permuting linear layer w {module.weight.shape}")
        with torch.no_grad():
            if perm is None:  # can define permutations if needed
                perm = torch.randperm(module.weight.data.size()[1])
                if (perm == torch.arange(module.weight.data.size()[
                                             1])).all():  # if the permutation is the identity, we need to make sure it's not
                    return self.permute_linear(module=module, perm=perm)
            perm = torch.randperm(module.weight.shape[1])
            pinv = torch.argsort(perm)
            module.weight.data = module.weight[:, perm].data
            return pinv.cuda()  # returns the inverse permutation

    def permute_model(self):
        #aren't closures cool? :)
        def create_conv2d_permute_hook(pinv):
            def permute_output(module, inp, output):
                permuted_channels = torch.index_select(output, 1, pinv)
                remaining_channels = output[:, len(pinv):] # in cases where there are more channels in the next layer, only the first channels are permuted
                permuted_output = torch.cat((permuted_channels, remaining_channels), 1)
                return permuted_output
            return permute_output

        def create_linear_permute_hook(pinv):
            def permute_input(module, inp):
                transformed =  tuple(i[:, pinv] for i in inp)
                return transformed
            return permute_input

        for conv2d_module in Conv2DIterator(self.model):  # only iterates over conv2d layers with stride 1
            conv2d_module.register_forward_hook(create_conv2d_permute_hook(self.permute_conv2d(conv2d_module)))
        for linear_module in LinearIterator(self.model):
            linear_module.register_forward_pre_hook(create_linear_permute_hook(self.permute_linear(linear_module)))


    def num_permutations(self):
        return sum([fac(m.weight.shape[1]) for m in Conv2DIterator(self.model)]) + sum([fac(m.weight.shape[0]) for m in LinearIterator(self.model)])

    def forward(self, x):
        return self.model(x)




if __name__ == '__main__':
    # model1 = nn.Sequential(nn.Conv2d(3, 3 , (3,3))).eval()
    # model1 = nn.Sequential(nn.Conv2d(3, 6 , (3,3)), nn.BatchNorm2d(6), nn.ReLU(), nn.Conv2d(6, 6 , (3,3)), nn.BatchNorm2d(6), nn.ReLU()).eval()
    model1 = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval()
    model1 = model1.cuda()
    # model1 = nn.Sequential(nn.Conv2d(3, 3 , (3,3)), nn.Conv2d(3, 6 , (3,3))).eval()
    model2 = deepcopy(model1)
    model1 = StegoSafeModel(model1)
    import numpy as np
    dummy = torch.rand(1, 3, 64,64).cuda()
    out = model1(dummy)
    # permute_model(model = model1)

    print("model equality:", check_model_equality(model1, model2))
    new_out = model1(dummy)
    print("output true equality: ", torch.allclose(out, new_out, atol=1e-5))
    from transformers.models.bert import BertModel

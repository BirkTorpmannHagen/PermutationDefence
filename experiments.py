import torch 
import torch.nn as nn
from copy import deepcopy
import torchvision
import matplotlib.pyplot as plt
from torchvision.models.resnet import BasicBlock
from utils import check_model_equality, PermuteIterator, Permutor

def permute(module, perm=None):
    
    with torch.no_grad():
        if perm==None: # previous layer returns unpermuted activations
            perm = torch.randperm(module.weight.data.size()[0])
            if (perm == torch.arange(module.weight.data.size()[0])).all():
                return permute(module=module, perm=perm)
        module.weight.data[:len(perm)] = module.weight.data[perm]
        if module.bias is not None:
            module.bias.data[:len(perm)] = module.bias.data[perm]
        inv_perm = torch.argsort(perm)
        return inv_perm



def permute_model(model):
    def create_permute_hook(pinv):
        def permute_output(module, input, output):
            permuted_channels = torch.index_select(output, 1, pinv)
                  # Concatenate the permuted channels and the remaining channels
            remaining_channels = output[:, len(pinv):]
            permuted_output = torch.cat((permuted_channels, remaining_channels), 1)

            return permuted_output
        return permute_output


    """
    iterates over pairs of modules, permutes the weights, then reverses the permutation by adding a permutor module
    """
    permutable_modules = PermuteIterator(model)
    for module1, module2 in permutable_modules:
        pinv = permute(module1)
        print(pinv.shape)
        print(module1, module2)
        _ = permute(module2, pinv)
        module2.register_forward_hook(create_permute_hook(pinv)) #repermute output


    

if __name__ == '__main__':
    # model1 = nn.Sequential(nn.Conv2d(3, 6 , (3,3)), nn.BatchNorm2d(6), nn.ReLU()).eval()
    # model1 = torchvision.models.resnet34(pretrained=True).eval()
    model1 = nn.Sequential(nn.Conv2d(3, 3 , (3,3)), nn.Conv2d(3, 6 , (3,3))).eval()
    model2 = deepcopy(model1).eval()
    dummy = torch.rand(1, 3, 512, 512)
    out = model1(dummy)
    permute_model(model = model1)
    print("model equality:", check_model_equality(model1, model2))
    new_out = model1(dummy)
    out_set = set(out.unique().tolist())
    new_out_set = set(new_out.unique().tolist())

    print("output set equality: ", out_set == new_out_set)
    print(out.shape)
    print(new_out.shape)
    print("output true equality: ", (out == new_out).all())
    print(out.shape)
    print(new_out.shape)

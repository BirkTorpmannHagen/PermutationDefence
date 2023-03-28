import torch 
import torch.nn as nn
from copy import deepcopy
import torchvision
import matplotlib.pyplot as plt
from torchvision.models.resnet import BasicBlock
from utils import check_model_equality, PermuteIterator, Permutor

def permute(module, perm=None):
    with torch.no_grad():
        if perm is None:  # previous layer returns unpermuted activations
            perm = torch.randperm(module.weight.data.size()[0])
            if (perm == torch.arange(module.weight.data.size()[0])).all():
                return permute(module=module, perm=perm)
                
        perm_full = torch.arange(module.weight.data.size()[0])
        perm_full[:len(perm)] = perm
        
        module.weight.data = module.weight.data[perm_full]
        if module.bias is not None:
            module.bias.data = module.bias.data[perm_full]
        inv_perm = torch.argsort(perm) #potential issue here?
        return inv_perm



def permute_model(model):
    def create_permute_hook(pinv):
        #wild closure magic, chatgpt helped me with this
        def permute_output(module, input, output):
            permuted_channels = torch.index_select(output, 1, pinv)
            remaining_channels = output[:, len(pinv):]
            permuted_output = torch.cat((permuted_channels, remaining_channels), 1)
            out_set = set(output.unique().tolist())
            new_out_set = set(permuted_output.unique().tolist())
            return permuted_output
        return permute_output


    """
    iterates over pairs of modules, permutes the weights, then reverses the permutation by adding a permutor module
    """

    for module in PermuteIterator(model):
        pinv = permute(module)
        module.register_forward_hook(create_permute_hook(pinv))
    # permutable_modules = PermuteIterator(model)
    # for module1, module2 in permutable_modules:
    #     pinv = permute(module1)
    #     print(pinv.shape)
    #     print(module1, module2)
    #     _ = permute(module2, pinv)
    #     module2.register_forward_hook(create_permute_hook(pinv)) #repermute output


    

if __name__ == '__main__':
    model1 = nn.Sequential(nn.Conv2d(3, 3 , (3,3))).eval()
    # model1 = nn.Sequential(nn.Conv2d(3, 6 , (3,3)), nn.BatchNorm2d(6), nn.ReLU(), nn.Conv2d(6, 6 , (3,3)), nn.BatchNorm2d(6), nn.ReLU()).eval()    
    # model1 = torchvision.models.resnet34(pretrained=True).eval()
    # model1 = nn.Sequential(nn.Conv2d(3, 3 , (3,3)), nn.Conv2d(3, 6 , (3,3))).eval()
    model2 = deepcopy(model1).eval()
    dummy = torch.rand(1, 3, 32, 32)
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

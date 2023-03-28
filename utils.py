import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from torch.nn.modules.linear import Linear

def check_model_equality(model1, model2):
    for m1, m2 in zip(filter(check_permutable, model1.modules()), 
                      filter(check_permutable, model2.modules())):
        if (m1.weight != m2.weight).any():
            return False
    return True


class Permutor(nn.Module):
    """
    Permutes the input channels according to the permutation argument, a list of indexes
    """
    def __init__(self, permutation) -> None:
        super().__init__()
        self.permutation = permutation
    
    def forward(self, x):
        print("permutor forward")
        print(x)
        print(x[self.permutation])
        return x[self.permutation]

def check_permutable(module):
    try:
        module.weight
    except AttributeError:
        return False
    
    return not isinstance(module, nn.Sequential) and \
    not isinstance(module, nn.ModuleList) and \
    not isinstance(module, nn.ModuleDict) and \
    not isinstance(module, BasicBlock) and \
    not isinstance(module, Linear) and \
    module.weight is not None

class PermutePairIterator:
    def __init__(self, model):
        #todo this is ugly, but it works. fix it!
        module_list = model.modules()
        self.module_list = filter(check_permutable, module_list)

    def __next__(self):
        try:
            return self.module_list.__next__(), self.module_list.__next__()
        except StopIteration:
            raise StopIteration

    def __iter__(self):
        return self
    
class PermuteIterator:
    def __init__(self, model):
        #todo this is ugly, but it works. fix it!
        module_list = model.modules()
        self.module_list = filter(check_permutable, module_list)

    def __next__(self):
        try:
            return self.module_list.__next__()
        except StopIteration:
            raise StopIteration

    def __iter__(self):
        return self
    

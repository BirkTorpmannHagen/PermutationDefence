import torch
import numpy as np
from torchvision.models import resnet34
from stego import FloatBinary, str_to_bits, bits_to_str
import math
import torch.nn as nn
from utils import PermuteIterator
def flatten(el):
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res
def hide_secret_in_model(model, secret, BITS_TO_USE=4):
    """
    Hides the secret in the model's weights. The secret is hidden in the fraction (mantissa) of the float values.
    Returns a list of indices of the weights that were modified as well for ease of presentation.
    """
    last_index_used_in_layer_dict = {}

    secret = str_to_bits(secret)
    nb_vals_needed = math.ceil(len(secret) / BITS_TO_USE)
    # Variable which tracks the number of values changed so far (used to index the secret message bits)
    i = 0

    for n in PermuteIterator(model):
        # Check if we need more values to use to hide the secret, if not then we are done with modifying the layer's weights
        if i >= nb_vals_needed:
            print("finished!")
            break

        w = n.weight
        w_shape = w.shape
        w = w.ravel()

        nb_params_in_layer: int = np.prod(w_shape)

        for j in range(nb_params_in_layer):
            # Chunk of data from the secret to hide
            _from_index = i * BITS_TO_USE
            _to_index = _from_index + BITS_TO_USE
            bits_to_hide = secret[_from_index:_to_index]
            bits_to_hide = list(map(bool, bits_to_hide))

            # Modify the defined bits of the float value fraction
            x = FloatBinary(w[j])
            fraction_modified = list(x.fraction)
            if len(bits_to_hide) > 0:
                fraction_modified[-BITS_TO_USE:] = bits_to_hide

            x_modified = x.modify_clone(fraction=tuple(fraction_modified))
            w[j] = x_modified.v # v is the float value of the modified binary float

            i += 1

            # Check if we need more values to use to hide the secret in the current layer, if not then we are done
            if i >= nb_vals_needed:
                break
        last_index_used_in_layer_dict[str(n)] = j
        w = w.reshape(w_shape)
        n.weight = nn.Parameter(w)
        print(f"Layer {n} is processed, last index modified: {j}")
    return last_index_used_in_layer_dict

def recover_secret_in_model(model, last_index_used_in_layer_dict, BITS_TO_USE=4):
    hidden_data = []

    for idx, n in enumerate(PermuteIterator(model)):
        if str(n) not in last_index_used_in_layer_dict.keys():
            continue

        # Check if the layer was used in hiding the secret or not (e.g.: we could hide the secret in the prev. layers)
        # We could get the modified weights directly from the model: model.get_layer(n).get_weights()...
        w = n.weight
        w_shape = w.shape
        w = w.ravel()
        nb_params_in_layer: int = np.prod(w_shape)

        for i in range(last_index_used_in_layer_dict[str(n)] + 1):
            x = FloatBinary(w[i])
            hidden_bits = x.fraction[-BITS_TO_USE:]
            hidden_data.extend(hidden_bits)
        print(f"Layer {n} is processed, bits are extracted")
    return bits_to_str(list(map(int, hidden_data)))
def generate_spreading_code(length):
    code = np.random.randint(0, 2, length)
    # Ensure the code has a balanced number of 0s and 1s
    while abs(np.sum(code) - length // 2) > length // 10:
        code = np.random.randint(0, 2, length)
    return code
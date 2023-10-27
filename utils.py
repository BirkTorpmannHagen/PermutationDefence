import struct
from typing import Union, List

import bitstring
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from torch.nn.modules.linear import Linear

def check_model_equality(model1, model2):
    for m1, m2 in zip(filter(check_permutable, model1.modules()), 
                      filter(check_permutable, model2.modules())):
        if (m1.weight != m2.weight).any():
            return False
    return True



def check_permutable(module):
    return isinstance(module, nn.Conv2d)
    # try:
    #     module.weight
    # except AttributeError:
    #     return False
    #
    # return not isinstance(module, nn.Sequential) and \
    # not isinstance(module, nn.ModuleList) and \
    # not isinstance(module, nn.ModuleDict) and \
    # not isinstance(module, BasicBlock) and \
    # not isinstance(module, Linear) and \
    # module.weight is not None

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


def str_to_bits(s: str, as_list: bool = False) -> Union[List[int], List[List[int]]]:
    tmp = []
    for b in bytes(s, "ascii"):
        s_bin = bin(b)[2:].rjust(8, "0")
        tmp.append(s_bin)
    if as_list:
        return [list(map(int, list(x))) for x in tmp]
    return [int(x) for x in "".join(tmp)]


def bits_to_str(b: List[int]) -> str:
    tmp = []
    for i in range(0, len(b), 8):
        c = chr(int("".join(map(str, b[i:i + 8])), 2))
        tmp.append(c)
    return "".join(tmp)


def float_to_bits(f):
    s = struct.pack('>f', f)
    return struct.unpack('>l', s)[0]


def bits_to_float(b):
    s = struct.pack('>l', b)
    return struct.unpack('>f', s)[0]


def set_lsb(b, bit):
    """
    :param b: Bit sequence to modify its least significant bit (LSB), ex: float_to_bits(69.0)
    :param bit: 0/1
    :return: Bit sequence after its LSB was modified
    """
    return (b & ~1) | bit


def get_lsb(b):
    """
    :param b: the bit representation of a floating point number ex: float_to_bits(69.0)
    :return: the least significant bit of the floating point number
    """
    return b & 1


def modify_bit(n, p, b):
    """
    :param n: bit representation of the number you want to modify (call float_to_bits(n))
    :param p: position in which you want to modify the bit [index 0 the rightmost, index 31 the leftmost]
    :param b: bit value (0/1)
    :return:
    """
    mask = 1 << p
    return (n & ~mask) | ((b << p) & mask)


def bits_from_bytes(bytes_list):
    """
    :param bytes_list: A List of bytes
    :return: A List of bits
    """
    bits = list()
    for b in bytes_list:
        for i in reversed(range(0, 8)):
            bits.append((b >> i) & 1)
    return bits


def bits_to_bytes(bits_list):
    """
    :param bits_list: A List of bits
    :return: A byte list
    """
    byte_list = list()
    index = 0
    while index < len(bits_list):
        byte = 0
        for j in range(0, 8):
            bit = bits_list[index + j]
            byte = (byte << 1) | bit
        index += 8
        byte_list.append(byte)

    return byte_list


def bits_from_file(input_file):
    """
    :param input_file: file path
    :return: A List containing the bits of the file
    """
    with open(input_file, 'rb') as fp:
        bytes_list = (b for b in fp.read())
        bits = bits_from_bytes(bytes_list)
    return bits


def bits_to_file(output_file, bits_list):
    """
    :param output_file: the file path where the file will be saved
    :param bits_list: list of bits, which will get converted to bytes and written to file
    :return: Nothing
    """
    bytes_list = bits_to_bytes(bits_list)
    with open(output_file, 'wb') as fp:
        fp.write(bytes(bytes_list))


def get_byte(fl, byte_position):
    """
    :param fl: floating point number
    :param byte_position: from left to right: 0-most significant byte, 3-least significant byte
    :return: string representation of the byte requested
    """
    f = bitstring.BitArray(float=fl, length=32)
    return f.bin[8 * byte_position:8 * byte_position + 8]

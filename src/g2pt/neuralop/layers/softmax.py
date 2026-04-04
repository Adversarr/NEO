from torch import nn
from importlib.util import find_spec

def get_softmax() -> nn.Module:
    """"""
    has_liger = find_spec("liger_kernel") is not None
    if has_liger:
        from liger_kernel.transformers import LigerSoftmax
        return LigerSoftmax()
    else:
        return nn.Softmax(dim=-1)

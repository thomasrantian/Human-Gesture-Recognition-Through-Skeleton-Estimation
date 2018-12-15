"""
some external functions from github
"""
from packages_lib import *

def get_loss_funcs():
    def _eucl_loss(x, y):
        return K.sum(K.square(x - y)) / 10 / 2

    losses = {}
    losses["weight_stage1_L1"] = _eucl_loss
    losses["weight_stage1_L2"] = _eucl_loss
    losses["weight_stage2_L1"] = _eucl_loss
    losses["weight_stage2_L2"] = _eucl_loss
    losses["weight_stage3_L1"] = _eucl_loss
    losses["weight_stage3_L2"] = _eucl_loss
    losses["weight_stage4_L1"] = _eucl_loss
    losses["weight_stage4_L2"] = _eucl_loss
    losses["weight_stage5_L1"] = _eucl_loss
    losses["weight_stage5_L2"] = _eucl_loss
    losses["weight_stage6_L1"] = _eucl_loss
    losses["weight_stage6_L2"] = _eucl_loss

    return losses
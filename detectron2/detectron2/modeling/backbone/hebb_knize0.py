'''
This script is meant to be analogous to other backbone scripts, such as resnet.py. It will be used in conjunction with the "meta architecture" of RCNN. 
A generalized R-CNN model (according to detectron2) is any model with the following three components:
    1. Per image feature extraction (TODO for the Hebb model)
    2. Region proposal generation (Something to take from RCNN and implement directly into our model)
    3. Per-region feature extraction and prediction (Another modular item to take directly from RCNN)

Essentially, I'll be trying to put together a Frankenstein script combining detectron2/modeling/backbone/resnet.py and hebbnet here. Much of the structure will
be copied directly from resnet.py, and I will adjust as needed.     
'''

# start with keeping the same imports as resnet.py
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

# __all__ = [
#     "ResNetBlockBase",
#     "BasicBlock",
#     "BottleneckBlock",
#     "DeformBottleneckBlock",
#     "BasicStem",
#     "ResNet",
#     "make_stage",
#     "build_resnet_backbone",
# ]
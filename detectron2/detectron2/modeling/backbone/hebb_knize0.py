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

__all__ = [
    "HebbNet",
    "HebbRuleWithActivationThreshold"
    "build_hebbnet_backbone"
    # "makestage"
]

# TODO: Adjust to class Backbone
class HebbNet(Backbone):
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hebbian_weights = nn.Linear(input_layer_size, hidden_layer_size, False)
        self.classification_weights = nn.Linear(hidden_layer_size, output_layer_size, True)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

        # initialize dictionary of outputs along forward pass layers / steps
        out_features = ["hebb_output"]
        self._out_feature_channels = {"hebb_output" : 9999}
        self._out_features = out_features


    def forward(self, x):
        x = self.flatten(x)
        z = self.hebbian_weights(x)
        z = self.relu(z)  # Apply ReLU activation after the Hebbian layer
        pred = self.classification_weights(z)
        pred = self.softmax(pred)
        outputs = {} # initialize output dict
        outputs['hebb_output'] = z # TODO fix hard-coding
        return outputs # TODO unsure about this. may need other outputs
    
class HebbRuleWithActivationThreshold(nn.Module):
    def __init__(self, hidden_layer_size=2000, input_layer_size=784):
        super().__init__()
        self.register_buffer('w1_activation_thresholds', torch.zeros((hidden_layer_size, input_layer_size)))
        self.t = 1

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        with torch.no_grad():
          activation = torch.matmul(z.T,x) #(2000,784) - matrix multipication

          if self.t==1:
              delta_w1 = activation
          else:
              delta_w1 = activation - (self.w1_activation_thresholds / (self.t-1))

          self.w1_activation_thresholds += activation
          self.t += 1
          return delta_w1
        
# returns delta_w1 after applying the gradient sparsity
def gradiant_sparsity(delta_w1, p, device):
  # Calculate the number of values to keep based on the percentile (p)
  num_values_to_keep = int( p * delta_w1.numel())

  # Find the top k values and their indices
  top_values, _ = torch.topk(torch.abs(delta_w1).view(-1), num_values_to_keep)
  threshold = top_values[-1]  # The threshold is the k-th largest value

  # Set values below the threshold to zero
  delta_w1 = torch.where(torch.abs(delta_w1) >= threshold, delta_w1, torch.tensor(0.0).to(device))
  return delta_w1
        
# TODO: currently doing nothing; this might not be necessary (see ViT)
@BACKBONE_REGISTRY.register()
def build_hebbnet_backbone(cfg, input_shape):
    """
    Create a HebbNet instance from config.

    Returns:
        HebbNet: a :class:`HebbNet` instance.
    """
    
    hidden_layer_size = 2000 # TODO cheating... don't feel like figuring out cfg right now
    output_layer_size = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    # return HebbNet(input_shape, hidden_layer_size, output_layer_size) # temporarily hard-code to see further into debugger execution
    return HebbNet(1024, hidden_layer_size, output_layer_size)

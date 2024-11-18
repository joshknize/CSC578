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
from typing import Dict

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
    "build_hebbnet_backbone",
    "gradiant_sparsity"
]


# TODO: Adjust to class Backbone
class HebbNet(Backbone):
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, cfg):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hebbian_weights = nn.Linear(input_layer_size, hidden_layer_size, False)
        self.classification_weights = nn.Linear(hidden_layer_size, output_layer_size, True)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

        # set necessary dictionary for region-based modular components of RCNN
            # unsure if these are still necessary after I changed from `_BASE_: "../Base-RCNN-FPN.yaml"` to the simpler "../Base-RCNN-C4.yaml"
        self._out_feature_strides = {"res4": 4} # TODO 16 is arbitrary hard-code; pave
        self._out_feature_channels = {"res4": cfg.MODEL.ROI_BOX_HEAD.FC_DIM} # TODO arbitrary hard-code for 1024; pave???

        # initialize dictionary of outputs along forward pass layers / steps
        out_features = ["res4"]
        self._out_features = out_features

        # initialize input_layer_size so it can be used by padding_constraints to specify our fixed image size to generalized rcnn
        self.input_layer_size = input_layer_size


    def forward(self, x):
        x = self.flatten(x)
        z = self.hebbian_weights(x)
        width = np.sqrt(self.input_layer_size / 3)
        features = z.clone().reshape(1, self._out_feature_channels['res4'], int(width), int(width)) # TODO: hard-coded
        z = self.relu(z)  # Apply ReLU activation after the Hebbian layer
        pred = self.classification_weights(z)
        pred = self.softmax(pred)
        # create output dict; TODO fix hard-coding
        outputs = {
            "res4": features
        } 
        return outputs # TODO unsure about this. may need other outputs
    
    @property
    def padding_constraints(self) -> Dict[str, int]:
            """
            This property is a generalization of size_divisibility. Some backbones and training
            recipes require specific padding constraints, such as enforcing divisibility by a specific
            integer (e.g., FPN) or padding to a square (e.g., ViTDet with large-scale jitter
            in :paper:vitdet). `padding_constraints` contains these optional items like:
            {
                "size_divisibility": int,
                "square_size": int,
                # Future options are possible
            }
            `size_divisibility` will read from here if presented and `square_size` indicates the
            square padding size if `square_size` > 0.

            TODO: use type of Dict[str, int] to avoid torchscipt issues. The type of padding_constraints
            could be generalized as TypedDict (Python 3.8+) to support more types in the future.
            """
            square_size = np.sqrt(self.input_layer_size / 3) # get the square image size

            return {"square_size": square_size}
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    
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
    
    input_layer_size = cfg.INPUT.MAX_SIZE_TRAIN * cfg.INPUT.MAX_SIZE_TRAIN * 3
    hidden_layer_size = (cfg.MODEL.ROI_BOX_HEAD.FC_DIM * input_layer_size) / 3 # TODO the box_features (ROI section) wants 1024 channels; trying this for drilling
    output_layer_size = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    # return HebbNet(input_shape, hidden_layer_size, output_layer_size) # temporarily hard-code to see further into debugger execution
    return HebbNet(input_layer_size, int(hidden_layer_size), output_layer_size, cfg)

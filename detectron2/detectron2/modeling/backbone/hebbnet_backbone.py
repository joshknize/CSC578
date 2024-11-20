# start with keeping the same imports as resnet.py
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict
import matplotlib.pyplot as plt

from detectron2.layers import (
    ShapeSpec,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "HebbNet",
    "HebbRuleWithActivationThreshold"
    "build_hebbnet_backbone",
    "gradiant_sparsity"
]


class HebbNet(Backbone):
    def __init__(self, cfg):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=(5, 5), stride=2, padding=0)       # Based on testing, doing MaxPool and then adaptive pooling to keep a shape of 80x80 consistent
        self.adaptive_pool = nn.AdaptiveAvgPool2d((cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN))
        self.cfg = cfg

        self.lr = cfg.MODEL.HEBB_LR

        self.layers = nn.ModuleList()
        self.input_layer_size = cfg.INPUT.MAX_SIZE_TRAIN * cfg.INPUT.MAX_SIZE_TRAIN * 3
        self.flattened_size = self.adaptive_pool.output_size[0] * self.adaptive_pool.output_size[1] * 3
        self.hidden_layer_size = (cfg.MODEL.ROI_BOX_HEAD.FC_DIM * self.flattened_size) // (3 * cfg.MODEL.NUM_HIDDEN)
        self.flatten = nn.Flatten()    
        self.layers.append(nn.Linear(self.input_layer_size, self.hidden_layer_size, False))    
        for i in range(self.cfg.MODEL.NUM_HIDDEN-1):
            self.layers.append(nn.Linear(self.hidden_layer_size, self.hidden_layer_size, False))
        # no output layer necessary
        # self.classification_weights = nn.Linear(self.hidden_layer_size, output_layer_size, True)

        self.activation_threshold_layers = []
        self.activation_threshold_layers.append(HebbRuleWithActivationThreshold(hidden_layer_size=self.hidden_layer_size,
                                                                input_layer_size=self.flattened_size).to(self.cfg.MODEL.DEVICE))
        for i in range(self.cfg.MODEL.NUM_HIDDEN-1):
            self.activation_threshold_layers.append(HebbRuleWithActivationThreshold(hidden_layer_size=self.hidden_layer_size,
                                                                input_layer_size=self.hidden_layer_size).to(self.cfg.MODEL.DEVICE))

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

        self._out_feature_strides = {"res4": 4}
        self._out_feature_channels = {"res4": cfg.MODEL.ROI_BOX_HEAD.FC_DIM}

        # initialize dictionary of outputs along forward pass layers / steps
        out_features = ["res4"]
        self._out_features = out_features

        # initialize input_layer_size so it can be used by padding_constraints to specify our fixed image size to generalized rcnn
        # self.input_layer_size = input_layer_size

        # for img and feature visualization
        self.img_vis = cfg.MODEL.IMG_VIS
        self.feat_vis = cfg.MODEL.FEAT_VIS
        self.feat_vis_num = cfg.MODEL.FEAT_VIS_NUM


    def forward(self, x):
        if self.img_vis:
            print('Original image:')
            visualize_image(x)

        # x = self.max_pool(x)
        # x = self.adaptive_pool(x)

        features = []

        # initial flatten
        x = self.flatten(x)

        for layer, activation_layer in zip(self.layers, self.activation_threshold_layers):
            z = layer(x)
            width = np.sqrt(self.flattened_size / 3)
            # reshape each flattened hebbian weight tensor to CHW, where C is total fully connected dim channels divided by # of hidden layers
            features.append(z.clone().reshape(int((self.cfg.MODEL.ROI_BOX_HEAD.FC_DIM / self.cfg.MODEL.NUM_HIDDEN)), int(width), int(width)))

            # activation
            z = self.relu(z)

            # update weights
            self.hebbian_update(x, z, activation_layer, layer)
            x = z

        # create output dict
        features = torch.cat(features, dim=0).unsqueeze(0)
        outputs = {
            "res4": features
        }

        if self.feat_vis:
            print('Feature map visualization: ' + f'{self.feat_vis_num}')
            features_vis = features.clone().squeeze(0)
            plt.imshow(features_vis[self.feat_vis_num].cpu().detach().numpy(), cmap='viridis')
            plt.axis('off')
            plt.show()

        return outputs
    
    def hebbian_update(self, x, z, activation_thresholder, layer):
        delta_w1 = activation_thresholder(x, z)

        # Gradient sparsity
        # delta_w1 = gradiant_sparsity(delta_w1, 0.95, self.cfg.MODEL.DEVICE)

        # update hebbian weights
        layer.weight.data = layer.weight.data - self.lr*delta_w1
        # self.lr = self.lr * 0.999

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
          activation = torch.matmul(z.T,x)

          if self.t==1:
              delta_w1 = activation
          else:
              delta_w1 = activation - (self.w1_activation_thresholds / (self.t-1))

          self.w1_activation_thresholds += activation
          self.t += 1
          # after one epoch, increase the learning rate
          if (self.t % 2000 == 0):
              self.lr = self.lr * 10
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

def visualize_image(img):
    if img.dim() < 3: 
        img = img.clone().squeeze(0).cpu().numpy()
        plt.plot(img)
    else:
        img = img.clone().squeeze(0).permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
    plt.axis('off')
    plt.show()
        
@BACKBONE_REGISTRY.register()
def build_hebbnet_backbone1(cfg, input_shape):
    """
    Create a HebbNet instance from config.

    Returns:
        HebbNet: a :class:`HebbNet` instance.
    """
    
    input_layer_size = cfg.INPUT.MAX_SIZE_TRAIN * cfg.INPUT.MAX_SIZE_TRAIN * 3
    hidden_layer_size = (cfg.MODEL.ROI_BOX_HEAD.FC_DIM * input_layer_size) / (3  * cfg.MODEL.NUM_HIDDEN) 
    output_layer_size = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    return HebbNet(cfg)

import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers
import torch


class VerbGivenObjectAppearanceConstants(io.JsonSerializableClass):
    def __init__(self):
        super(VerbGivenObjectAppearanceConstants,self).__init__()
        self.appearance_feat_size = 1024

    @property
    def mlp_const(self):
        factor_const = {
            'in_dim': self.appearance_feat_size,
            'layer_units': [512],
            'activation': 'ReLU',
            'use_bn': True
        }
        return factor_const


class VerbGivenObjectAppearance(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(VerbGivenObjectAppearance,self).__init__()
        self.const = copy.deepcopy(const)
        self.mlp = pytorch_layers.create_mlp(self.const.mlp_const)
        
    def forward(self,feats):
        in_feat = feats['object_rcnn']

        
        
        factor_scores = self.mlp(in_feat)
        return factor_scores

    

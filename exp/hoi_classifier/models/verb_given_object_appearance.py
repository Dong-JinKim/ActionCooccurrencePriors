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
        self.num_verbs = 117

    @property
    def mlp_const(self):
        factor_const = {
            'in_dim': self.appearance_feat_size,
            'out_dim': self.num_verbs,
            'emb_dim': 300,#------!!!!
            'out_activation': 'Identity',
            'layer_units': [512],#[int(self.appearance_feat_size)], #[512],#
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True
        }
        return factor_const


class VerbGivenObjectAppearance(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(VerbGivenObjectAppearance,self).__init__()
        self.const = copy.deepcopy(const)
        self.mlp = pytorch_layers.create_mlp(self.const.mlp_const)
        
    def forward(self,feats):
        # original
        in_feat = feats['object_rcnn']
        # sum
        #in_feat = feats['object_rcnn']+feats['global_feat']#---------!!!!!!!
        # concat
        #in_feat = torch.cat((feats['object_rcnn'],feats['global_feat']),1)#---------!!!!!!!
        
        
        factor_scores, embedding = self.mlp(in_feat) #-----!!!!
        return factor_scores, embedding #-----!!!!

    

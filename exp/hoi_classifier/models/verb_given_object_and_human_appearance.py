import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers
import torch


class VerbGivenObjectAndHumanAppearanceConstants(io.JsonSerializableClass):
    def __init__(self):
        super(VerbGivenObjectAndHumanAppearanceConstants,self).__init__()
        self.appearance_feat_size = 2048
        self.num_verbs = 117

    @property
    def mlp_const(self):
        factor_const = {
            'in_dim': self.appearance_feat_size*2,
            'out_dim': self.num_verbs,
            'emb_dim': 300,
            'out_activation': 'Identity',
            'layer_units': [int(self.appearance_feat_size/8)], # [512],##-----!!!!![512],#
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True
        }
        return factor_const

class MLRN(nn.Module):
    def __init__(self):
        super(MLRN,self).__init__()
        emb_size = 16#---------!!!!!!
        
        self.Wa = nn.Sequential(nn.Linear(2048,emb_size),nn.ReLU(inplace=True))
        self.Wb = nn.Sequential(nn.Linear(2048,emb_size),nn.ReLU(inplace=True))
        self.Wp = nn.Sequential(nn.Linear(emb_size,117))
        self.embedding = nn.Linear(emb_size,300)

    def forward(self,feats):
        
        feat_a = self.Wa(feats['human_rcnn'] )
        feat_b = self.Wb(feats['object_rcnn'])
        
        feat_c = torch.mul(feat_a,feat_b)
        
        x1 = self.Wp(feat_c)
        x2 = self.embedding(feat_c)
        
        return x1,x2

class VerbGivenObjectAndHumanAppearance(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(VerbGivenObjectAndHumanAppearance,self).__init__()
        ## (1) For concat
        self.const = copy.deepcopy(const)
        self.mlp = pytorch_layers.create_mlp(self.const.mlp_const)
        
        ## (2) For bilinear pooling
        #self.mlp = MLRN()
        
    def forward(self,feats):
        ## (1) For concat
        in_feat = torch.cat((feats['human_rcnn'],feats['object_rcnn']),1)
        factor_scores, embedding = self.mlp(in_feat) #-----!!!!
        
        ## (2) For bilinear pooling
        #factor_scores, embedding = self.mlp(feats) #-----!!!!
        
        return factor_scores, embedding #-----!!!!

    

import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class VerbGivenAllConstants(io.JsonSerializableClass):
    def __init__(self):
        super(VerbGivenAllConstants,self).__init__()
        self.box_feat_size = 21
        self.num_objects = 300#80
        self.num_verbs = 117
        self.use_object_label = True
        self.use_log_feat = True
        
        self.pose_feat_size = 54+90
        self.use_absolute_pose = True
        self.use_relative_pose = True

        self.appearance_feat_size = 2048
        

    @property
    def mlp_const(self):
        in_dim = 2*self.appearance_feat_size + self.num_objects + 2*self.box_feat_size + 2*self.pose_feat_size 
        layer_units = [512]*2
        factor_const = {
            'in_dim': in_dim,
            'out_dim': self.num_verbs,
            'emb_dim': 300,#------!!!!
            'out_activation': 'Identity',
            'layer_units': layer_units,
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True
        }
        return factor_const
    
    
class VerbGivenAll(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(VerbGivenAll,self).__init__()
        self.const = copy.deepcopy(const)
        self.mlp = pytorch_layers.create_mlp(self.const.mlp_const)

        
    def transform_feat_box(self,feat):
        if self.const.use_log_feat is True:
            log_feat = torch.log(torch.abs(feat)+1e-6)
        else:
            log_feat = 0*feat
        transformed_feat = torch.cat((feat,log_feat),1) 
        return transformed_feat
    
    def transform_feat_pose(self,feat):
        log_feat = torch.log(torch.abs(feat)+1e-6)
        transformed_feat = torch.cat((feat,log_feat),1) 
        return transformed_feat
        
    def forward(self,feats):
        ## pose
        if not self.const.use_absolute_pose:
            absolute_pose = 0*feats['absolute_pose']
        else:
            absolute_pose = feats['absolute_pose']
        if not self.const.use_relative_pose:
            relative_pose = 0*feats['relative_pose']
        else:
            relative_pose = feats['relative_pose']
        pose_feats = torch.cat((absolute_pose,relative_pose),1)
        transformed_pose_feats = self.transform_feat_pose(pose_feats)
        ## obj label and box
        transformed_box_feats = self.transform_feat_box(feats['box'])
        if self.const.use_object_label is True:
            object_label = feats['object_one_hot']
        else:
            object_label = 0*feats['object_one_hot']
        
        
        
        
        in_feat = torch.cat((feats['human_rcnn'],feats['object_rcnn'],object_label,transformed_box_feats,transformed_pose_feats,),1)
        factor_scores, embedding = self.mlp(in_feat) #-----!!!!
        return factor_scores, embedding #-----!!!!

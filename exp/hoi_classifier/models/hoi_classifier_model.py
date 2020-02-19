import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers
from exp.hoi_classifier.models.verb_given_object_appearance import \
    VerbGivenObjectAppearanceConstants, VerbGivenObjectAppearance
from exp.hoi_classifier.models.verb_given_human_appearance import \
    VerbGivenHumanAppearanceConstants, VerbGivenHumanAppearance    
from exp.hoi_classifier.models.verb_given_boxes_and_object_label import \
    VerbGivenBoxesAndObjectLabelConstants, VerbGivenBoxesAndObjectLabel
from exp.hoi_classifier.models.verb_given_human_pose import \
    VerbGivenHumanPoseConstants, VerbGivenHumanPose
from exp.hoi_classifier.models.scatter_verbs_to_hois import \
    ScatterVerbsToHoisConstants, ScatterVerbsToHois   
    
import pdb

class MTL(nn.Module):
    def __init__(self):
        super(MTL,self).__init__()
        input_size = 512
        num_cluster = 32#45+1
        
        self.classifier = nn.Linear(input_size,117*num_cluster)# 117 or 600------------------(1) For MoE
        #self.classifier = nn.Linear(input_size,(117-num_cluster+2)*num_cluster)# 117 or 600-----(2) For Anchor
        
        self.embedding = nn.Linear(input_size,num_cluster)
    def forward(self,feats):
        
        output2 = nn.functional.softmax(self.embedding(feats))

        
        output1 = self.classifier(feats)# B*(117*G)
        return output1,output2

class ScatterClusterToHois(nn.Module):
    def __init__(self,json_file):
        super(ScatterClusterToHois,self).__init__()
        self.gid2verb = io.load_json_object(json_file) 
        
    def forward(self,group_scores):
        verb_scores = group_scores[:,self.gid2verb]

        return verb_scores
          
class HoiClassifierConstants(io.JsonSerializableClass):
    FACTOR_NAME_TO_MODULE_CONSTANTS = {
        'verb_given_object_app': VerbGivenObjectAppearanceConstants(),
        'verb_given_human_app': VerbGivenHumanAppearanceConstants(),
        'verb_given_boxes_and_object_label': VerbGivenBoxesAndObjectLabelConstants(),
        'verb_given_human_pose': VerbGivenHumanPoseConstants(),
    }

    def __init__(self):
        super(HoiClassifierConstants,self).__init__()
        self.verb_given_appearance = True
        self.verb_given_human_appearance = True
        self.verb_given_object_appearance = True
        self.verb_given_boxes_and_object_label = True
        self.verb_given_human_pose = True
        self.rcnn_det_prob = True
        self.use_object_label = True
        self.use_log_feat = True
        self.scatter_verbs_to_hois = ScatterVerbsToHoisConstants()

    @property
    def selected_factor_constants(self):
        factor_constants = {}
        for factor_name in self.selected_factor_names:
            const = self.FACTOR_NAME_TO_MODULE_CONSTANTS[factor_name]
            factor_constants[factor_name] = const
        return factor_constants

    @property
    def selected_factor_names(self): 
        factor_names = []
        if self.verb_given_appearance:
            factor_names.append('verb_given_object_app')
            factor_names.append('verb_given_human_app')
        elif self.verb_given_human_appearance:
            factor_names.append('verb_given_human_app')
        elif self.verb_given_object_appearance:
            factor_names.append('verb_given_object_app')
        if self.verb_given_boxes_and_object_label:
            factor_names.append('verb_given_boxes_and_object_label')
        if self.verb_given_human_pose:
            factor_names.append('verb_given_human_pose')
        return factor_names


class HoiClassifier(nn.Module,io.WritableToFile):
    FACTOR_NAME_TO_MODULE = {
        'verb_given_object_app': VerbGivenObjectAppearance,
        'verb_given_human_app': VerbGivenHumanAppearance,
        'verb_given_boxes_and_object_label': VerbGivenBoxesAndObjectLabel,
        'verb_given_human_pose': VerbGivenHumanPose,
    }

    def __init__(self,const):
        super(HoiClassifier,self).__init__()
        self.const = copy.deepcopy(const)

        self.USE_cluster = True
                
        self.FC = MTL()
        self.sigmoid = pytorch_layers.get_activation('Sigmoid')


        self.scatter_verbs_to_hois = ScatterVerbsToHois(
            self.const.scatter_verbs_to_hois)
        for name, const in self.const.selected_factor_constants.items():
            self.create_factor(name,const)

    def create_factor(self,factor_name,factor_const):
        if factor_name in ['verb_given_boxes_and_object_label','verb_given_human_pose']:
            factor_const.use_object_label = self.const.use_object_label
        if factor_name in ['verb_given_boxes_and_object_label']:
            factor_const.use_log_feat = self.const.use_log_feat
        factor = self.FACTOR_NAME_TO_MODULE[factor_name](factor_const)
        setattr(self,factor_name,factor)

    def forward(self,feats):
        factor_scores = {}
        embedding = {}
        any_verb_factor = False
        verb_factor_scores = 0
        for factor_name in self.const.selected_factor_names:
            module = getattr(self,factor_name)
            factor_scores[factor_name] = module(feats)
            if 'verb_given' in factor_name:
                any_verb_factor = True
                verb_factor_scores += factor_scores[factor_name]
        
        verb_factor_scores = verb_factor_scores/len(self.const.selected_factor_names)
 
        verb_factor_scores,embedding = self.FC(verb_factor_scores)

        
        if any_verb_factor:
            verb_prob = self.sigmoid(verb_factor_scores)
            
            if self.USE_cluster:
                NUM_of_CLUSTER = 32#45+1
                #self.scatter_cluster_to_hois = ScatterClusterToHois(f'anchor_only/gid2cid{NUM_of_CLUSTER-1}.json')#-------(*) only for anchors!!!
                cluster_weight= embedding
                cluster_weight = cluster_weight.unsqueeze(1)# B,1,G
                
                #verb_prob = verb_prob.reshape(-1,NUM_of_CLUSTER,117-NUM_of_CLUSTER+2)# B,G,117#-(1) for anchors 
                verb_prob = verb_prob.reshape(-1,NUM_of_CLUSTER,117)# B,G,117#------------------(2) for MoE
                
                verb_prob = torch.bmm(cluster_weight,verb_prob).squeeze()
                #verb_prob = self.scatter_cluster_to_hois(torch.cat((embedding[:,1:-1],verb_prob),1))#-------(*) only for anchors!!!
            assert(verb_prob.shape[1]==117)

            verb_prob_vec = self.scatter_verbs_to_hois(verb_prob)
  
        else:
            verb_prob_vec = 0*feats['human_prob_vec'] + 1
        
        if self.const.rcnn_det_prob:
            human_prob_vec = feats['human_prob_vec']
            object_prob_vec = feats['object_prob_vec']
        else:
            human_prob_vec = 0*feats['human_prob_vec'] + 1
            object_prob_vec = 0*feats['object_prob_vec'] + 1

        prob_vec = {
            'human': human_prob_vec,
            'object': object_prob_vec,
            'verb': verb_prob_vec,
        }

        prob_vec['hoi'] = \
            prob_vec['human'] * \
            prob_vec['object'] * \
            prob_vec['verb']
        
        return prob_vec, factor_scores,embedding

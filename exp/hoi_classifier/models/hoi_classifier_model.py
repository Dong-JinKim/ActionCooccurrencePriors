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
from exp.hoi_classifier.models.verb_given_all import \
    VerbGivenAllConstants, VerbGivenAll
from exp.hoi_classifier.models.verb_given_global import \
    VerbGivenGlobalConstants, VerbGivenGlobal
from exp.hoi_classifier.models.scatter_verbs_to_hois import \
    ScatterVerbsToHoisConstants, ScatterVerbsToHois
from exp.hoi_classifier.models.verb_given_object_and_human_appearance import \
    VerbGivenObjectAndHumanAppearanceConstants, VerbGivenObjectAndHumanAppearance    
    
import pdb

class MTL(nn.Module):
    def __init__(self):
        super(MTL,self).__init__()
        input_size = 512
        num_cluster = 32#45+1
        
        self.classifier = nn.Linear(input_size,117*num_cluster)# 117 or 600------------------(1) For MoE
        #self.classifier = nn.Linear(input_size,(117-num_cluster+2)*num_cluster)# 117 or 600-----(2) For Anchor
        
        self.embedding = nn.Linear(input_size,num_cluster)
        ##self.embedding2 = nn.Linear(num_cluster,input_size)
    def forward(self,feats):
        
        output2 = nn.functional.softmax(self.embedding(feats))
        
        ##output22 = self.embedding2(output2)
        ##feats = torch.cat((feats,output22),1)
        
        output1 = self.classifier(feats)# B*(117*G)
        return output1,output2
'''
class CLS_Emb(nn.Module):
    def __init__(self,prototype_name):# prototype_name = 'verb_prototype.pkl'
        super(CLS_Emb,self).__init__()
        input_size = 512
        self.classifier = nn.Sequential(
                            #nn.Linear(input_size,300),#(600)or (300)
                            #nn.ReLU(inplace=True),
                            )
        cls_emb = torch.load(prototype_name)   
        
        
        self.cls_emb = cls_emb.cuda()#------!!!! (1)
        
        #self.cls_emb = nn.Linear(300,117,bias=False)#-----!!! (2) (600,600)or (300,117)
        #self.cls_emb.weight = nn.Parameter(cls_emb.t())
    
    
    def forward(self,feats):
        
        output1 = self.classifier(feats)
        
        output = torch.mm(output1,self.cls_emb) # (B*300)*(300*117) = (B*117) ------(1)
        
        #output = self.cls_emb(output1)#-----(2)
        
        return output
'''
class ScatterClusterToHois(nn.Module):
    def __init__(self,json_file):
        super(ScatterClusterToHois,self).__init__()
        self.gid2verb = io.load_json_object(json_file) 
        
    def forward(self,group_scores):
        #batch_size, num_verbs = group_scores.size()
        #verb_scores = Variable(torch.zeros(batch_size,117)).cuda()
        #for source_idx in range(117):
        #    target_idx = self.gid2verb[str(source_idx)]
        #    verb_scores[:,target_idx] = group_scores[:,source_idx]
        
        
        #pdb.set_trace()
        #tmp = [ii[1] for ii in self.gid2verb.items()]
        #group_scores[:,tmp]
        verb_scores = group_scores[:,self.gid2verb]

        return verb_scores
        
'''      
class Refiner(nn.Module):
    def __init__(self):
        super(Refiner,self).__init__()
        co_occurrence= torch.load('co-occurrence_verb.pkl')   
        #self.co_occurrence = nn.Linear(117,117)
        #with torch.no_grad():
        #    self.co_occurrence.weight= nn.Parameter(torch.Tensor(co_occurrence))
        self.co_occurrence = torch.cuda.FloatTensor(co_occurrence)
        
        self.softmax = nn.Softmax(dim=1)
    def forward(self,feats,one_hot):
        
        co_occurrence = torch.matmul(one_hot,self.co_occurrence.reshape(80,-1))
        co_occurrence = co_occurrence.reshape(-1,117,117)# (B,80)*(80,117,117)= (B*117,117)
        output = torch.bmm(self.softmax(feats).unsqueeze(1),co_occurrence)
        #output = self.co_occurrence(feats)
        return output.squeeze()
'''
        
        
'''     
class AttPool(nn.Module):
    def __init__(self):
        super(AttPool,self).__init__()
        self.output_size = 512
        
        self.Wa = nn.Conv1d(self.output_size,117,1)
        
        self.Wb = nn.Conv1d(self.output_size,1,1)#original attpoool !!! (1)
        #self.Wb = nn.Linear(self.output_size*4,4)#new attpoool !!!  (2)
        
        
        
        ## nonlinear version.
        #self.Wa = nn.Sequential(nn.Conv1d(output_size,117,1),nn.ReLU(inplace=True))
        #self.Wb = nn.Sequential(nn.Conv1d(output_size,1,1),nn.ReLU(inplace=True))
        
    def forward(self,feats):
        #feats: B*D*4
        feat_a = self.Wa(feats)#B*117*4
        
        feat_b = self.Wb(feats).squeeze()#B*4 #  original attpool !!! (1)
        #feat_b = self.Wb(feats.view(-1,self.output_size*4)).squeeze()#B*4 # new att poool !!!! (2)
        
        ## with softmax?
        feat_b = nn.functional.softmax(feat_b,dim=1)
        
        att = feat_b.unsqueeze(2)# attention map B*4*1
        
        output = torch.bmm(feat_a,att).squeeze() # (B*117*4)*(B*4*1) ->  B*177
        #output = feat_a.mean(2)
        return output, att.squeeze()
'''       
'''
class RelationalEmbedding(nn.Module):
    def __init__(self):
        super(RelationalEmbedding,self).__init__()
        feat_size = 117
        emb_size = 64#---------!!!!!!
        
        self.Wa = nn.Sequential(nn.Linear(feat_size,emb_size),nn.ReLU(inplace=True))
        self.Wb = nn.Sequential(nn.Linear(feat_size,emb_size),nn.ReLU(inplace=True))
        self.Wx = nn.Sequential(nn.Linear(feat_size,emb_size),nn.ReLU(inplace=True))
        self.Wz = nn.Sequential(nn.Linear(emb_size,feat_size),nn.ReLU(inplace=True))
        

    def forward(self,feats):
        feat_a = self.Wa(feats) # B*d
        feat_b = self.Wb(feats).t() # d*B
        feat_x = self.Wx(feats) # B*d
        
        RR = torch.mm(feat_a,feat_b) # (B*d)*(d*B) = (B*B)
        
        RR = torch.softmax(RR,dim=1)
        
        AA = torch.mm(RR,feat_x) # (B*B) * (B*d) = B*d
        
        output = self.Wz(AA) # B*D
        
        return output + feats    
'''    
'''
class RelationalEmbedding(nn.Module):
    def __init__(self):
        super(RelationalEmbedding,self).__init__()
        self.mlp = nn.Sequential(nn.Linear(117,117),\
                                #nn.BatchNorm1d(117),\
                                #nn.ReLU(inplace=True),\
                                )
    def forward(self,feats):
        output = self.mlp(feats)
        return output + feats
'''
'''
class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        input_size =117
        output_size=117
        self.weight = nn.Sequential(\
                                nn.BatchNorm1d(input_size),\
                                nn.ReLU(inplace=True),\
                                #nn.Conv1d(input_size,output_size,1),\
                                #nn.ReLU(inplace=True),\
                                nn.Conv1d(output_size,1,1),\
                                )
        self.softmax = nn.Softmax(dim=1) 
        
    def forward(self,feats):
        #feats: B*D*4
        logit = self.weight(feats).squeeze()#B*4
        att = self.softmax(logit).unsqueeze(2)# attention map B*4*1
        output = torch.bmm(feats,att).squeeze()#B*D

        return output ,att.squeeze()
'''
'''
class Attention_emb(nn.Module):
    def __init__(self):
        super(Attention_emb,self).__init__()
        self.weight = nn.Conv1d(16,1,1)
        self.softmax = nn.Softmax(dim=1) 
        
    def forward(self,feats,emb):
        #feats: B*D*4
        logit = self.weight(emb).squeeze()#B*4
        att = self.softmax(logit).unsqueeze(2)# attention map B*4*1
        output = torch.bmm(feats,att).squeeze()#B*D

        return output  , att.squeeze()
'''        
class HoiClassifierConstants(io.JsonSerializableClass):
    FACTOR_NAME_TO_MODULE_CONSTANTS = {
        'verb_given_object_app': VerbGivenObjectAppearanceConstants(),
        'verb_given_human_app': VerbGivenHumanAppearanceConstants(),
        'verb_given_boxes_and_object_label': VerbGivenBoxesAndObjectLabelConstants(),
        'verb_given_human_pose': VerbGivenHumanPoseConstants(),
        'verb_given_global': VerbGivenGlobalConstants(),
        'verb_given_all': VerbGivenAllConstants(),
        'verb_given_object_and_human_app': VerbGivenObjectAndHumanAppearanceConstants(),
        
    }

    def __init__(self):
        super(HoiClassifierConstants,self).__init__()
        self.verb_given_appearance = True
        self.verb_given_human_appearance = True
        self.verb_given_object_appearance = True
        self.verb_given_boxes_and_object_label = True
        self.verb_given_human_pose = True
        self.verb_given_global = False
        self.verb_given_all = False
        self.verb_given_object_and_human_appearance = False
        self.rcnn_det_prob = True
        self.use_prob_mask = False
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
        elif self.verb_given_object_and_human_appearance:
            factor_names.append('verb_given_object_and_human_app')
        elif self.verb_given_human_appearance:
            factor_names.append('verb_given_human_app')
        elif self.verb_given_object_appearance:
            factor_names.append('verb_given_object_app')

        if self.verb_given_boxes_and_object_label:
            factor_names.append('verb_given_boxes_and_object_label')
        
        if self.verb_given_human_pose:
            factor_names.append('verb_given_human_pose')
        
        if self.verb_given_global:
            factor_names.append('verb_given_global')
        if self.verb_given_all:
            factor_names.append('verb_given_all')
        
        return factor_names


class HoiClassifier(nn.Module,io.WritableToFile):
    FACTOR_NAME_TO_MODULE = {
        'verb_given_object_app': VerbGivenObjectAppearance,
        'verb_given_human_app': VerbGivenHumanAppearance,
        'verb_given_boxes_and_object_label': VerbGivenBoxesAndObjectLabel,
        'verb_given_human_pose': VerbGivenHumanPose,
        'verb_given_global': VerbGivenGlobal,
        'verb_given_all': VerbGivenAll,
        'verb_given_object_and_human_app':VerbGivenObjectAndHumanAppearance,
    }

    def __init__(self,const):
        super(HoiClassifier,self).__init__()
        self.const = copy.deepcopy(const)
        
        
        self.USE_ATT = False
        self.AVE = True
        
        self.USE_REM=False
        self.USE_FC = True#-------------------------------------------------------------!!!!!
        self.USE_ATTpool = False#-------------------------------------------------------------!!!!!
        self.USE_Refine = False
        self.USE_VERB_Emb = False #----------!!!!
        self.USE_OBJ_Emb = False #---------!!!!
        self.USE_NIS = False# 
        self.USE_LIS = False
        self.USE_cluster = True
        self.USE_scatter = True
        
        self.USE_softmax = False
        
        if self.USE_FC:
            self.FC = MTL()
        elif self.USE_VERB_Emb:
            self.FC_verb = CLS_Emb('verb_prototype_word2vec.pkl')# hoi_prototype or verb_prototype or verb_prototype_word2vec
            if self.USE_OBJ_Emb:
                self.FC_obj = CLS_Emb('obj_prototype.pkl')
            
        if self.USE_REM:
            self.REM =  RelationalEmbedding()
        if self.USE_ATT:
            #self.ATT = Attention_emb()
            self.ATT = Attention()
        if self.USE_ATTpool:
            self.ATT=AttPool()
        if self.USE_Refine:
            self.Refiner = Refiner()
        if self.USE_softmax:
            self.sigmoid = nn.Softmax(dim=1)
        else:
            self.sigmoid = pytorch_layers.get_activation('Sigmoid')

        
        if self.USE_scatter:
            self.scatter_verbs_to_hois = ScatterVerbsToHois(
                self.const.scatter_verbs_to_hois)
        for name, const in self.const.selected_factor_constants.items():
            self.create_factor(name,const)

    def create_factor(self,factor_name,factor_const):
        if factor_name in ['verb_given_boxes_and_object_label','verb_given_human_pose','verb_given_all']:
            factor_const.use_object_label = self.const.use_object_label
        if factor_name in ['verb_given_boxes_and_object_label','verb_given_all']:
            factor_const.use_log_feat = self.const.use_log_feat
        factor = self.FACTOR_NAME_TO_MODULE[factor_name](factor_const)
        setattr(self,factor_name,factor)

    def forward(self,feats):
        factor_scores = {}
        embedding = {}#----------!!!!!
        any_verb_factor = False
        if self.USE_ATT or self.USE_ATTpool:#---!!!!!
            verb_factor_scores = []
            embeddings = []
        else:
            verb_factor_scores = 0
        for factor_name in self.const.selected_factor_names:
            module = getattr(self,factor_name)
            factor_scores[factor_name],embedding[factor_name] = module(feats)
            if 'verb_given' in factor_name:
                any_verb_factor = True
                if self.USE_ATT or self.USE_ATTpool:#---------------------------!!!!!!!!
                    verb_factor_scores.append(factor_scores[factor_name])
                    embeddings.append(embedding[factor_name])
                else:
                    verb_factor_scores += factor_scores[factor_name]
                    
                    
                    
                    ##------!!!!! only for inception
                    #factor_scores[factor_name] = self.scatter_verbs_to_hois(torch.sigmoid(factor_scores[factor_name]))
                    #-----!!!!!!!!!
        
        
        if self.USE_ATT or self.USE_ATTpool:
            
            verb_factor_scores = torch.stack(verb_factor_scores,2)
            embeddings = torch.stack(embeddings,2)
            
            #verb_factor_scores,_ = self.ATT(verb_factor_scores,embeddings)
            verb_factor_scores,att= self.ATT(verb_factor_scores)
            #pdb.set_trace()
        
        if self.AVE:
            verb_factor_scores = verb_factor_scores/len(self.const.selected_factor_names)
        
        if self.USE_REM:
            verb_factor_scores = self.REM(verb_factor_scores)
            
        if self.USE_FC:
            verb_factor_scores,embedding = self.FC(verb_factor_scores)
            #verb_factor_scores = self.FC(verb_factor_scores)
            
            #for interactiveness#
            if self.USE_NIS:
                embedding = self.sigmoid(embedding)
        
        elif self.USE_VERB_Emb:
            if self.USE_OBJ_Emb:
                embedding = self.sigmoid(self.FC_obj(verb_factor_scores))
            verb_factor_scores = self.FC_verb(verb_factor_scores)
            
                
        if self.USE_Refine:
            verb_factor_scores = self.Refiner(verb_factor_scores,feats['object_one_hot2'])
        
        
        if any_verb_factor:
            if self.USE_Refine:
                verb_prob = verb_factor_scores
            else:
                verb_prob = self.sigmoid(verb_factor_scores)
           
            if self.USE_NIS:
                verb_prob = verb_prob*embedding
            
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

            if self.USE_scatter:
                verb_prob_vec = self.scatter_verbs_to_hois(verb_prob)
            else:
                verb_prob_vec = verb_prob
                    
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

        if self.USE_LIS:
            prob_vec['hoi'] = \
                (prob_vec['human']**2) * \
                (prob_vec['object']**2) * \
                prob_vec['verb']
        else:
            prob_vec['hoi'] = \
                prob_vec['human'] * \
                prob_vec['object'] * \
                prob_vec['verb']
        
        if self.const.use_prob_mask:
            prob_vec['hoi'] = prob_vec['hoi'] * feats['prob_mask']
        
        return prob_vec, factor_scores,embedding#-------!!!!

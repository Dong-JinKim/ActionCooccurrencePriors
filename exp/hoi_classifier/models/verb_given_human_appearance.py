import torch

from exp.hoi_classifier.models.verb_given_object_appearance import \
    VerbGivenObjectAppearanceConstants, VerbGivenObjectAppearance


class VerbGivenHumanAppearanceConstants(VerbGivenObjectAppearanceConstants):
    def __init__(self):
        super(VerbGivenHumanAppearanceConstants,self).__init__()


class VerbGivenHumanAppearance(VerbGivenObjectAppearance):
    def __init__(self,const):
        super(VerbGivenHumanAppearance,self).__init__(const)

    def forward(self,feats):
        # original
        in_feat = feats['human_rcnn']
        # sum
        #in_feat = feats['human_rcnn']+feats['global_feat']#---------!!!!!!!
        # concat
        #in_feat = torch.cat((feats['human_rcnn'],feats['global_feat']),1)#---------!!!!!!!

        factor_scores, embedding = self.mlp(in_feat) #-----!!!!
        return factor_scores, embedding #-----!!!!


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
        
        in_feat = feats['human_rcnn']

        factor_scores = self.mlp(in_feat)
        return factor_scores


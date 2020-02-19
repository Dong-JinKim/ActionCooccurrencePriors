import os
import h5py
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tensorboard_logger import configure, log_value

import utils.io as io
from utils.model import Model
from utils.constants import save_constants
from exp.hoi_classifier.models.hoi_classifier_model import HoiClassifier
from exp.hoi_classifier.data.features_dataset import Features
import pdb


USE_refine = False # ACP projection for testing phase

def eval_model(model,dataset,exp_const):
    print('Creating hdf5 file for predicted hoi dets ...')
    pred_hoi_dets_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'pred_hoi_dets_{dataset.const.subset}_{model.const.model_num}.hdf5')
    pred_hois = h5py.File(pred_hoi_dets_hdf5,'w')
    model.hoi_classifier.eval()
    sampler = SequentialSampler(dataset)

    Word2Vec= torch.load('Word2Vec_Glove.pkl') 
    word2index=io.load_json_object('word2vec_vocab_Glove.json') 
    print('Word2Vec model Loaded!')
    
    if USE_refine:
        co_occurrence= torch.load('co-occurrence2.pkl')   
        co_occurrence = torch.cuda.FloatTensor(co_occurrence)
        
        co_occurrence_neg= torch.load('co-occurrence_neg.pkl')   
        co_occurrence_neg = torch.cuda.FloatTensor(co_occurrence_neg)
        
        mask = torch.eye(600,600).byte().cuda()# keep the diagonal to be 1.0
        co_occurrence.masked_fill_(mask,1.0)
        
    for sample_id in tqdm(sampler):
        data = dataset[sample_id]

        GT_obj = [Word2Vec(torch.LongTensor([word2index[dataset.hoi_dict[dd]['object']]])).sum(0) for dd in data['hoi_id']]
                
        GT_obj = torch.stack(GT_obj,0).cuda()

        
        with torch.no_grad():
            feats = {
                'human_rcnn': Variable(torch.cuda.FloatTensor(data['human_feat'])),
                'object_rcnn': Variable(torch.cuda.FloatTensor(data['object_feat'])),
                'box': Variable(torch.cuda.FloatTensor(data['box_feat'])),
                'absolute_pose': Variable(torch.cuda.FloatTensor(data['absolute_pose'])),
                'relative_pose': Variable(torch.cuda.FloatTensor(data['relative_pose'])),
                'human_prob_vec': Variable(torch.cuda.FloatTensor(data['human_prob_vec'])),
                'object_prob_vec': Variable(torch.cuda.FloatTensor(data['object_prob_vec'])),
                'prob_mask': Variable(torch.cuda.FloatTensor(data['prob_mask']))
            }        

            feats['object_one_hot'] = GT_obj
            feats['object_one_hot2'] = Variable(torch.cuda.FloatTensor(data['object_one_hot']))

                    
        prob_vec, factor_scores,_ = model.hoi_classifier(feats)
        
        hoi_prob = prob_vec['hoi'] #  B*600
        
        if USE_refine:
            ww = (hoi_prob>0).float()
            hoi_prob = ((torch.matmul(hoi_prob,co_occurrence)*0.9 + torch.matmul((1-hoi_prob)*ww,co_occurrence_neg)*0.1).t()/ww.sum(1)).t()

        
        hoi_prob = hoi_prob * feats['prob_mask'] #  B*600
        
        hoi_prob = hoi_prob.data.cpu().numpy()
        
        num_cand = hoi_prob.shape[0]# 1~B
        scores = hoi_prob[np.arange(num_cand),np.array(data['hoi_idx'])]
        

        human_obj_boxes_scores = np.concatenate((
            data['human_box'],
            data['object_box'],
            np.expand_dims(scores,1)),1)

        global_id = data['global_id']
        pred_hois.create_group(global_id)
        pred_hois[global_id].create_dataset(
            'human_obj_boxes_scores',
            data=human_obj_boxes_scores)
        pred_hois[global_id].create_dataset(
            'start_end_ids',
            data=data['start_end_ids_'])

    pred_hois.close()


def main(exp_const,data_const,model_const):
    print('Loading model ...')
    model = Model()
    model.const = model_const
    model.hoi_classifier = HoiClassifier(model.const.hoi_classifier).cuda()
    if model.const.model_num == -1:
        print('No pretrained model will be loaded since model_num is set to -1')
    else:
        model.hoi_classifier.load_state_dict(
            torch.load(model.const.hoi_classifier.model_pth))

    print('Creating data loader ...')
    dataset = Features(data_const)

    eval_model(model,dataset,exp_const)


    
    

    

import os
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from tensorboard_logger import configure, log_value
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import random

import utils.io as io
import utils.losses as losses
from utils.model import Model
from utils.constants import save_constants
from exp.hoi_classifier.models.hoi_classifier_model import HoiClassifier
from exp.hoi_classifier.data.features_dataset import Features


import pdb


def train_model(model,dataset_train,dataset_val,exp_const):

    hoi2gid = io.load_json_object('hoi2gid1.json') 
    #hoi2gid = io.load_json_object('anchor_only/hoi2anchor45.json') 
    #hoi2gid = io.load_json_object('anchor+others/hoi2anchor10.json') 
     
    Word2Vec= torch.load('Word2Vec_Glove.pkl') 
    word2index=io.load_json_object('word2vec_vocab_Glove.json') 
    neighbors = io.load_json_object('neighbor_object_Glove.json') 
 
        
    #probability = [0.2, 0.175, 0.150, 0.125, 0.1, 0.1, 0.075, 0.05, 0.025, 0.0]#----generalize2
    #probability = [0.3, 0.25, 0.2, 0.15, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]#----generalize3 / Glove_genearalize1
    #probability = [0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]#----generalize4 / Glove_genearalize2
    probability = [0.5, 0.3, 0.2,0,0,0,0,0,0,0]#----Glove_genearalize3
    
    print('Word2Vec model Loaded!')
    
    co_occurrence = torch.load('co-occurrence_pos.pkl') #------------for co-occurrence prior
    co_occurrence_neg = torch.load('co-occurrence_neg.pkl') #------------for negative co-occurrence prior
    
    co_occurrence = torch.cuda.FloatTensor(co_occurrence)
    co_occurrence_neg = torch.cuda.FloatTensor(co_occurrence_neg)


    params = itertools.chain(
    model.hoi_classifier.parameters()
    )
    optimizer = optim.Adam(params,lr=exp_const.lr)
    
    criterion = nn.BCELoss(reduction='none')
    criterion_softmax = nn.CrossEntropyLoss()
    
    step = 0
    optimizer.zero_grad()
    for epoch in range(exp_const.num_epochs):
        sampler = RandomSampler(dataset_train)
        for i, sample_id in enumerate(sampler):
            data = dataset_train[sample_id]
          
            ind = np.random.choice(10,1,p=probability)[0] 
            GT_obj = [Word2Vec(torch.LongTensor([word2index[ neighbors[dataset_train.hoi_dict[dd]['object']][ind][0]  ]])) for dd in data['hoi_id']]#-----generalized version! (weighted)
            GT_obj = torch.cat(GT_obj,0).cuda()

            
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

                
            model.hoi_classifier.train()
            prob_vec, factor_scores,embedding = model.hoi_classifier(feats)

            hoi_prob2 = prob_vec['hoi']
            hoi_prob = prob_vec['hoi'] * feats['prob_mask']

            
            hoi_labels = Variable(torch.cuda.FloatTensor(data['hoi_label_vec']))                
            
            
            loss_cls = criterion(hoi_prob,hoi_labels).mean()


            cluster_label = [hoi2gid[str(dd)] for dd in data['hoi_idx']]*data['hoi_label']
            cluster_label = Variable(torch.cuda.LongTensor(cluster_label))
            loss_cluster=criterion_softmax(embedding,cluster_label)
            

            new_label1 = torch.matmul(torch.cuda.FloatTensor(data['hoi_label_vec']),co_occurrence)
            weight = torch.cuda.FloatTensor((new_label1>0).float())
            
            new_label1_neg = torch.matmul((1-torch.cuda.FloatTensor(data['hoi_label_vec']))*weight,co_occurrence_neg)
            new_label1_neg = (new_label1_neg.t()/(weight.sum(1)-1+1e-8)).t()

            prior_labels1 = Variable(new_label1*0.9+new_label1_neg*0.1)

            loss_distillation = torch.mul(criterion(hoi_prob2,prior_labels1),weight).mean()
            
                    
            loss = 0.7 * loss_cls  +  0.3 * loss_distillation  +  0.1* loss_cluster

            

            loss.backward()
            if step%exp_const.imgs_per_batch==0:
                optimizer.step()
                optimizer.zero_grad()

            max_prob = hoi_prob.max().data
            max_prob_tp = torch.max(hoi_prob*hoi_labels).data

            if step%20==0:
                num_tp = np.sum(data['hoi_label'])
                num_fp = data['hoi_label'].shape[0]-num_tp
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | ' + \
                    'Train Loss: {:.8f} | TPs: {} | FPs: {} | ' + \
                    'Max TP Prob: {:.8f} | Max Prob: {:.8f}'
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    loss.data,
                    num_tp,
                    num_fp,
                    max_prob_tp,
                    max_prob)
                print(log_str)
            
            if step%5000==0:
                val_loss = eval_model(model,dataset_val,exp_const,Word2Vec,word2index,num_samples=2500)
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | Val Loss: {:.8f}'
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    loss.data,
                    val_loss)
                print(log_str)
                
            if step%5000==0:
                hoi_classifier_pth = os.path.join(
                    exp_const.model_dir,
                    f'hoi_classifier_{step}')
                torch.save(
                    model.hoi_classifier.state_dict(),
                    hoi_classifier_pth)

            step += 1


def eval_model(model,dataset,exp_const,Word2Vec,word2index,num_samples):
    model.hoi_classifier.eval()
    criterion = nn.BCELoss()
    step = 0
    val_loss = 0
    count = 0
    sampler = RandomSampler(dataset)
    torch.manual_seed(0)
    for sample_id in tqdm(sampler):
        if step==num_samples:
            break

        data = dataset[sample_id]
        
        
        GT_obj = [Word2Vec(torch.LongTensor([word2index[dataset.hoi_dict[dd]['object']]])).sum(0) for dd in data['hoi_id']]
        GT_obj = torch.stack(GT_obj,0).cuda()
        
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
        
        hoi_prob = prob_vec['hoi']
        hoi_labels = Variable(torch.cuda.FloatTensor(data['hoi_label_vec']))
        loss = criterion(hoi_prob,hoi_labels)

        batch_size = hoi_prob.size(0)
        val_loss += (batch_size*loss.data)
        count += batch_size
        step += 1

    val_loss = val_loss / float(count)
    return val_loss


def main(exp_const,data_const_train,data_const_val,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    save_constants({
            'exp':exp_const,
            'data_train':data_const_train,
            'data_val':data_const_val,
            'model':model_const},
        exp_const.exp_dir)

    print('Creating model ...')
    model = Model()
    model.const = model_const
    model.hoi_classifier = HoiClassifier(model.const.hoi_classifier).cuda()
    model.to_txt(exp_const.exp_dir,single_file=True)

    print('Creating data loaders ...')
    dataset_train = Features(data_const_train)
    dataset_val = Features(data_const_val)

    train_model(model,dataset_train,dataset_val,exp_const)

import gensim
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
from tensorboard_logger import configure, log_value
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import random

import utils.io as io
import utils.losses as losses
from utils.model import Model
from utils.constants import save_constants
from exp.hoi_classifier.models.hoi_classifier_model import HoiClassifier
from exp.hoi_classifier.data.features_dataset_unlabeled import Features_unlabeled#------!!!

from exp.hoi_classifier.data.features_dataset import Features

from utils.losses import FocalLoss

import pdb

def sigmoid_loss(feat,GT):
    logit = torch.einsum('bi,bi->b',feat,GT) #batch-wise dot
    scores = nn.functional.sigmoid(logit)
    return scores

def gen_teacher(student,prior1,prior2):       
        
        
    weight = torch.cuda.FloatTensor((student>0).float())
    student_neg =   (1-student)*weight
            
    #student = (student.t()/(weight.sum(1)+1e-8)).t()
    #student_neg = (student_neg.t()/(weight.sum(1)+1e-8)).t()
    
    
    #weight2 = torch.cuda.FloatTensor((prior1>0).float())
    #prior2 = prior2*weight2
    
    #prior1 = prior1/((weight2.sum(0))+1e-8)
    #prior2 = prior2/((weight2.sum(0))+1e-8)
    

    #mask = torch.eye(600,600).bool().cuda()# keep the diagonal to be 1.0
    #prior1.masked_fill_(mask,0.5)
    
    
    
    
            
    #teacher = ((torch.matmul(student,prior1)*0.9 + torch.matmul(student_neg,prior2)*0.1).t()/weight.sum(1)).t()
    teacher = ((torch.matmul(student,prior1)).t()/weight.sum(1)).t()
    #teacher = ((torch.matmul(student_neg,prior2)).t()/weight.sum(1)).t()
    #pdb.set_trace()
    mask = (teacher>1).cuda()
    teacher.masked_fill_(mask,1.0)
    if teacher.max().item()>1:
        pdb.set_trace()
    
    ## Normalization
    # (1)
    #teacher = teacher/600
    # (2)
    #teacher = (teacher.t()/student.sum(1)).t()#normalize with student 
    
    return teacher#/600

USE_word2vec_feat = True
USE_generalization = True
USE_glove = True

USE_emb = False
USE_OBJ_Emb = False
USE_joint = False
USE_distillation = True
USE_distillation_soft = False
USE_inception = False
USE_refine = False
USE_weight_count = False
USE_focal_loss = False
USE_batch_cls_weight = False
USE_batch_posneg = False
IGNORE_negative =False
USE_cluster = True
Semi_Supervised = False
USE_ZEROSHOT = False

def train_model(model,dataset_train,dataset_unlabeled,dataset_val,exp_const):
    
    if USE_joint:
        size1 = 256
        Embedder = nn.Sequential(nn.Linear(300,size1),nn.ReLU(inplace=True),nn.Linear(size1,size1)).cuda()
    else:
        Embedder = ''
    
    if USE_weight_count:
        weight_count =  torch.load('weight_count51.pkl').cuda() 

        
    if USE_cluster:
        hoi2gid = io.load_json_object('hoi2gid1.json') 
        #hoi2gid = io.load_json_object('anchor_only/hoi2anchor45.json') 
        #hoi2gid = io.load_json_object('anchor+others/hoi2anchor10.json') 
    
    if USE_ZEROSHOT:
        #zeroshot = ['567', '040', '516', '545', '249', '132', '472', '095', '209', '147', '458', '062', '473', '213', '076', '577', '142','090', '583', '154', '578', '462', '061', '140', '156']#-zsl1
        #zeroshot = ['371', '567', '040', '516', '545', '249', '132', '472', '095', '209', '147', '458', '062', '473', '213', '076', '577','142', '090', '583', '154', '578', '462', '061', '140']#----zsl2
        #zeroshot = ['482', '371', '567', '040', '516', '545', '249', '132', '472', '095', '209', '147', '458', '062', '473', '213', '076','577', '142', '090', '583', '154', '578', '462', '061']#----zsl3
        #zeroshot = ['227', '482', '371', '567', '040', '516', '545', '249', '132', '472', '095', '209', '147', '458', '062', '473', '213','076', '577', '142', '090', '583', '154', '578', '462']#----zsl4
        #zeroshot = ['177', '191', '160', '386', '480', '324', '251', '227', '371', '040', '545', '132', '095', '147', '062', '213', '577','090', '154', '462', '140', '043', '155', '012', '019']#----zsl5
        #zeroshot = ['250', '378', '381', '074', '231', '481', '471', '170', '482', '567', '516', '249', '472', '209', '458', '473', '076','142', '583', '578', '061', '156', '460', '020', '246']#----zsl6
        #zeroshot = ['049', '177', '191', '160', '386', '480', '324', '251', '227', '371', '040', '545', '132', '095', '147', '062', '213','577', '090', '154', '462', '140', '043', '155', '012']#----zsl7
        zeroshot = ['372', '250', '378', '381', '074', '231', '481', '471', '170', '482', '567', '516', '249', '472', '209', '458', '473','076', '142', '583', '578', '061', '156', '460', '020']#----zsl8

        zeroshot_mask = np.ones(600)

        for zz in zeroshot:
            zeroshot_mask[int(zz)-1]=0
     
    if USE_glove:
        Word2Vec= torch.load('Word2Vec_Glove.pkl') 
        word2index=io.load_json_object('word2vec_vocab_Glove.json') 
        neighbors = io.load_json_object('neighbor_object_Glove.json') 
    else:
        Word2Vec= torch.load('Word2Vec.pkl') 
        word2index=io.load_json_object('word2vec_vocab.json') 
        neighbors = io.load_json_object('neighbor_object.json') 
        
    #probability = [0.2, 0.175, 0.150, 0.125, 0.1, 0.1, 0.075, 0.05, 0.025, 0.0]#----generalize2
    #probability = [0.3, 0.25, 0.2, 0.15, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]#----generalize3 / Glove_genearalize1
    #probability = [0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]#----generalize4 / Glove_genearalize2
    probability = [0.5, 0.3, 0.2,0,0,0,0,0,0,0]#----Glove_genearalize3
    
    print('Word2Vec model Loaded!')
    
    if USE_distillation or USE_distillation_soft or Semi_Supervised:
        co_occurrence = torch.load('co-occurrence_pos.pkl') #------------for co-occurrence prior
        co_occurrence_neg = torch.load('co-occurrence_neg2.pkl') #------------for co-occurrence prior
        word_similarity = torch.load('word_similarity_Glove.pkl') #--for word2vec prior
        
        co_occurrence = torch.cuda.FloatTensor(co_occurrence)
        co_occurrence_neg = torch.cuda.FloatTensor(co_occurrence_neg)
        
        
        
        if False:#USE_cluster: # for distillaiton3
            hoi_count = torch.load('hoi_count_pos.pkl')
            hoi_count = torch.cuda.FloatTensor(hoi_count)
            co_occurrence_ = (co_occurrence.t()*hoi_count).t()
            MAX_G = max([dd[1] for dd in hoi2gid.items()])
            co_occurrence_group = torch.zeros((MAX_G+1,600)).cuda()
            group_count = torch.zeros((MAX_G+1,)).cuda()
            for hoi in range(600):
                gid = hoi2gid[str(hoi)]
                group_count[gid] += hoi_count[hoi]
                co_occurrence_group[gid,:] = co_occurrence_[hoi]
            co_occurrence_group = (co_occurrence_group.t()/group_count).t()
            co_occurrence_group[0,:] = torch.zeros(600,)
            assert(co_occurrence_group.max()<=1)

        #WEIGHT = 1.0/600.0#0.5
        #co_occurrence = co_occurrence * (1-WEIGHT)/599.0
        #co_occurrence_neg = co_occurrence_neg * (1-WEIGHT)/599.0
        #mask = torch.eye(600,600).bool().cuda()# keep the diagonal to be 1.0
        #co_occurrence.masked_fill_(mask,WEIGHT)

    if USE_joint:
        params = itertools.chain(
            model.hoi_classifier.parameters()
            ,Embedder.parameters()
            )
    else:
        params = itertools.chain(
        model.hoi_classifier.parameters()
        )
    optimizer = optim.Adam(params,lr=exp_const.lr)
    
    if USE_focal_loss:
        criterion = FocalLoss(gamma=2)
    else:
        criterion = nn.BCELoss(reduction='none')
    criterion_L2 = nn.MSELoss()
    criterion_softmax = nn.CrossEntropyLoss()
    
    step = 0
    optimizer.zero_grad()
    for epoch in range(exp_const.num_epochs):
        sampler = RandomSampler(dataset_train)
        for i, sample_id in enumerate(sampler):
            data = dataset_train[sample_id]
            ###############################################################################            
            if not USE_generalization:
                if not USE_glove:
                    GT_obj = [Word2Vec(torch.LongTensor([word2index[ddd] for ddd in dataset_train.hoi_dict[dd]['object'].split('_')])).sum(0) for dd in data['hoi_id']]
                else:
                    ## glove
                    GT_obj = [Word2Vec(torch.LongTensor([word2index[dataset_train.hoi_dict[dd]['object']]])).sum(0) for dd in data['hoi_id']]
                GT_obj = torch.stack(GT_obj,0).cuda()
            else:
                ind = np.random.choice(10,1,p=probability)[0] 
                GT_obj = [Word2Vec(torch.LongTensor([word2index[ neighbors[dataset_train.hoi_dict[dd]['object']][ind][0]  ]])) for dd in data['hoi_id']]#-----generalized version! (weighted)
                GT_obj = torch.cat(GT_obj,0).cuda()
            
            
            
            if Semi_Supervised:
                unlabeled_id = np.random.choice(len(dataset_unlabeled),1)[0] 
                data_u = dataset_unlabeled[unlabeled_id]
                if not USE_generalization:
                    if not USE_glove:
                        GT_obj_u = [Word2Vec(torch.LongTensor([word2index[ddd] for ddd in dataset_unlabeled.hoi_dict[dd]['object'].split('_')])).sum(0) for dd in data_u['hoi_id']]
                    else:
                        ## glove
                        GT_obj_u = [Word2Vec(torch.LongTensor([word2index[dataset_unlabeled.hoi_dict[dd]['object']]])).sum(0) for dd in data_u['hoi_id']]
                    GT_obj_u = torch.stack(GT_obj_u,0).cuda()
                else:
                    ind = np.random.choice(10,1,p=probability)[0] 
                    GT_obj_u = [Word2Vec(torch.LongTensor([word2index[ neighbors[dataset_unlabeled.hoi_dict[dd]['object']][ind][0]  ]])) for dd in data_u['hoi_id']]#-----generalized version! (weighted)
                    GT_obj_u = torch.cat(GT_obj_u,0).cuda()
            
            '''
            if USE_emb == True:
                GT_obj_real = [Word2Vec(torch.LongTensor([word2index[ neighbors[dataset_train.hoi_dict[dd]['object']][0][0]  ]])) for dd in data['hoi_id']]#-----generalized version! (weighted)
                GT_obj_real = torch.cat(GT_obj_real,0).cuda()
                
                if not USE_glove:
                    GT_verb = [Word2Vec(torch.LongTensor([word2index[ddd] for ddd in dataset_train.hoi_dict[dd]['verb'].split('_')])).sum(0) for dd in data['hoi_id']]
                else:
                    GT_verb = [Word2Vec(torch.LongTensor([word2index[dataset_train.hoi_dict[dd]['verb']]])).sum(0) for dd in data['hoi_id']]#---for glove
                GT_verb = torch.stack(GT_verb,0).cuda()
                
                if USE_joint :
                    GT_obj_real = Embedder(GT_obj_real)
                #pdb.set_trace()
            '''   
            ###############################################################################
            
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
            feats['global_feat'] = Variable(torch.cuda.FloatTensor(data['global_feat']).expand_as(feats['human_rcnn']))#-----!!!!!
            #pdb.set_trace()
            
            if Semi_Supervised:
                feats_u = {
                    'human_rcnn': Variable(torch.cuda.FloatTensor(data_u['human_feat'])),
                    'object_rcnn': Variable(torch.cuda.FloatTensor(data_u['object_feat'])),
                    'box': Variable(torch.cuda.FloatTensor(data_u['box_feat'])),
                    'absolute_pose': Variable(torch.cuda.FloatTensor(data_u['absolute_pose'])),
                    'relative_pose': Variable(torch.cuda.FloatTensor(data_u['relative_pose'])),
                    'human_prob_vec': Variable(torch.cuda.FloatTensor(data_u['human_prob_vec'])),
                    'object_prob_vec': Variable(torch.cuda.FloatTensor(data_u['object_prob_vec'])),
                    'prob_mask': Variable(torch.cuda.FloatTensor(data_u['prob_mask']))
                }
                feats_u['object_one_hot'] = GT_obj_u
            
            if USE_word2vec_feat == True:
                feats['object_one_hot'] = GT_obj#-----------!!!
                feats['object_one_hot2'] = Variable(torch.cuda.FloatTensor(data['object_one_hot']))
            else:
                feats['object_one_hot'] = Variable(torch.cuda.FloatTensor(data['object_one_hot']))
            
            if USE_ZEROSHOT:
                for hid, hoi in enumerate(data['hoi_id']):
                    if hoi in zeroshot: data['hoi_label'][hid]=0
                data['hoi_label_vec'] = data['hoi_label_vec']*zeroshot_mask
                assert(data['hoi_label_vec'].sum() == data['hoi_label'].sum())
                
                
                
            model.hoi_classifier.train()
            prob_vec, factor_scores,embedding = model.hoi_classifier(feats)
            if (Semi_Supervised and epoch>0):
                prob_vec_u, _,_ = model.hoi_classifier(feats_u)
            ###############################################################################
            if USE_emb == False:
                loss_emb = torch.LongTensor([0])
            else:                
                
                ## for FC_emb
                GT_both = torch.cat((GT_obj_real,GT_verb),1)
                Ones = Variable(    torch.ones(len(data['human_feat'])).cuda()  )
                loss_emb = criterion(sigmoid_loss(embedding,GT_both),Ones).mean()
                
                ## for interactiveness
                #interactiveness_label = Variable(torch.cuda.FloatTensor(data['hoi_label']))
                #loss_emb = criterion(embedding,interactiveness_label).mean()
            ########################################################################
            if USE_distillation or USE_distillation_soft or Semi_Supervised:
                hoi_prob2 = prob_vec['hoi']
            hoi_prob = prob_vec['hoi'] * feats['prob_mask']

            
            hoi_labels = Variable(torch.cuda.FloatTensor(data['hoi_label_vec']))                
            
            
            if USE_weight_count:
                if len(weight_count.size())==2:
                    label_vec = torch.cuda.LongTensor(data['hoi_label_vec'])# B*600                    
                    weight_count = torch.gather(weight_count,0,label_vec)

                loss_cls = torch.mul(criterion(hoi_prob,hoi_labels),weight_count).mean()

            elif USE_batch_cls_weight:
                ## batch level class weighting#-----------------------------!!!!!!!
                #if (hoi_labels.sum(0)>0).sum().item()==0:
                #    weight = torch.ones(len(hoi_labels)).cuda()
                #else:
                weight = (hoi_labels.sum()/(hoi_labels.sum(0)+1e-8))
                weight = weight*(hoi_labels.sum(0)>0).float()# zero for non-relevant classes
                weight = weight/((hoi_labels.sum(0)>0).sum().float()+1e-8)#  W = N_tot/Ni / 600
                loss_cls = torch.mul(criterion(hoi_prob,hoi_labels),weight).mean()
            
            elif USE_batch_posneg:
                ## batch level pos/neg weighting#-----------------------------!!!!!!!
                
                pos = torch.cuda.FloatTensor(data['hoi_label']==1)
                neg = torch.cuda.FloatTensor(data['hoi_label']==0)
                
                weight = (pos*len(data['hoi_label'])/(sum(pos)+1e-8) + neg*len(data['hoi_label'])/(sum(neg)+1e-8))/2
                                    
                loss_cls = torch.mul(criterion(hoi_prob,hoi_labels).mean(1),weight).mean()
                
            elif IGNORE_negative:
                weight = (len(data['hoi_label'])/(sum(data['hoi_label'])+1e-8))*torch.cuda.FloatTensor(data['hoi_label'])
                loss_cls = torch.mul(criterion(hoi_prob,hoi_labels).mean(1),weight).mean()
            else:
                ## from func generalization
                #weight = torch.ones(data['hoi_label_vec'].shape).cuda()
                #weight = weight-(hoi_prob>0).float()
                #weight = weight+10*torch.cuda.FloatTensor(data['hoi_label_vec']>0)
                #loss_cls = torch.mul(criterion(hoi_prob,hoi_labels),weight).mean()
                
                
                loss_cls = criterion(hoi_prob,hoi_labels).mean()
            

            if USE_OBJ_Emb:
                
                obj_labels = feats['object_one_hot2']
                loss_obj = criterion(embedding,obj_labels).mean()
            
            
            #if USE_inception:
            #    loss_inception = 0
            #    for factor_name in factor_scores.keys():
            #        loss_inception += criterion(factor_scores[factor_name],hoi_labels).mean()
            #    loss_inception = loss_inception/len(factor_scores)
            #    loss = 0.7 * loss_cls + 0.3 * loss_inception 
                
            if USE_cluster:
                cluster_label = [hoi2gid[str(dd)] for dd in data['hoi_idx']]*data['hoi_label']
                cluster_label = Variable(torch.cuda.LongTensor(cluster_label))
                loss_cluster=criterion_softmax(embedding,cluster_label)
            else:
                loss_cluster = torch.LongTensor([0])
            
            if (Semi_Supervised and epoch>1):# for unlabled
                new_label1 = torch.matmul(torch.cuda.FloatTensor(data['hoi_label_vec']),co_occurrence)
                weight = torch.cuda.FloatTensor((new_label1>0).float())
                new_label1_neg = torch.matmul((1-torch.cuda.FloatTensor(data['hoi_label_vec']))*weight,co_occurrence_neg)
                new_label1_neg = (new_label1_neg.t()/(weight.sum(1)-1+1e-8)).t()

                prior_labels1 = Variable(new_label1*0.9+new_label1_neg*0.1)

                loss_distillation = torch.mul(criterion(hoi_prob2,prior_labels1),weight).mean()
                

                teacher = Variable(gen_teacher(hoi_prob2.data,co_occurrence,co_occurrence_neg))
                weight2 = torch.cuda.FloatTensor((hoi_prob2>0).float())
                loss_unlabeled = torch.mul(criterion(hoi_prob2,teacher),weight2).mean()

            elif USE_distillation or Semi_Supervised:# for GT teacher
                
                if False:# distillation3
                    new_label1 = co_occurrence_group.index_select(0,cluster_label)
                    weight = torch.cuda.FloatTensor((new_label1>0).float())
                    prior_labels1 = Variable(new_label1)
                    
                if True:# ditillation2
                    new_label1 = torch.matmul(torch.cuda.FloatTensor(data['hoi_label_vec']),co_occurrence)
                    weight = torch.cuda.FloatTensor((new_label1>0).float())
                    
                    new_label1_neg = torch.matmul((1-torch.cuda.FloatTensor(data['hoi_label_vec']))*weight,co_occurrence_neg)
                    #new_label1 = (new_label1.t()/(weight.sum(1)+1)).t()
                    new_label1_neg = (new_label1_neg.t()/(weight.sum(1)-1+1e-8)).t()

                    prior_labels1 = Variable(new_label1*0.9+new_label1_neg*0.1)
                
                
                
                loss_distillation = torch.mul(criterion(hoi_prob2,prior_labels1),weight).mean()
                #loss_distillation = criterion(hoi_prob2,prior_labels2).mean()
                #loss_distillation = 0.5 * torch.mul(criterion(hoi_prob2,prior_labels1),weight).mean() + 0.5 * criterion(hoi_prob2,prior_labels2).mean()
                
                loss_unlabeled = torch.LongTensor([0])
            elif USE_distillation_soft or Semi_Supervised:# student disillation
                #new_label1 = np.matmul(data['hoi_label_vec'],co_occurrence)#---- 1) co-occurrence?

                
                #prior_labels1 = torch.cuda.FloatTensor(new_label1)#---- 1) co-occurrence?
                
                #teacher = gen_teacher(hoi_prob2.data,prior_labels1,prior_labels2)
                
                
                ## distillation2
                #teacher = Variable(gen_teacher(hoi_prob2.data,co_occurrence,co_occurrence_neg))
                ## distillation3
                teacher = Variable(torch.matmul(embedding.data,co_occurrence_group))
                
                weight = torch.cuda.FloatTensor((hoi_prob2>0).float())
                
                loss_distillation = torch.mul(criterion(hoi_prob2,teacher),weight).mean()
                loss_unlabeled = torch.LongTensor([0])
            else:
                loss_distillation = torch.LongTensor([0])
                loss_unlabeled = torch.LongTensor([0])
            
            
                    
            
            if USE_emb:
                loss = 1.0 * loss_cls  +  0.1 * loss_emb #------------!!!!!!!!!!!!!!!!!
            elif (Semi_Supervised and epoch>1):
                loss = 0.7 * loss_cls  +  0.3* loss_distillation  +  0.1* loss_cluster +  0.1* loss_unlabeled
            elif (USE_distillation and USE_cluster) or Semi_Supervised:
                Lambda=0.7
                loss = Lambda * loss_cls  +  (1-Lambda)* loss_distillation  +  0.1* loss_cluster
            elif ((USE_distillation_soft and epoch>0) and USE_cluster) or Semi_Supervised:
                loss = 1.0 * loss_cls  +  0.1* loss_distillation  +  0.1* loss_cluster
            elif USE_distillation or Semi_Supervised:
                loss = 0.7 * loss_cls  +  0.3* loss_distillation #------------!!!!!!!!!!!!!!!!!
            elif (USE_distillation_soft and epoch>0) or Semi_Supervised:
                #loss = loss_distillation
                loss = 1.0 * loss_cls  +  0.1* loss_distillation 
            elif USE_OBJ_Emb:
                loss = 1.0 * loss_cls  +  0.1* loss_obj
            elif USE_cluster:
                loss = 1.0 * loss_cls  +  0.1* loss_cluster
            else:
                loss = loss_cls  
            
            
            
            #pdb.set_trace() 
            loss.backward()
            if step%exp_const.imgs_per_batch==0:
                optimizer.step()
                optimizer.zero_grad()

            max_prob = hoi_prob.max().data#[0]
            max_prob_tp = torch.max(hoi_prob*hoi_labels).data#[0]

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
                    loss.data,#[0],
                    num_tp,
                    num_fp,
                    max_prob_tp,
                    max_prob)
                print(log_str)

            if step%100==0:
                log_value('train_loss',loss.data,step)#[0],step)
                log_value('cls_loss',loss_cls.data,step)#[0],step)#----!!!
                log_value('distillation_loss',loss_distillation.data,step)#[0],step)#----!!!
                log_value('cluster_loss',loss_cluster.data,step)#[0],step)#----!!!
                log_value('max_prob',max_prob,step)
                log_value('max_prob_tp',max_prob_tp,step)
                print(exp_const.exp_name)
            
            if step%5000==0:
                val_loss = eval_model(model,dataset_val,exp_const,Word2Vec,word2index,num_samples=2500)
                log_value('val_loss',val_loss,step)
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | Val Loss: {:.8f}'
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    loss.data,#[0],
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
        
        ###############################################################################
        if USE_word2vec_feat == True:
            if not USE_glove:
                GT_obj = [Word2Vec(torch.LongTensor([word2index[ddd] for ddd in dataset.hoi_dict[dd]['object'].split('_')])).sum(0) for dd in data['hoi_id']]
            else:
                GT_obj = [Word2Vec(torch.LongTensor([word2index[dataset.hoi_dict[dd]['object']]])).sum(0) for dd in data['hoi_id']]
                    
            GT_obj = torch.stack(GT_obj,0).cuda()
        ###############################################################################
        
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
        feats['global_feat'] = Variable(torch.cuda.FloatTensor(data['global_feat']).expand_as(feats['human_rcnn']))#-----!!!!!
        if USE_word2vec_feat == True:
            feats['object_one_hot'] = GT_obj#-----------!!!
            feats['object_one_hot2'] = Variable(torch.cuda.FloatTensor(data['object_one_hot']))
        else:
            feats['object_one_hot'] = Variable(torch.cuda.FloatTensor(data['object_one_hot']))
        prob_vec, factor_scores,_ = model.hoi_classifier(feats)
        
        
        
        hoi_prob = prob_vec['hoi']
        hoi_labels = Variable(torch.cuda.FloatTensor(data['hoi_label_vec']))
        loss = criterion(hoi_prob,hoi_labels)

        batch_size = hoi_prob.size(0)
        val_loss += (batch_size*loss.data)#[0])
        count += batch_size
        step += 1

    val_loss = val_loss / float(count)
    return val_loss


def main(exp_const,data_const_train,data_const_val,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    configure(exp_const.log_dir)
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
    if Semi_Supervised:
        dataset_unlabeled = Features_unlabeled(data_const_train)
    else:
        dataset_unlabeled = None
    dataset_val = Features(data_const_val)

    
    train_model(model,dataset_train,dataset_unlabeled,dataset_val,exp_const)#------!!!!


    
    


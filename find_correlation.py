import scipy.io 
import numpy as np
import pickle
import torch

mat = scipy.io.loadmat('data_symlinks/hico_clean/anno.mat') 

#mat_det = scipy.io.loadmat('anno_bbox.mat') 
#mat_det['bbox_test'][0][1000][2][0][2][0] 
#0-imge name- label(2) - 0- label index(0~N) - [labelname(0~600), subj _box, obj_box]

probMat_pos = np.zeros((600,600))
probMat_neg = np.zeros((600,600))
total_count_pos = np.zeros((600,))
total_count_neg = np.zeros((600,))

for iid in range(len(mat['anno_train'][0])):# for all training images 
    
    positive = np.where(mat['anno_train'][:,iid]==1)#----
    negative = np.where(mat['anno_train'][:,iid]!=1)
    
    for pp1 in positive[0]:
        total_count_pos[pp1] += 1
        
        for pp2 in positive[0]:
            if mat['list_action'][pp1][0][0][0] == mat['list_action'][pp2][0][0][0]:# only for same object
                probMat_pos[pp1,pp2]+=1
        
     
    for pp1 in negative[0]:
        total_count_neg[pp1] += 1   
        
        for pp2 in positive[0]:
            if mat['list_action'][pp1][0][0][0] == mat['list_action'][pp2][0][0][0]:# only for same object
                probMat_neg[pp1,pp2]+=1

            

probMat_pos2=probMat_pos.transpose()/total_count_pos 
probMat_pos2 = probMat_pos2.transpose()

probMat_neg2=probMat_neg.transpose()/total_count_neg
probMat_neg2 = probMat_neg2.transpose()



torch.save(probMat_pos2,'co-occurrence_pos.pkl')
torch.save(probMat_neg2,'co-occurrence_neg.pkl')
        
     

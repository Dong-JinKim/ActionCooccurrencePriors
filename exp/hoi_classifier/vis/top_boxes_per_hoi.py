import os
import h5py
import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
from torch.autograd import Variable
import skimage.io as skio
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import utils.io as io
from utils.pytorch_layers import get_activation
from utils.html_writer import HtmlWriter
from utils.bbox_utils import vis_sub_obj_bboxes, vis_human_keypts
from utils.model import Model
from exp.hoi_classifier.models.hoi_classifier_model import HoiClassifier
from exp.hoi_classifier.data.features_dataset import Features
import pdb

def get_gt_boxes(anno_dict,global_id,hoi_id):
    #all_boxes = []
    indexes = []
    boxes = None
    for hoi in anno_dict[global_id]['hois']:
        if hoi['id']!=hoi_id:
            continue
        num_boxes = len(hoi['connections'])
        boxes = np.zeros([num_boxes,8])
        
        for count, (i,j) in enumerate(hoi['connections']):
            boxes[count] = np.concatenate((
                hoi['human_bboxes'][i],
                hoi['object_bboxes'][j]
            ))
            indexes.append([i,j])#----!!!
        #all_boxes.append(boxes)
        
        break
    #all_boxes = np.concatenate(all_boxes)#-----!!

    #return all_boxes,indexes
    return boxes,indexes



    
    
    
    
def vis_keypts(pose,human_box,img,modify=False):
    num_keypts = pose.shape[0]
    x1y1 = human_box[:2]
    wh = 0*x1y1
    wh[0] = (human_box[2] - human_box[0])
    wh[1] = (human_box[3] - human_box[1])
    x1y1 = np.tile(x1y1[np.newaxis,:],(num_keypts,1))
    wh = np.tile(wh[np.newaxis,:],(num_keypts,1))
    keypts = 0*pose
    keypts[:,:2] = pose[:,:2]*wh + x1y1
    keypts[:,2] = pose[:,2]
    img_ = vis_human_keypts(img,keypts,modify=modify)
    return img_


def select_best_boxes_across_dataset(
        pred_hois,
        anno_dict,
        human_pose_feats,
        data_const,
        exp_const):
    global_id_det_id_score = {}
    global_id_det_id_score_backup = {}#-----!!!!
    for i in range(600):
        global_id_det_id_score[str(i+1).zfill(3)] = []
    
    for global_id in tqdm(pred_hois.keys()):
        human_obj_boxes_scores = pred_hois[global_id]['human_obj_boxes_scores'][()]
        start_end_ids = pred_hois[global_id]['start_end_ids'][()]
        for i in range(600):
            hoi_id = str(i+1).zfill(3)
            start_id, end_id = start_end_ids[i]
            for j in range(start_id,end_id):
                global_id_det_id_score[hoi_id].append(
                    (
                        global_id,
                        j,
                        human_obj_boxes_scores[j,8],
                    )
                )
                #if global_id not in global_id_det_id_score_backup.keys():#------!!!!
                #    global_id_det_id_score_backup[global_id] = []
                #global_id_det_id_score_backup[global_id].append(#------!!!!
                #    (
                #        j,
                #        hoi_id,
                #        human_obj_boxes_scores[j,8],
                #    )
                #)
    
    #for global_id in global_id_det_id_score_backup.keys():###---!!!
    #    global_id_det_id_score_backup[global_id] = sorted(
    #        global_id_det_id_score_backup[global_id],
    #        key=lambda x: x[2],
    #        reverse=True)
            
    top_boxes = {}
    for hoi_id in tqdm(global_id_det_id_score.keys()):
        global_id_det_id_score[hoi_id] = sorted(
            global_id_det_id_score[hoi_id],
            key=lambda x: x[2],
            reverse=True)

        boxes_scores = np.zeros([exp_const.num_to_vis,9])

        global_ids = []
        top_predictions = []#-------!!!!
        gt_boxes = []
        human_pose = []
        boxes_index  = [] #-----!!!!!
        for i in range(exp_const.num_to_vis):
            global_id,det_id,score = global_id_det_id_score[hoi_id][i]
            global_ids.append(global_id)

            #top_predictions.append([[dd[1],dd[2]] for dd in global_id_det_id_score_backup[global_id][:10]])#-------!!!
            #boxes_scores[i] = pred_hois[global_id]['human_obj_boxes_scores'][det_id]
            
            box_score = pred_hois[global_id]['human_obj_boxes_scores'][det_id]
            boxes_scores[i] = box_score[:9]
            other_predictions = [[str(ii+1).zfill(3),bb] for ii, bb in enumerate(box_score[9:])]
            top_predictions.append(sorted(other_predictions,key=lambda k:k[1],reverse=True)[:10])
            #pdb.set_trace()
            
            
            boxes,indexes = get_gt_boxes(anno_dict,global_id,hoi_id)#----!!!
            gt_boxes.append(boxes)
            boxes_index.append(indexes)
            #pdb.set_trace()
            
            human_pose.append(
                np.reshape(
                    human_pose_feats[global_id]['absolute_pose'][det_id],
                    (data_const.num_pose_keypoints,3)))
        #pdb.set_trace()
        top_boxes[hoi_id] = {
            'boxes_scores': boxes_scores,
            #'top_predictions': top_predictions,#-----!
            'gt_boxes':  gt_boxes,
            'global_ids': global_ids,
            'human_pose': human_pose,
            'boxes_index': boxes_index,#-----!!!!!
            
        }
    return top_boxes


def get_gt_hois(anno,hoi_dict):
    gt_hoi_names = ''
    for hid, hoi_id in enumerate(anno['pos_hoi_ids']):
        obj_name = hoi_dict[hoi_id]['object']
        verb_name = hoi_dict[hoi_id]['verb']
        connections = [f's{dd[0]}-o{dd[1]}' for dd in anno['hois'][hid]['connections']]#-----------------!!!!
        hoi_name = f'{verb_name}_{obj_name},{connections}'#-----!!!!!!
        gt_hoi_names += hoi_name
        gt_hoi_names += '<br />'
    
    return gt_hoi_names

        
        
def get_pred_hois(pred,hoi_dict):#----!!!!
    gt_hoi_names = ''
    #pdb.set_trace()
    for iid in range(10):
        hoi_id = pred[iid][0]
        score = pred[iid][1]
        if score>1e-8:
            obj_name = hoi_dict[hoi_id]['object']
            verb_name = hoi_dict[hoi_id]['verb']
            hoi_name = f'{verb_name}_{obj_name}'
            output_str = hoi_name+f' ({round(score,3)})'
            gt_hoi_names += output_str
            gt_hoi_names += '<br />'
    
    return gt_hoi_names

def create_html(top_boxes,anno_dict,hoi_dict,img_dir,vis_dir):
    for hoi_id in tqdm(top_boxes.keys()):
        hoi_name = '_'.join(
            [hoi_id, hoi_dict[hoi_id]['verb'], hoi_dict[hoi_id]['object']])
        object_name = hoi_dict[hoi_id]['object']
        hoi_vis_dir = os.path.join(vis_dir,hoi_name)
        io.mkdir_if_not_exists(hoi_vis_dir,recursive=True)
        html_filename = os.path.join(hoi_vis_dir,'index.html')
        html_writer = HtmlWriter(html_filename)
        col_dict = {
            0: 'Global ID',
            1: 'Predicted Score',
            2: f'Predictions for {hoi_name}',
            3: f'Detected Boxes and Pose for human and {object_name} categories',
            4: f'GT Detections for {hoi_name}',
            5: 'All GT HOI categories annotated in the image <br /> (for any human-object pair)',
            #6: 'Top-10 predictions of the image',#-----!!!!
        }
        html_writer.add_element(col_dict)
        boxes_scores = top_boxes[hoi_id]['boxes_scores']
        boxes_index = top_boxes[hoi_id]['boxes_index']
        gt_boxes = top_boxes[hoi_id]['gt_boxes']
        pose_keypts = top_boxes[hoi_id]['human_pose']
        for i, global_id in enumerate(top_boxes[hoi_id]['global_ids']):
            anno = anno_dict[global_id]
            img_path = os.path.join(img_dir,anno['image_path_postfix'])
            img = skio.imread(img_path)
            if len(img.shape)==2:
                img = img[:,:,np.newaxis]
                img = np.tile(img,(1,1,3))

            out_img = vis_sub_obj_bboxes(
                [boxes_scores[i,:4]],
                [boxes_scores[i,4:8]],
                [],#-------!!!
                img,
                modify=False)
            #out_img = vis_keypts(#--------------I don't want to draw kypoints!!!!
            #    pose_keypts[i],
            #    boxes_scores[i,:4],
            #    out_img,
            #    modify=True)
            out_img_path = os.path.join(
                hoi_vis_dir,
                str(i).zfill(3)+'.png')
            skio.imsave(out_img_path,out_img)

            out_img_on_white = vis_sub_obj_bboxes(
                [boxes_scores[i,:4]],
                [boxes_scores[i,4:8]],
                [],#---------!!!
                0*img+255,
                modify=False)
            out_img_on_white = vis_keypts(
                pose_keypts[i],
                boxes_scores[i,:4],
                out_img_on_white,
                modify=True)
            out_img_on_white_path = os.path.join(
                hoi_vis_dir,
                str(i).zfill(3)+'_on_white.png')
            skio.imsave(out_img_on_white_path,out_img_on_white)
            
            if gt_boxes[i] is not None:
                #pdb.set_trace()
                gt_out_img = vis_sub_obj_bboxes(
                    gt_boxes[i][:,:4],
                    gt_boxes[i][:,4:8],
                    boxes_index[i],#----!!!
                    img,
                    modify=False)
            else:
                gt_out_img = copy.deepcopy(img)

            gt_out_img_path = os.path.join(
                hoi_vis_dir,
                str(i).zfill(3)+'_gt.png')
            skio.imsave(gt_out_img_path,gt_out_img)

            gt_hoi_names = get_gt_hois(anno,hoi_dict)
            
            #pdb.set_trace()
            #pred_hoi_names = get_pred_hois(top_boxes[hoi_id]['top_predictions'][i],hoi_dict)#------!!!!

            col_dict = {
                0: global_id,
                1: round(boxes_scores[i,8],4),
                2: html_writer.image_tag(str(i).zfill(3)+'.png'),
                3: html_writer.image_tag(str(i).zfill(3)+'_on_white.png'),
                4: html_writer.image_tag(str(i).zfill(3)+'_gt.png'),
                5: gt_hoi_names,
                #6:pred_hoi_names,#----!!!!
            }
            html_writer.add_element(col_dict)
        
        html_writer.close()


def main(exp_const,data_const,model_const):
    print('Loading pred dets ...')
    pred_hois = h5py.File(data_const.pred_hoi_dets_h5py,'r')
    human_pose_feats = h5py.File(data_const.human_pose_feats_hdf5,'r')

    print('Reading anno_list.json ...')
    anno_list = io.load_json_object(data_const.anno_list_json)
    anno_dict = {anno['global_id']:anno for anno in anno_list}

    print('Selecting top box configurations for each hoi ...')
    top_boxes = select_best_boxes_across_dataset(
        pred_hois,
        anno_dict,
        human_pose_feats,
        data_const,
        exp_const)

    hoi_list = io.load_json_object(data_const.hoi_list_json)
    hoi_dict = {hoi['id']: hoi for hoi in hoi_list}

    print('Creating visualization images ...')
    vis_dir = os.path.join(exp_const.exp_dir,'vis/top_boxes_per_hoi')
    create_html(
        top_boxes,
        anno_dict,
        hoi_dict,
        data_const.images_dir,
        vis_dir)


    

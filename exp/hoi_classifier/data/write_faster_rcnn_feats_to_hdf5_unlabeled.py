import os
import h5py
import numpy as np
from tqdm import tqdm

import utils.io as io

def main():
    faster_rcnn_boxes = 'data_symlinks/hico_processed_finetune101_unlabeled/faster_rcnn_boxes'
    proc_dir = 'data_symlinks/hico_processed_finetune101_unlabeled/faster_rcnn_fc7.hdf5'
    
    filenames = os.listdir(faster_rcnn_boxes)
    global_ids = set([ff.split('_')[0] for ff in filenames])
    feats_hdf5 = os.path.join(proc_dir)
    feats = h5py.File(feats_hdf5,'w') 
    for global_id in tqdm(global_ids):
        fc7_npy = os.path.join(
            faster_rcnn_boxes,
            f'{global_id}_fc7.npy')
        fc7 = np.load(fc7_npy)
        feats.create_dataset(global_id,data=fc7)
        
    feats.close()

if __name__=='__main__':
    main()

#!/bin/bash
MODEL_NUM=$1
SUBSET="test"
HICO_EXP_DIR="${PWD}/data_symlinks/hico_exp_finetune101/hoi_classifier"
EXP_NAME="factors_101_glove_generalize3_FC2_zsl8"
echo $EXP_NAME
#MODEL_NUM="100000"
PRED_HOI_DETS_HDF5="${HICO_EXP_DIR}/${EXP_NAME}/pred_hoi_dets_${SUBSET}_${MODEL_NUM}.hdf5" #_refined00001
OUT_DIR="${HICO_EXP_DIR}/${EXP_NAME}/mAP_eval/${SUBSET}_${MODEL_NUM}" #_refined00001/
PROC_DIR="${PWD}/data_symlinks/hico_processed"

python3 -m exp.hico_eval.compute_map \
    --pred_hoi_dets_hdf5 $PRED_HOI_DETS_HDF5 \
    --out_dir $OUT_DIR \
    --proc_dir $PROC_DIR \
    --subset $SUBSET

python3 -m exp.hico_eval.sample_complexity_analysis \
    --out_dir $OUT_DIR

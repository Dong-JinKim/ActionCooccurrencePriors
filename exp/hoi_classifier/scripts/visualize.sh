MODEL_NUM=$1
CUDA_VISIBLE_DEVICES=$GPU python3 -m exp.hoi_classifier.run \
    --exp exp_top_boxes_per_hoi \
    --rcnn_det_prob \
    --verb_given_appearance \
    --verb_given_boxes_and_object_label \
    --verb_given_human_pose \
    --model_num $MODEL_NUM \
    --fappend 101_glove_generalize3_FC2_MTLv1_MoE_distillation2_600

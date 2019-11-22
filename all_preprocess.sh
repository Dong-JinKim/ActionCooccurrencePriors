echo "write_faster_rcnn_feats_to_hdf5 ... "
python3 -m exp.hoi_classifier.data.write_faster_rcnn_feats_to_hdf5
echo "exp_select_and_evaluate_confident_boxes_in_hico ... "
python3 -m exp.detect_coco_objects.run --exp exp_select_and_evaluate_confident_boxes_in_hico



SUBSETS=( 'train' 'val' 'test')
for subset in "${SUBSETS[@]}"
do
    echo "Generate and label candidates for ${subset} ... "
    python3 -m exp.hoi_classifier.run \
        --exp exp_gen_and_label_hoi_cand \
        --subset $subset \
        --gen_hoi_cand \
        --label_hoi_cand

    echo "Cache box features for ${subset} ... "
    python3 -m exp.hoi_classifier.run \
        --exp exp_cache_box_feats \
        --subset $subset

    echo "Assign pose to human candidates for ${subset} ... "
    python3 -m exp.hoi_classifier.run \
        --exp exp_assign_pose_to_human_cand \
        --subset $subset

    echo "Cache pose features for ${subset} ... "
    python3 -m exp.hoi_classifier.run \
        --exp exp_cache_pose_feats \
        --subset $subset

done

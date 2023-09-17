INTEGRITY_FILE="FILES_COMPLETED.txt"

while true; do
    if [ -f "$INTEGRITY_FILE" ]; then
        echo "File $INTEGRITY_FILE exists. Running remaining scripts."
        break
    fi
    sleep 10
done

# main training
python training_ROI_preds.py --learning_rate 3e-4 --momentum 0.99 --num_extra_nonexpert_training 80 --model ROI_stage1_fold2 --device 0 --memory_limit 0.495 --epochs 36 --num_extra_steps 14 --train_data ROI_classifier_fold1_train --val_data ROI_classifier_fold1_val --num_slices 9 --hidden_blocks 1 2 6 8 19 4 --use_async_sampler
# fine tune on boundary
python training_ROI_preds.py --learning_rate 3e-4 --momentum 0.99 --num_extra_nonexpert_training 80 --model ROI_stage1_boundary_fold2 --device 0 --memory_limit 0.495 --epochs 12 --num_extra_steps 14 --train_data ROI_classifier_fold1_train --val_data ROI_classifier_fold1_val --num_slices 9 --hidden_blocks 1 2 6 8 19 4 --use_boundary_augmentation --use_async_sampler --load_model ROI_stage1_fold2
# validation dataset
python stage1_organ_segmentation.py --model ROI_stage1_boundary_fold2 --device 0 --memory_limit 0.495 --dataset ROI_classifier_fold1_val --hidden_blocks 1 2 6 8 19 4
# training dataset
python stage1_organ_segmentation.py --model ROI_stage1_boundary_fold2 --device 0 --memory_limit 0.495 --dataset ROI_classifier_fold1_train --hidden_blocks 1 2 6 8 19 4
# eval
python stage1_organ_segmentation_validate.py --device 0 --train_data ROI_classifier_fold1_train --val_data ROI_classifier_fold1_val

# extra
python training_ROI_preds.py --learning_rate 3e-4 --momentum 0.99 --num_extra_nonexpert_training 80 --model ROI_stage1_fold2_verify --device 0 --memory_limit 0.495 --epochs 3 --num_extra_steps 14 --train_data ROI_classifier_fold1_train --val_data ROI_classifier_fold1_val --num_slices 9 --hidden_blocks 1 2 6 8 19 4 --use_async_sampler
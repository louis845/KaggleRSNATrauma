INTEGRITY_FILE="FILES_COMPLETED.txt"

while true; do
    if [ -f "$INTEGRITY_FILE" ]; then
        echo "File $INTEGRITY_FILE exists. Running remaining scripts."
        break
    fi
    sleep 10
done

#python training_ROI_preds.py --learning_rate 3e-4 --momentum 0.99 --num_extra_nonexpert_training 0 --model ROI_fold2_3d_noextra --device 0 --memory_limit 0.495 --epochs 180 --num_extra_steps 1 --train_data segmentation_fold1_train --val_data segmentation_fold1_val --num_slices 9 --hidden_blocks 1 2 6 8 19 4 --use_3d_prediction --use_async_sampler --positive_weight 5.0
python training_ROI_classifier.py --learning_rate 3e-4 --momentum 0.99 --model ROI_classifier_fold2 --device 0 --epochs 120 --num_extra_steps 0 --train_data ROI_classifier_fold1_expert_train --val_data ROI_classifier_fold1_expert_val --num_slices 9 --hidden_blocks 1 2 6 8 12 4 --use_3d_prediction --use_async_sampler
INTEGRITY_FILE="FILES_COMPLETED.txt"

while true; do
    if [ -f "$INTEGRITY_FILE" ]; then
        echo "File $INTEGRITY_FILE exists. Running remaining scripts."
        break
    fi
    sleep 10
done

#python training_ROI_preds.py --learning_rate 3e-4 --momentum 0.99 --num_extra_nonexpert_training 0 --model ROI_fold3_3d_noextra --device 1 --epochs 180 --num_extra_steps 1 --train_data segmentation_fold2_train --val_data segmentation_fold2_val --num_slices 9 --hidden_blocks 1 2 6 8 19 4 --use_3d_prediction --use_async_sampler --positive_weight 5.0
#python training_ROI_classifier.py --learning_rate 3e-4 --momentum 0.99 --model ROI_classifier_segonly_real_fold3 --device 1 --epochs 240 --num_extra_steps 0 --train_data ROI_classifier_fold2_expert_train --val_data ROI_classifier_fold2_expert_val --num_slices 9 --hidden_blocks 1 2 6 8 16 4 --use_3d_prediction --use_async_sampler
#python training_ROI_classifier.py --learning_rate 3e-4 --momentum 0.99 --model ROI_classifier_all_fold3 --device 1 --epochs 240 --num_extra_steps 0 --train_data ROI_classifier_fold2_train --val_data ROI_classifier_fold2_expert_val --num_slices 9 --hidden_blocks 1 2 6 8 16 4 --use_3d_prediction --use_async_sampler --load_model ROI_classifier_segonly_real_fold3
python training_ROI_classifier.py --learning_rate 3e-4 --momentum 0.99 --model ROI_classifier_deep_fold3 --device 1 --epochs 240 --num_extra_steps 0 --train_data ROI_classifier_fold2_train --val_data ROI_classifier_fold2_expert_val --num_slices 9 --hidden_blocks 1 2 6 8 16 4 --use_3d_prediction --use_async_sampler --use_deep_reduction
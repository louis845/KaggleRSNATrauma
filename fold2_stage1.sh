INTEGRITY_FILE="FILES_COMPLETED.txt"

while true; do
    if [ -f "$INTEGRITY_FILE" ]; then
        echo "File $INTEGRITY_FILE exists. Running remaining scripts."
        break
    fi
    sleep 10
done

python training_ROI_preds.py --learning_rate 3e-4 --momentum 0.99 --num_extra_nonexpert_training 80 --model ROI_stage1_fold2 --device 0 --memory_limit 0.495 --epochs 30 --num_extra_steps 9 --train_data ROI_classifier_fold1_train --val_data ROI_classifier_fold1_val --num_slices 9 --hidden_blocks 1 2 6 8 19 4 --use_async_sampler

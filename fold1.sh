INTEGRITY_FILE="FILES_COMPLETED.txt"

while true; do
    if [ -f "$INTEGRITY_FILE" ]; then
        echo "File $INTEGRITY_FILE exists. Running remaining scripts."
        break
    fi
    sleep 10
done

python training_ROI_preds.py --learning_rate 3e-4 --model ROI_fold1_3d --device 0 --epochs 120 --num_extra_steps 1 --train_data segmentation_extra_fold0_train --val_data segmentation_extra_fold0_val --num_slices 9 --hidden_blocks 1 2 6 8 23 4 --use_3d_prediction --use_async_sampler

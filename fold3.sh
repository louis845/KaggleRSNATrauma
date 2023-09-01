INTEGRITY_FILE="FILES_COMPLETED.txt"

while true; do
    if [ -f "$INTEGRITY_FILE" ]; then
        echo "File $INTEGRITY_FILE exists. Running remaining scripts."
        break
    fi
    sleep 10
done

python training_ROI_preds.py --learning_rate 3e-4 --model ROI_fold3_elastic --device 1 --epochs 120 --num_extra_steps 1 --train_data segmentation_fold2_train --val_data segmentation_fold2_val --num_slices 9 --hidden_blocks 1 2 6 8 23 5

INTEGRITY_FILE="FILES_COMPLETED.txt"

while true; do
    if [ -f "$INTEGRITY_FILE" ]; then
        echo "File $INTEGRITY_FILE exists. Running remaining scripts."
        break
    fi
    sleep 10
done

python training_ROI_preds.py --learning_rate 3e-4 --model ROI_fold2 --device 0 --epochs 120 --num_extra_steps 1 --train_data segmentation_fold1_train --val_data segmentation_fold1_val --num_slices 9

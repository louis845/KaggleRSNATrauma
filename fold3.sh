INTEGRITY_FILE="FILES_COMPLETED.txt"

while true; do
    if [ -f "$INTEGRITY_FILE" ]; then
        echo "File $INTEGRITY_FILE exists. Running remaining scripts."
        break
    fi
    sleep 10
done

python training_ROI_preds.py --learning_rate 3e-4 --model ROI_fold3 --device 1 --epochs 100 --train_data segmentation_fold2_train --val_data segmentation_fold2_val --num_slices 10

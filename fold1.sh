INTEGRITY_FILE="FILES_COMPLETED.txt"

while true; do
    if [ -f "$INTEGRITY_FILE" ]; then
        echo "File $INTEGRITY_FILE exists. Running remaining scripts."
        break
    fi
    sleep 10
done

python training_ROI_preds.py --learning_rate 1e-3 --model ROI_fold1 --device 0 --epochs 30 --train_data segmentation_fold0_train --val_data segmentation_fold0_val

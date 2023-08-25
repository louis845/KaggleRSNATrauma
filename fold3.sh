INTEGRITY_FILE="FILES_COMPLETED.txt"

while true; do
    if [ -f "$INTEGRITY_FILE" ]; then
        echo "File $INTEGRITY_FILE exists. Running remaining scripts."
        break
    fi
    sleep 10
done

python training_classifier.py --model ks_fold2_mean --device 1 --epochs 20 --train_data kidney_spleen_train_fold_2 --val_data kidney_spleen_val_fold_2 --kidney 1 --spleen 1 --proba_head mean --momentum 0.9995 --second_momentum 0.9999
python training_classifier.py --model ks_fold2_union --device 1 --epochs 20 --train_data kidney_spleen_train_fold_2 --val_data kidney_spleen_val_fold_2 --kidney 1 --spleen 1 --proba_head union --momentum 0.9995 --second_momentum 0.9999

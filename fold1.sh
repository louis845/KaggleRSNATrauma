python training_classifier.py --model ks_fold0_mean --device 0 --epochs 20 --train_data kidney_spleen_train_fold_0 --val_data kidney_spleen_val_fold_0 --kidney 1 --spleen 1 --proba_head mean --momentum 0.999 --second_momentum 0.9995 --batch_norm_momentum 0.001
python training_classifier.py --model ks_fold0_union --device 0 --epochs 20 --train_data kidney_spleen_train_fold_0 --val_data kidney_spleen_val_fold_0 --kidney 1 --spleen 1 --proba_head union --momentum 0.999 --second_momentum 0.9995 --batch_norm_momentum 0.001

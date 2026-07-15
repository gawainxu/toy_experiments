#for i in $(seq 0 29); do
#  python3 Toy_features.py --model_path "./models/toy_toy_E1_${i}.pth" --inliers_id 0 --outliers_id -1 --data_path "./toy_data_train" --model_name "toy"
#done


python3 Toy_finetune.py --experiment_name "E12" --dataset "toy" --old_classes_idx 0 --classes_idx 2 --model_name "toy" --last_model_path "./models/toy_toy_E1.pth" --losses_path "toy_toy_E1"
#for i in $(seq 0 29); do
#  python3 Toy_features.py --model_path "./models/toy_toy_E1_${i}.pth" --inliers_id 0 --outliers_id -1 --data_path "./toy_data_train" --model_name "toy"
#done


#python3 Toy_finetune.py --experiment_name "E13" --dataset "toy" --classes_idx 2 --old_classes_idx 0 --model_name "cnn" --last_model_path "./models/cnn_toy_E1.pth" --losses_path "cnn_toy_E13"
#python3 Toy_finetune.py --experiment_name "E14" --dataset "toy" --classes_idx 3 --old_classes_idx 0 --model_name "cnn" --last_model_path "./models/cnn_toy_E1.pth" --losses_path "cnn_toy_E14"
#python3 Toy_finetune.py --experiment_name "E15" --dataset "toy" --classes_idx 4 --old_classes_idx 0 --model_name "cnn" --last_model_path "./models/cnn_toy_E1.pth" --losses_path "cnn_toy_E15" --lr 1e-1
#python3 Toy_finetune.py --experiment_name "E26" --dataset "toy" --classes_idx 5 --old_classes_idx 1 --model_name "cnn" --last_model_path "./models/cnn_toy_E2.pth" --losses_path "cnn_toy_E26"
#python3 Toy_finetune.py --experiment_name "E27" --dataset "toy" --classes_idx 6 --old_classes_idx 1 --model_name "cnn" --last_model_path "./models/cnn_toy_E2.pth" --losses_path "cnn_toy_E27"
#python3 Toy_finetune.py --experiment_name "E28" --dataset "toy" --classes_idx 7 --old_classes_idx 1 --model_name "cnn" --last_model_path "./models/cnn_toy_E2.pth" --losses_path "cnn_toy_E28" --lr 1e-1

#python3 Toy_finetune.py --experiment_name "E15" --dataset "toy" --classes_idx 4 --old_classes_idx 0 --model_name "toy" --last_model_path "./models/toy_toy_E1.pth" --losses_path "toy_toy_E15"
#python3 Toy_finetune.py --experiment_name "E28" --dataset "toy" --classes_idx 7 --old_classes_idx 1 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth" --losses_path "toy_toy_E28"

#python3 Toy_train.py --experiment_name "E4" --dataset "toy" --classes_idx 3 --old_classes_idx 0 --model_name "toy" --last_model_path "./models/toy_toy_E1.pth" --losses_path "toy_toy_E4"
#python3 Toy_train.py --experiment_name "E5" --dataset "toy" --classes_idx 4 --old_classes_idx 0 --model_name "toy" --last_model_path "./models/toy_toy_E1.pth" --losses_path "toy_toy_E5"
#python3 Toy_train.py --experiment_name "E6" --dataset "toy" --classes_idx 5 --old_classes_idx 1 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth" --losses_path "toy_toy_E6"
#python3 Toy_train.py --experiment_name "E7" --dataset "toy" --classes_idx 6 --old_classes_idx 1 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth" --losses_path "toy_toy_E7"
#python3 Toy_train.py --experiment_name "E8" --dataset "toy" --classes_idx 7 --old_classes_idx 1 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth" --losses_path "toy_toy_E8"

python3 toy_test.py --experiment_name "E1" --dataset "toy" --classes_idx1 0 --model_name "toy" --model_path "./models/toy_toy_E1.pth"
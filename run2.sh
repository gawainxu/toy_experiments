#echo "For E2"
#RESULTS_FILE="results2.txt"
#echo "inliers_id, result" > $RESULTS_FILE

#for i in $(seq 0 99); do
#  echo "Running iteration $i"
#  result=$(python3 Toy_distances.py --feature_path "./features/toy_toy_E2_${i}_train" --feature_path_test "./features/toy_toy_E2_${i}_rectangle_blue" --num_classes 3 --feature_to_visualize "linear2" --fig_save_path "./plots/hist_E2_${i}.png")
#  echo "$i, $result" >> $RESULTS_FILE
#done


#python3 Toy_train.py --experiment_name "E1" --dataset "toy" --classes_idx 0 --model_name "cnn" --losses_path "cnn_toy_E1"
#python3 Toy_train.py --experiment_name "E2" --dataset "toy" --classes_idx 1 --model_name "cnn" --losses_path "cnn_toy_E2"


#python3 main_ce.py --batch_size 64 --epochs 300 --learning_rate 0.1 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar100_marco" --trail 0
#python3 main_ce.py --batch_size 64 --epochs 300 --learning_rate 0.1 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar100_marco" --trail 1
#python3 main_ce.py --batch_size 64 --epochs 300 --learning_rate 0.1 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar100_marco" --trail 2
#python3 main_ce.py --batch_size 64 --epochs 300 --learning_rate 0.1 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar100_marco" --trail 3

#python3 main_linear.py --model "resnet18" --datasets "cifar100_marco" --trail 7 --backbone_model_direct "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_0_128_128" --backbone_model_name "last.pth"

#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E1_task_0_data_0_train" --feature_path2 "./features/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E1_task_0_data_0_train" --feature_path2 "./features/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E1_task_0_data_0_train" --feature_path2 "./features/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E2_task_0_data_0_train" --feature_path2 "./features/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E2_task_0_data_0_train" --feature_path2 "./features/toy_toy_E7_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E2_task_0_data_0_train" --feature_path2 "./features/toy_toy_E8_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"


#python3 Toy_train.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx 1 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth" --losses_path "toy_toy_E5"
#python3 Toy_train.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx 1 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth" --losses_path "toy_toy_E6"


python3 Toy_train.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx 2 --model_name "toy" --last_model_path "./models/toy_toy_E5_task_1.pth" --losses_path "toy_toy_E5_task_2"
python3 Toy_train.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx 2 --model_name "toy" --last_model_path "./models/toy_toy_E6_task_1.pth" --losses_path "toy_toy_E6_task_2"
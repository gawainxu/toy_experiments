#echo "init training"
#python3 Toy_train.py --experiment_name "E1" --dataset "toy" --experiment_idx 0 --task_idx 0 --model_name "toy" --losses_path "toy_toy_E1" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_train.py --experiment_name "E2" --dataset "toy" --experiment_idx 1 --task_idx 0 --model_name "toy" --losses_path "toy_toy_E2" --model_root "./models2/" --losses_root "./losses2/"

#echo "continue training for task 1"
#python3 Toy_train.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx 1 --model_name "toy" --last_model_path "./models2/toy_toy_E1_task_0.pth" --losses_path "toy_toy_E3" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_train.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx 1 --model_name "toy" --last_model_path "./models2/toy_toy_E1_task_0.pth" --losses_path "toy_toy_E4" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_train.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx 1 --model_name "toy" --last_model_path "./models2/toy_toy_E2_task_0.pth" --losses_path "toy_toy_E5" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_train.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx 1 --model_name "toy" --last_model_path "./models2/toy_toy_E2_task_0.pth" --losses_path "toy_toy_E6" --model_root "./models2/" --losses_root "./losses2/"



#echo "linear probe of the init model with the task 1 data"
#python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 0 --task_idx_data 1 --model_name "toy" --last_model_path "./models2/toy_toy_E1_task_0.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 0 --task_idx_data 1 --model_name "toy" --last_model_path "./models2/toy_toy_E1_task_0.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 0 --task_idx_data 1 --model_name "toy" --last_model_path "./models2/toy_toy_E2_task_0.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 0 --task_idx_data 1 --model_name "toy" --last_model_path "./models2/toy_toy_E2_task_0.pth" --model_root "./models2/" --losses_root "./losses2/"


#echo "linear probe of the task 1 model with the task 0 data"
#python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 1 --task_idx_data 0 --model_name "toy" --last_model_path "./models2/toy_toy_E3_task_1.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 1 --task_idx_data 0 --model_name "toy" --last_model_path "./models2/toy_toy_E4_task_1.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 1 --task_idx_data 0 --model_name "toy" --last_model_path "./models2/toy_toy_E5_task_1.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 1 --task_idx_data 0 --model_name "toy" --last_model_path "./models2/toy_toy_E6_task_1.pth" --model_root "./models2/" --losses_root "./losses2/"



#echo "continual training for task 2"
#python3 Toy_train.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx 2 --model_name "toy" --last_model_path "./models2/toy_toy_E3_task_1.pth" --losses_path "toy_toy_E3_task_2" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_train.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx 2 --model_name "toy" --last_model_path "./models2/toy_toy_E4_task_1.pth" --losses_path "toy_toy_E4_task_2" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_train.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx 2 --model_name "toy" --last_model_path "./models2/toy_toy_E5_task_1.pth" --losses_path "toy_toy_E5_task_2" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_train.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx 2 --model_name "toy" --last_model_path "./models2/toy_toy_E6_task_1.pth" --losses_path "toy_toy_E6_task_2" --model_root "./models2/" --losses_root "./losses2/"


#echo "linear probe of the init model with the task 2 data"
#python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models2/toy_toy_E1_task_0.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models2/toy_toy_E1_task_0.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models2/toy_toy_E2_task_0.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models2/toy_toy_E2_task_0.pth" --model_root "./models2/" --losses_root "./losses2/"


#echo "linear probe of the task 1 model with the task 2 data"
#python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models2/toy_toy_E3_task_1.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models2/toy_toy_E4_task_1.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models2/toy_toy_E5_task_1.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models2/toy_toy_E6_task_1.pth" --model_root "./models2/" --losses_root "./losses2/"


#echo "linear probe of the task 2 model with the task 0 data"
#python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models2/toy_toy_E3_task_2.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models2/toy_toy_E4_task_2.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models2/toy_toy_E5_task_2.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models2/toy_toy_E6_task_2.pth" --model_root "./models2/" --losses_root "./losses2/"


#echo "linear probe of the task 2 model with the task 1 data"
#python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models2/toy_toy_E3_task_2.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models2/toy_toy_E4_task_2.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models2/toy_toy_E5_task_2.pth" --model_root "./models2/" --losses_root "./losses2/"
#python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models2/toy_toy_E6_task_2.pth" --model_root "./models2/" --losses_root "./losses2/"


#echo "feature reading for task0"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 0 --model_name "toy" --model_path "./models2/toy_toy_E1_task_0.pth" --feature_save_path "./features2/"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 1 --model_name "toy" --model_path "./models2/toy_toy_E2_task_0.pth" --feature_save_path "./features2/"

#echo "feature reading for task1"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 2 --model_name "toy" --model_path "./models2/toy_toy_E3_task_1.pth" --feature_save_path "./features2/"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 3 --model_name "toy" --model_path "./models2/toy_toy_E4_task_1.pth" --feature_save_path "./features2/"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 4 --model_name "toy" --model_path "./models2/toy_toy_E5_task_1.pth" --feature_save_path "./features2/"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 5 --model_name "toy" --model_path "./models2/toy_toy_E6_task_1.pth" --feature_save_path "./features2/"


#echo "feature reading for task2"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 2 --experiment_idx 2 --model_name "toy" --model_path "./models2/toy_toy_E3_task_2.pth" --feature_save_path "./features2/"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 2 --experiment_idx 3 --model_name "toy" --model_path "./models2/toy_toy_E4_task_2.pth" --feature_save_path "./features2/"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 2 --experiment_idx 4 --model_name "toy" --model_path "./models2/toy_toy_E5_task_2.pth" --feature_save_path "./features2/"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 2 --experiment_idx 5 --model_name "toy" --model_path "./models2/toy_toy_E6_task_2.pth" --feature_save_path "./features2/"


# CKA between base model data between the task 1 and task 2 models
#echo "task 1 linear3"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E3_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E4_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E5_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E6_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear3"
#echo "task 2 linear3"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"

echo "task 12 linear3"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"


#echo "task 1 linear2"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E3_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E4_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E5_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E6_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear2"
#echo "task 2 linear2"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"


echo "task 2 linear2"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"


#echo "task 1 linear1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E3_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E4_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E5_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E6_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear1"
#echo "task 2 linear1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"

echo "task 12 linear1"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"


#echo "task 1 conv1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E3_task_1_task_1_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E4_task_1_task_1_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E5_task_1_task_1_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E6_task_1_task_1_data_0_train" --num_classes 2 --feature_name "conv1"
#echo "task 2 conv1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features2/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"

echo "task 12 conv1"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E1_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
python3 Toy_metrics.py --feature_path1 "./features2/toy_toy_E2_task_1_task_1_data_0_train" --feature_path2 "./features2/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
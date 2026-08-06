echo "init training"
python3 Toy_train.py --experiment_name "E1" --dataset "toy" --experiment_idx 0 --task_idx 0 --model_name "toy" --losses_path "toy_toy_E1" --model_root "./models6/" --losses_root "./losses6/"
python3 Toy_train.py --experiment_name "E2" --dataset "toy" --experiment_idx 1 --task_idx 0 --model_name "toy" --losses_path "toy_toy_E2" --model_root "./models6/" --losses_root "./losses6/"


echo "continue training for task 1"
python3 Toy_train.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx 1 --model_name "toy" --last_model_path "./models6/toy_toy_E1_task_0.pth" --losses_path "toy_toy_E3" --model_root "./models6/" --losses_root "./losses6/"
python3 Toy_train.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx 1 --model_name "toy" --last_model_path "./models6/toy_toy_E1_task_0.pth" --losses_path "toy_toy_E4" --model_root "./models6/" --losses_root "./losses6/"
python3 Toy_train.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx 1 --model_name "toy" --last_model_path "./models6/toy_toy_E2_task_0.pth" --losses_path "toy_toy_E5" --model_root "./models6/" --losses_root "./losses6/"
python3 Toy_train.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx 1 --model_name "toy" --last_model_path "./models6/toy_toy_E2_task_0.pth" --losses_path "toy_toy_E6" --model_root "./models6/" --losses_root "./losses6/"


echo "continual training for task 2"
python3 Toy_train.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx 2 --model_name "toy" --last_model_path "./models6/toy_toy_E3_task_1.pth" --losses_path "toy_toy_E3_task_2" --model_root "./models6/" --losses_root "./losses6/"
python3 Toy_train.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx 2 --model_name "toy" --last_model_path "./models6/toy_toy_E4_task_1.pth" --losses_path "toy_toy_E4_task_2" --model_root "./models6/" --losses_root "./losses6/"
python3 Toy_train.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx 2 --model_name "toy" --last_model_path "./models6/toy_toy_E5_task_1.pth" --losses_path "toy_toy_E5_task_2" --model_root "./models6/" --losses_root "./losses6/"
python3 Toy_train.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx 2 --model_name "toy" --last_model_path "./models6/toy_toy_E6_task_1.pth" --losses_path "toy_toy_E6_task_2" --model_root "./models6/" --losses_root "./losses6/"



# CKA between base model data between the task 1 and task 2 models
#echo "task 1 linear3"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E3_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E4_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E5_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E6_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear3"
#echo "task 2 linear3"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
# CKA between base model data between the task 1 and task 2 models
#echo "task 2 linear3"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E3_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E4_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E5_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E6_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"


#echo "task 1 linear2"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E3_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E4_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E5_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E6_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear2"
#echo "task 2 linear2"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"
#echo "task 2 linear2"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E3_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E4_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E5_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E6_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear2"


#echo "task 1 linear1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E3_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E4_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E5_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E6_task_1_task_1_data_0_train" --num_classes 2 --feature_name "linear1"
#echo "task 2 linear1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#echo "task 2 linear1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E3_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E4_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E5_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E6_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear1"



#echo "task 1 conv1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E3_task_1_task_1_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E4_task_1_task_1_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E5_task_1_task_1_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E6_task_1_task_1_data_0_train" --num_classes 2 --feature_name "conv1"
#echo "task 2 conv1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E1_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E2_task_0_task_0_data_0_train" --feature_path2 "./features6/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
#echo "task 2 conv1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E3_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E4_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E5_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features6/toy_toy_E6_task_1_task_1_data_0_train" --feature_path2 "./features6/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "conv1"



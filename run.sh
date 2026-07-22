#for i in $(seq 0 29); do
#  python3 Toy_features.py --model_path "./models/toy_toy_E1_${i}.pth" --inliers_id 0 --outliers_id -1 --data_path "./toy_data_train" --model_name "toy"
#done

# Finetune for session 0
#python3 Toy_finetune.py --experiment_name "E13" --dataset "toy" --classes_idx 2 --old_classes_idx 0 --model_name "cnn" --last_model_path "./models/cnn_toy_E1.pth" --losses_path "cnn_toy_E13"
#python3 Toy_finetune.py --experiment_name "E14" --dataset "toy" --classes_idx 3 --old_classes_idx 0 --model_name "cnn" --last_model_path "./models/cnn_toy_E1.pth" --losses_path "cnn_toy_E14"
#python3 Toy_finetune.py --experiment_name "E15" --dataset "toy" --classes_idx 4 --old_classes_idx 0 --model_name "cnn" --last_model_path "./models/cnn_toy_E1.pth" --losses_path "cnn_toy_E15" --lr 1e-1
#python3 Toy_finetune.py --experiment_name "E26" --dataset "toy" --classes_idx 5 --old_classes_idx 1 --model_name "cnn" --last_model_path "./models/cnn_toy_E2.pth" --losses_path "cnn_toy_E26"
#python3 Toy_finetune.py --experiment_name "E27" --dataset "toy" --classes_idx 6 --old_classes_idx 1 --model_name "cnn" --last_model_path "./models/cnn_toy_E2.pth" --losses_path "cnn_toy_E27"
#python3 Toy_finetune.py --experiment_name "E28" --dataset "toy" --classes_idx 7 --old_classes_idx 1 --model_name "cnn" --last_model_path "./models/cnn_toy_E2.pth" --losses_path "cnn_toy_E28" --lr 1e-1

#python3 Toy_finetune.py --experiment_name "E15" --dataset "toy" --classes_idx 4 --old_classes_idx 0 --model_name "toy" --last_model_path "./models/toy_toy_E1.pth" --losses_path "toy_toy_E15"
#python3 Toy_finetune.py --experiment_name "E28" --dataset "toy" --classes_idx 7 --old_classes_idx 1 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth" --losses_path "toy_toy_E28"

# continue training for session 1
#python3 Toy_train.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx 1 --model_name "toy" --last_model_path "./models/toy_toy_E1.pth" --losses_path "toy_toy_E3"
#python3 Toy_train.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx 1 --model_name "toy" --last_model_path "./models/toy_toy_E1.pth" --losses_path "toy_toy_E4"
#python3 Toy_train.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx 1 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth" --losses_path "toy_toy_E5"
#python3 Toy_train.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx 1 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth" --losses_path "toy_toy_E6"
#python3 Toy_train.py --experiment_name "E7" --dataset "toy" --experiment_idx 6 --task_idx 1 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth" --losses_path "toy_toy_E7"
#python3 Toy_train.py --experiment_name "E8" --dataset "toy" --experiment_idx 7 --task_idx 1 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth" --losses_path "toy_toy_E8"

# direct testing for session 1
#python3 toy_test.py --experiment_name "E1" --dataset "toy" --data_idx1 0 --model_name "toy" --model_path "./models/toy_toy_E1.pth"
#python3 toy_test.py --experiment_name "E2" --dataset "toy" --data_idx1 1 --model_name "toy" --model_path "./models/toy_toy_E2.pth"
#python3 toy_test.py --experiment_name "E3" --dataset "toy" --data_idx1 0 --model_name "toy" --model_path "./models/toy_toy_E3.pth"
#python3 toy_test.py --experiment_name "E4" --dataset "toy" --data_idx1 0 --model_name "toy" --model_path "./models/toy_toy_E4.pth"
#python3 toy_test.py --experiment_name "E5" --dataset "toy" --data_idx1 0 --model_name "toy" --model_path "./models/toy_toy_E5.pth"
#python3 toy_test.py --experiment_name "E6" --dataset "toy" --data_idx1 1 --model_name "toy" --model_path "./models/toy_toy_E6.pth"
#python3 toy_test.py --experiment_name "E7" --dataset "toy" --data_idx1 1 --model_name "toy" --model_path "./models/toy_toy_E7.pth"
#python3 toy_test.py --experiment_name "E8" --dataset "toy" --data_idx1 1 --model_name "toy" --model_path "./models/toy_toy_E8.pth"

#  linear probe of the init model with the task 1 data
#python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 0 --task_idx_data 1 --model_name "toy" --last_model_path "./models/toy_toy_E1.pth"
#python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 0 --task_idx_data 1 --model_name "toy" --last_model_path "./models/toy_toy_E1.pth"
#python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 0 --task_idx_data 1 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth"
#python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 0 --task_idx_data 1 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth"


#  linear probe of the task 1 model with the task 0 data
#python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 1 --task_idx_data 0 --model_name "toy" --last_model_path "./models/toy_toy_E3_task_1.pth"
#python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 1 --task_idx_data 0 --model_name "toy" --last_model_path "./models/toy_toy_E4_task_1.pth"
#python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 1 --task_idx_data 0 --model_name "toy" --last_model_path "./models/toy_toy_E5_task_1.pth"
#python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 1 --task_idx_data 0 --model_name "toy" --last_model_path "./models/toy_toy_E6_task_1.pth"


#python3 Toy_features.py --inliers_id 0 --model_data_id 0 --model_name "toy" --model_path "./models/toy_toy_E1.pth"
#python3 Toy_features.py --inliers_id 0 --model_data_id 1 --model_name "toy" --model_path "./models/toy_toy_E2.pth"
#python3 Toy_features.py --inliers_id 0 --model_data_id 2 --model_name "toy" --model_path "./models/toy_toy_E3.pth"
#python3 Toy_features.py --inliers_id 0 --model_data_id 3 --model_name "toy" --model_path "./models/toy_toy_E4.pth"
#python3 Toy_features.py --inliers_id 0 --model_data_id 4 --model_name "toy" --model_path "./models/toy_toy_E5.pth"
#python3 Toy_features.py --inliers_id 0 --model_data_id 5 --model_name "toy" --model_path "./models/toy_toy_E6.pth"
#python3 Toy_features.py --inliers_id 0 --model_data_id 6 --model_name "toy" --model_path "./models/toy_toy_E7.pth"
#python3 Toy_features.py --inliers_id 0 --model_data_id 7 --model_name "toy" --model_path "./models/toy_toy_E8.pth"


#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E1_train" --feature_path2 "./features/toy_toy_E3_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E1_train" --feature_path2 "./features/toy_toy_E4_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E1_train" --feature_path2 "./features/toy_toy_E5_train" --num_classes 2 --feature_name "conv1"

#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E2_train" --feature_path2 "./features/toy_toy_E6_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E2_train" --feature_path2 "./features/toy_toy_E7_train" --num_classes 2 --feature_name "conv1"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E2_train" --feature_path2 "./features/toy_toy_E8_train" --num_classes 2 --feature_name "conv1"

# continual training for task 2
python3 Toy_train.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx 2 --model_name "toy" --last_model_path "./models/toy_toy_E3_task_1.pth" --losses_path "toy_toy_E3_task_2"
python3 Toy_train.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx 2 --model_name "toy" --last_model_path "./models/toy_toy_E4_task_1.pth" --losses_path "toy_toy_E4_task_2"
#python3 Toy_train.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx 2 --model_name "toy" --last_model_path "./models/toy_toy_E5_task1.pth" --losses_path "toy_toy_E5_task_2"
#python3 Toy_train.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx 2 --model_name "toy" --last_model_path "./models/toy_toy_E6_task1.pth" --losses_path "toy_toy_E6_task_2"



# finetune to evalute the models after task2
#python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models/toy_toy_E3_task_2.pth"
#python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models/toy_toy_E4_task_2.pth"
#python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models/toy_toy_E5_task_2.pth"
#python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models/toy_toy_E6_task_2.pth"
#python3 Toy_finetune.py --experiment_name "E7" --dataset "toy" --experiment_idx 6 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models/toy_toy_E7_task_2.pth"
#python3 Toy_finetune.py --experiment_name "E8" --dataset "toy" --experiment_idx 7 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models/toy_toy_E8_task_2.pth"


#python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models/toy_toy_E3_task_2.pth"
#python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models/toy_toy_E4_task_2.pth"
#python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models/toy_toy_E5_task_2.pth"
#python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models/toy_toy_E6_task_2.pth"
#python3 Toy_finetune.py --experiment_name "E7" --dataset "toy" --experiment_idx 6 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models/toy_toy_E7_task_2.pth"
#python3 Toy_finetune.py --experiment_name "E8" --dataset "toy" --experiment_idx 7 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models/toy_toy_E8_task_2.pth"


# finetune to evalute the models after task1 on task 2 data
#python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models/toy_toy_E3.pth"
#python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models/toy_toy_E4.pth"
#python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models/toy_toy_E5.pth"
#python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models/toy_toy_E6.pth"
#python3 Toy_finetune.py --experiment_name "E7" --dataset "toy" --experiment_idx 6 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models/toy_toy_E7.pth"
#python3 Toy_finetune.py --experiment_name "E8" --dataset "toy" --experiment_idx 7 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models/toy_toy_E8.pth"


# finetune to evalute the init models  on task 2 data
#python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models/toy_toy_E1.pth"
#python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models/toy_toy_E1.pth"
#python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models/toy_toy_E1.pth"
#python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth"
#python3 Toy_finetune.py --experiment_name "E7" --dataset "toy" --experiment_idx 6 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth"
#python3 Toy_finetune.py --experiment_name "E8" --dataset "toy" --experiment_idx 7 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models/toy_toy_E2.pth"


# feature reading for task2
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 0 --model_name "toy" --model_path "./models/toy_toy_E1.pth"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 1 --model_name "toy" --model_path "./models/toy_toy_E2.pth"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 2 --model_name "toy" --model_path "./models/toy_toy_E3.pth"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 3 --model_name "toy" --model_path "./models/toy_toy_E4.pth"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 4 --model_name "toy" --model_path "./models/toy_toy_E5.pth"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 5 --model_name "toy" --model_path "./models/toy_toy_E6.pth"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 6 --model_name "toy" --model_path "./models/toy_toy_E7.pth"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 7 --model_name "toy" --model_path "./models/toy_toy_E8.pth"

# CKA between base model data between the task 1 and task 2 models
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E1_task_0_data_0_train" --feature_path2 "./features/toy_toy_E3_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E1_task_0_data_0_train" --feature_path2 "./features/toy_toy_E4_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E1_task_0_data_0_train" --feature_path2 "./features/toy_toy_E5_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E2_task_0_data_0_train" --feature_path2 "./features/toy_toy_E6_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E2_task_0_data_0_train" --feature_path2 "./features/toy_toy_E7_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
#python3 Toy_metrics.py --feature_path1 "./features/toy_toy_E2_task_0_data_0_train" --feature_path2 "./features/toy_toy_E8_task_2_task_2_data_0_train" --num_classes 2 --feature_name "linear3"
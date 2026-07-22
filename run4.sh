echo "continue training for task 1"
python3 Toy_train.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx 1 --model_name "toy" --last_model_path "./models4/toy_toy_E1.pth" --losses_path "toy_toy_E3" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_train.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx 1 --model_name "toy" --last_model_path "./models4/toy_toy_E1.pth" --losses_path "toy_toy_E4" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_train.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx 1 --model_name "toy" --last_model_path "./models4/toy_toy_E2.pth" --losses_path "toy_toy_E5" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_train.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx 1 --model_name "toy" --last_model_path "./models4/toy_toy_E2.pth" --losses_path "toy_toy_E6" --model_root "./models4/" --losses_root "./losses4/"



echo "linear probe of the init model with the task 1 data"
python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 0 --task_idx_data 1 --model_name "toy" --last_model_path "./models4/toy_toy_E1.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 0 --task_idx_data 1 --model_name "toy" --last_model_path "./models4/toy_toy_E1.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 0 --task_idx_data 1 --model_name "toy" --last_model_path "./models4/toy_toy_E2.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 0 --task_idx_data 1 --model_name "toy" --last_model_path "./models4/toy_toy_E2.pth" --model_root "./models4/" --losses_root "./losses4/"


echo "linear probe of the task 1 model with the task 0 data"
python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 1 --task_idx_data 0 --model_name "toy" --last_model_path "./models4/toy_toy_E3_task_1.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 1 --task_idx_data 0 --model_name "toy" --last_model_path "./models4/toy_toy_E4_task_1.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 1 --task_idx_data 0 --model_name "toy" --last_model_path "./models4/toy_toy_E5_task_1.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 1 --task_idx_data 0 --model_name "toy" --last_model_path "./models4/toy_toy_E6_task_1.pth" --model_root "./models4/" --losses_root "./losses4/"



echo "continual training for task 2"
python3 Toy_train.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx 2 --model_name "toy" --last_model_path "./models4/toy_toy_E3_task_1.pth" --losses_path "toy_toy_E3_task_2" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_train.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx 2 --model_name "toy" --last_model_path "./models4/toy_toy_E4_task_1.pth" --losses_path "toy_toy_E4_task_2" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_train.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx 2 --model_name "toy" --last_model_path "./models4/toy_toy_E5_task1.pth" --losses_path "toy_toy_E5_task_2" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_train.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx 2 --model_name "toy" --last_model_path "./models4/toy_toy_E6_task1.pth" --losses_path "toy_toy_E6_task_2" --model_root "./models4/" --losses_root "./losses4/"


echo "linear probe of the init model with the task 2 data"
python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models4/toy_toy_E1.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models4/toy_toy_E1.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models4/toy_toy_E2.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 0 --task_idx_data 2 --model_name "toy" --last_model_path "./models4/toy_toy_E2.pth" --model_root "./models4/" --losses_root "./losses4/"


echo "linear probe of the task 1 model with the task 2 data"
python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models4/toy_toy_E3_task_1.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models4/toy_toy_E4_task_1.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models4/toy_toy_E2_task_1.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 1 --task_idx_data 2 --model_name "toy" --last_model_path "./models4/toy_toy_E2_task_1.pth" --model_root "./models4/" --losses_root "./losses4/"


echo "linear probe of the task 2 model with the task 0 data"
python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models4/toy_toy_E3_task_2.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models4/toy_toy_E4_task_2.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models4/toy_toy_E5_task_2.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 2 --task_idx_data 0 --model_name "toy" --last_model_path "./models4/toy_toy_E6_task_2.pth" --model_root "./models4/" --losses_root "./losses4/"


echo "linear probe of the task 2 model with the task 1 data"
python3 Toy_finetune.py --experiment_name "E3" --dataset "toy" --experiment_idx 2 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models4/toy_toy_E3_task_2.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E4" --dataset "toy" --experiment_idx 3 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models4/toy_toy_E4_task_2.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E5" --dataset "toy" --experiment_idx 4 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models4/toy_toy_E5_task_2.pth" --model_root "./models4/" --losses_root "./losses4/"
python3 Toy_finetune.py --experiment_name "E6" --dataset "toy" --experiment_idx 5 --task_idx_model 2 --task_idx_data 1 --model_name "toy" --last_model_path "./models4/toy_toy_E6_task_2.pth" --model_root "./models4/" --losses_root "./losses4/"

#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 13 --expand_data 1.5
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 14 --expand_data 1.5
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 15 --expand_data 1.5

#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_13_128_128/last.pth" --last_trail "13"
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_14_128_128/last.pth" --last_trail "14"
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_15_128_128/last.pth" --last_trail "15"

#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_13/last.pth" --last_trail "113"
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_14/last.pth" --last_trail "114"
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_15/last.pth" --last_trail "115"


echo "Session 0 models on data 1"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_13_128_128_data_3_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_13_128_128_data_3_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_14_128_128_data_3_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_14_128_128_data_3_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_15_128_128_data_3_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_15_128_128_data_3_test_known"


echo "Session 0 models on data 2"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_13_128_128_data_4_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_13_128_128_data_4_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_14_128_128_data_4_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_14_128_128_data_4_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_15_128_128_data_4_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_15_128_128_data_4_test_known"


echo "Session 1 models on data 0"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_3_128_128_last_13_data_13_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_3_128_128_last_13_data_13_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_3_128_128_last_14_data_14_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_3_128_128_last_14_data_14_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_3_128_128_last_15_data_15_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_3_128_128_last_15_data_15_test_known"

echo "Session 1 models on data 2"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_3_128_128_last_13_data_4_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_3_128_128_last_13_data_4_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_3_128_128_last_14_data_4_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_3_128_128_last_14_data_4_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_3_128_128_last_15_data_4_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_3_128_128_last_15_data_4_test_known"


echo "Session 2 models on data 0"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_4_128_128_last_113_data_13_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_4_128_128_last_113_data_13_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_4_128_128_last_114_data_14_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_4_128_128_last_114_data_14_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_4_128_128_last_115_data_15_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_4_128_128_last_115_data_15_test_known"

echo "Session 2 models on data 1"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_4_128_128_last_113_data_3_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_4_128_128_last_113_data_3_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_4_128_128_last_114_data_3_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_4_128_128_last_114_data_3_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_4_128_128_last_115_data_3_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_4_128_128_last_115_data_3_test_known"

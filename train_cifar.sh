python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 0 --expand_data 3
python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 1 --expand_data 3
python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 2 --expand_data 3

python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 5 --expand_data 1.5
python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 6 --expand_data 1.5
python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 7 --expand_data 1.5

python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 8

########################################################################################################################################

python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_0_128_128/last.pth" --last_trail 0
python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_1_128_128/last.pth" --last_trail 1
python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_2_128_128/last.pth" --last_trail 2

python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_5_128_128/last.pth" --last_trail 5
python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_6_128_128/last.pth" --last_trail 6
python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_7_128_128/last.pth" --last_trail 7

python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_8_128_128/last.pth" --last_trail 8

########################################################################################################################################

python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_0/last.pth" --last_trail "03"
python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_1/last.pth" --last_trail "13"
python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_2/last.pth" --last_trail "23"


python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_5/last.pth" --last_trail "33"
python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_6/last.pth" --last_trail "43"
python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_7/last.pth" --last_trail "53"

python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --last_trail "63"

########################################################################################################################################
# Session 0 models on D3
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_0_128_128_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_0_128_128_data_3_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_1_128_128_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_1_128_128_data_3_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_2_128_128_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_2_128_128_data_3_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_5_128_128_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_5_128_128_data_3_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_6_128_128_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_6_128_128_data_3_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_7_128_128_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_7_128_128_data_3_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_8_128_128_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_8_128_128_data_3_test_known"

########################################################################################################################################
# Session 0 models on D4
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_0_128_128_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_0_128_128_data_4_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_1_128_128_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_1_128_128_data_4_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_2_128_128_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_2_128_128_data_4_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_5_128_128_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_5_128_128_data_4_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_6_128_128_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_6_128_128_data_4_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_7_128_128_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_7_128_128_data_4_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_8_128_128_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_8_128_128_data_4_test_known"

########################################################################################################################################
# Session 1 models on D4
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_0_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_0_data_4_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_1_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_1_data_4_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_2_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_2_data_4_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_5_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_5_data_4_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_6_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_6_data_4_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_7_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_7_data_4_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_8_data_4_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_8_data_4_test_known"

########################################################################################################################################
# Session 1 models on Session 0 data
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_0_data_0_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_0_data_0_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_1_data_1_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_1_data_1_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_2_data_2_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_2_data_2_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_5_data_5_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_5_data_5_test_known" --remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_6_data_6_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_6_data_6_test_known" --remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_7_data_7_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_7_data_7_test_known" --remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_3_128_128_last_8_data_8_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_3_128_128_last_8_data_8_test_known" --remove_extra_classes 1

########################################################################################################################################
# Session 2 models on Session 0 data
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_03_data_0_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_03_data_0_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_13_data_1_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_13_data_1_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_23_data_2_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_23_data_2_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_33_data_5_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_33_data_5_test_known" --remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_43_data_6_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_43_data_6_test_known" --remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_53_data_7_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_53_data_7_test_known" --remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_63_data_8_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_63_data_8_test_known" --remove_extra_classes 1

########################################################################################################################################
# Session 2 models on Session 1 data
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_03_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_03_data_3_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_13_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_13_data_3_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_23_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_23_data_3_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_33_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_33_data_3_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_43_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_43_data_3_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_53_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_53_data_3_test_known"
python3 main_probe.py --feature_path_train "./features/cifar100_marco_resnet18_1trail_4_128_128_last_63_data_3_train" --feature_path_test "./features/cifar100_marco_resnet18_1trail_4_128_128_last_63_data_3_test_known"
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 0 --expand_data 3
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 1 --expand_data 3
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 2 --expand_data 3

#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 5 --expand_data 1.5
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 6 --expand_data 1.5
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 7 --expand_data 1.5

#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 8

########################################################################################################################################

#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_0_128_128/last.pth" --last_trail 0
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_1_128_128/last.pth" --last_trail 1
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_2_128_128/last.pth" --last_trail 2

#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_5_128_128/last.pth" --last_trail 5
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_6_128_128/last.pth" --last_trail 6
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_7_128_128/last.pth" --last_trail 7

#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_8_128_128/last.pth" --last_trail 8

########################################################################################################################################

#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_0/last.pth" --last_trail "03"
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_1/last.pth" --last_trail "13"
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_2/last.pth" --last_trail "23"


#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_5/last.pth" --last_trail "33"
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_6/last.pth" --last_trail "43"
#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_7/last.pth" --last_trail "53"

#python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --last_trail "63"

########################################################################################################################################
echo "Session 0 models on M0"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_0_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_0_128_128_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_1_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_1_128_128_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_2_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_2_128_128_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_5_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_5_128_128_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_6_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_6_128_128_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_7_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_7_128_128_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_8_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_8_128_128_data_11_test_known"

########################################################################################################################################
echo "Session 0 models on M1"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_0_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_0_128_128_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_1_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_1_128_128_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_2_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_2_128_128_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_5_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_5_128_128_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_6_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_6_128_128_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_7_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_7_128_128_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_8_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_8_128_128_data_12_test_known"

########################################################################################################################################
echo "Session 1 models on M1"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_0_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_0_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_1_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_1_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_2_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_2_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_5_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_5_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_6_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_6_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_7_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_7_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_8_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_8_data_12_test_known"

########################################################################################################################################
echo "Session 1 models on Session 0 data"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_0_data_0_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_0_data_0_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_1_data_1_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_1_data_1_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_2_data_2_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_2_data_2_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_5_data_5_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_5_data_5_test_known" #--remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_6_data_6_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_6_data_6_test_known" #--remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_7_data_7_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_7_data_7_test_known" #--remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_8_data_8_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_8_data_8_test_known" #--remove_extra_classes 1

python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_5_data_5_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_5_data_5_test_known" --remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_6_data_6_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_6_data_6_test_known" --remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_7_data_7_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_7_data_7_test_known" --remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_8_data_8_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_8_data_8_test_known" --remove_extra_classes 1
########################################################################################################################################
echo "Session 2 models on Session 0 data"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_011_data_0_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_011_data_0_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_111_data_1_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_111_data_1_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_211_data_2_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_211_data_2_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_311_data_5_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_311_data_5_test_known" #--remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_411_data_6_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_411_data_6_test_known" #--remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_511_data_7_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_511_data_7_test_known" #--remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_611_data_8_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_611_data_8_test_known" #--remove_extra_classes 1

python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_311_data_5_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_311_data_5_test_known" --remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_411_data_6_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_411_data_6_test_known" --remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_511_data_7_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_511_data_7_test_known" --remove_extra_classes 1
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_611_data_8_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_611_data_8_test_known" --remove_extra_classes 1
########################################################################################################################################
echo "Session 2 models on Session 1 data"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_011_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_011_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_111_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_111_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_211_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_211_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_311_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_311_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_411_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_411_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_511_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_511_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_611_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_12_128_128_last_611_data_11_test_known"

########################################################################################################################################
echo "Session 0 models on M0"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_0_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_0_128_128_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_1_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_1_128_128_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_2_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_2_128_128_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_5_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_5_128_128_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_6_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_6_128_128_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_7_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_7_128_128_data_11_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_8_128_128_data_11_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_8_128_128_data_11_test_known"

########################################################################################################################################
echo "Session 0 models on M1"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_0_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_0_128_128_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_1_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_1_128_128_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_2_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_2_128_128_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_5_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_5_128_128_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_6_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_6_128_128_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_7_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_7_128_128_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_8_128_128_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_8_128_128_data_12_test_known"

########################################################################################################################################
echo "Session 1 models on F1"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_0_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_0_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_1_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_1_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_2_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_2_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_5_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_5_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_6_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_6_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_7_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_7_data_12_test_known"
python3 main_probe.py --feature_path_train "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_8_data_12_train" --feature_path_test "./features2/cifar100_marco_resnet18_1trail_11_128_128_last_8_data_12_test_known"

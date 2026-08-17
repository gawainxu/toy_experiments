python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_0/last.pth" --last_trail "03"
python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --last_trail "63"

# session 1 models on D4
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_0/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_0/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_1/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_1/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_2/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_2/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_5/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_5/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_6/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_6/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_7/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_7/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --if_train 'test_known'


#  Session 2 models on Session 0 data
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 0 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_03/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 0 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_03/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 1 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_13/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 1 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_13/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 2 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_23/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 2 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_23/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 5 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_33/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 5 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_33/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 6 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_43/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 6 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_43/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 7 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_53/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 7 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_53/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 8 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_63/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 8 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_63/last.pth" --if_train 'test_known'


#  Session 2 models on Session 1 data
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_03/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_03/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_13/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_13/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_23/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_23/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_33/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_33/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_43/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_43/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_53/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_53/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_63/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_63/last.pth" --if_train 'test_known'
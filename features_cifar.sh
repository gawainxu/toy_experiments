# session 1 models on session 0 data
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 0 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_0/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 0 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_0/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 1 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_1/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 1 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_1/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 2 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_2/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 2 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_2/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 5 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_5/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 5 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_5/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 6 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_6/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 6 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_6/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 7 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_7/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 7 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_7/last.pth" --if_train 'test_known'

#python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 8 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --if_train 'train'
#python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 8 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --if_train 'test_known'
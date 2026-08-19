python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 8 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 8 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --if_train 'test_known'


python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 8 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_63/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 8 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_63/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_63/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_63/last.pth" --if_train 'test_known'
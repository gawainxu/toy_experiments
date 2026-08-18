python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 8

python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 3 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_8_128_128/last.pth" --last_trail 8

python3 main_ce.py --print_freq 20 --save_freq 50 --batch_size 128 --epochs 300 --model 'resnet18' --datasets 'cifar100_marco' --trail 4 --last_model_path "./save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --last_trail "63"


python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 8 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 8 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --if_train 'test_known'


python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --trail 4 --model_trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_3_128_128_last_8/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 8 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_63/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 8 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_63/last.pth" --if_train 'test_known'

python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_63/last.pth" --if_train 'train'
python3 feature_reading_old.py --datasets 'cifar100_marco' --model "resnet18" --model_trail 4 --trail 3 --model_path "/save/CE/cifar100_marco_models/cifar100_marco_resnet18_1trail_4_128_128_last_63/last.pth" --if_train 'test_known'
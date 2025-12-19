#python3 Toy_train.py --dataset "cifar" --model_name "cnn"  --data_size 32 --classes_idx 1 --freeze True --last_model_path "./models/cnn_cifar_0_0.pth_cifar_0_0.pth" --old_classes_idx 0
python3 Toy_train.py --dataset "cifar" --model_name "cnn"  --data_size 32 --classes_idx 2 --freeze True --last_model_path "./models/cnn_cifar_0_0.pth_cifar_0_0.pth" --old_classes_idx 0 --freeze_layers "conv"
python3 Toy_train.py --dataset "cifar" --model_name "cnn"  --data_size 32 --classes_idx 3 --freeze True --last_model_path "./models/cnn_cifar_0_0.pth_cifar_0_0.pth" --old_classes_idx 0 --freeze_layers "conv"

 

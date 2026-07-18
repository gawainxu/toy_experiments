#echo "For E2"
#RESULTS_FILE="results2.txt"
#echo "inliers_id, result" > $RESULTS_FILE

#for i in $(seq 0 99); do
#  echo "Running iteration $i"
#  result=$(python3 Toy_distances.py --feature_path "./features/toy_toy_E2_${i}_train" --feature_path_test "./features/toy_toy_E2_${i}_rectangle_blue" --num_classes 3 --feature_to_visualize "linear2" --fig_save_path "./plots/hist_E2_${i}.png")
#  echo "$i, $result" >> $RESULTS_FILE
#done


#python3 Toy_train.py --experiment_name "E1" --dataset "toy" --classes_idx 0 --model_name "cnn" --losses_path "cnn_toy_E1"
#python3 Toy_train.py --experiment_name "E2" --dataset "toy" --classes_idx 1 --model_name "cnn" --losses_path "cnn_toy_E2"

python3 Toy_train.py --experiment_name "cifar1" --dataset "cifar" --classes_idx 0 --old_classes_idx 0 --model_name "toy" --losses_path "toy_toy_cifar1"
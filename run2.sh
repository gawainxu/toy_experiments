echo "For E2"
RESULTS_FILE="results2.txt"
echo "inliers_id, result" > $RESULTS_FILE

for i in $(seq 0 99); do
  echo "Running iteration $i"
  result=$(python3 Toy_distances.py --feature_path "./features/toy_toy_E2_${i}_train" --feature_path_test "./features/toy_toy_E2_${i}_rectangle_blue" --num_classes 3 --feature_to_visualize "linear2" --fig_save_path "./plots/hist_E2_${i}.png")
  echo "$i, $result" >> $RESULTS_FILE
done



echo "For E1"
RESULTS_FILE="results1.txt"
echo "inliers_id, result" > $RESULTS_FILE

for i in $(seq 0 99); do
  echo "Running iteration $i"
  result=$(python3 Toy_distances.py --feature_path "./features/toy_toy_E1_${i}_train" --feature_path_test "./features/toy_toy_E1_${i}_rectangle_blue" --num_classes 2 --feature_to_visualize "linear2" --fig_save_path "./plots/hist_E1_${i}.png")
  echo "$i, $result" >> $RESULTS_FILE
done

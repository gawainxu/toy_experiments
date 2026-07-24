echo "feature reading for task0"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 0 --model_name "toy" --model_path "./models/toy_toy_E1_task_0.pth" --feature_save_path "./features/"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 1 --model_name "toy" --model_path "./models/toy_toy_E2_task_0.pth" --feature_save_path "./features/"


echo "feature reading for task0"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 0 --model_name "toy" --model_path "./models2/toy_toy_E1_task_0.pth" --feature_save_path "./features2/"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 1 --model_name "toy" --model_path "./models2/toy_toy_E2_task_0.pth" --feature_save_path "./features2/"


echo "feature reading for task0"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 0 --model_name "toy" --model_path "./models3/toy_toy_E1_task_0.pth" --feature_save_path "./features3/"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 1 --model_name "toy" --model_path "./models3/toy_toy_E2_task_0.pth" --feature_save_path "./features3/"
3

echo "feature reading for task0"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 0 --model_name "toy" --model_path "./models4/toy_toy_E1_task_0.pth" --feature_save_path "./features4/"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 1 --model_name "toy" --model_path "./models4/toy_toy_E2_task_0.pth" --feature_save_path "./features4/"


echo "feature reading for task0"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 0 --model_name "toy" --model_path "./models5/toy_toy_E1_task_0.pth" --feature_save_path "./features5/"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 1 --model_name "toy" --model_path "./models5/toy_toy_E2_task_0.pth" --feature_save_path "./features5/"


echo "feature reading for task0"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 0 --model_name "toy" --model_path "./models6/toy_toy_E1_task_0.pth" --feature_save_path "./features6/"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 1 --model_name "toy" --model_path "./models6/toy_toy_E2_task_0.pth" --feature_save_path "./features6/"


echo "feature reading for task0"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 0 --model_name "toy" --model_path "./models7/toy_toy_E1_task_0.pth" --feature_save_path "./features7/"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 1 --model_name "toy" --model_path "./models7/toy_toy_E2_task_0.pth" --feature_save_path "./features7/"


echo "feature reading for task0"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 0 --model_name "toy" --model_path "./models8/toy_toy_E1_task_0.pth" --feature_save_path "./features8/"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 1 --model_name "toy" --model_path "./models8/toy_toy_E2_task_0.pth" --feature_save_path "./features8/"


echo "feature reading for task0"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 0 --model_name "toy" --model_path "./models9/toy_toy_E1_task_0.pth" --feature_save_path "./features9/"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 1 --model_name "toy" --model_path "./models9/toy_toy_E2_task_0.pth" --feature_save_path "./features9/"


echo "feature reading for task0"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 0 --model_name "toy" --model_path "./models10/toy_toy_E1_task_0.pth" --feature_save_path "./features10/"
python3 Toy_features.py --task_idx_data 0 --task_idx_model 0 --experiment_idx 1 --model_name "toy" --model_path "./models10/toy_toy_E2_task_0.pth" --feature_save_path "./features10/"





echo "feature reading for model 0 with task 1 data (shape)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 3 --model_name "toy" --model_path "./models/toy_toy_E1_task_0.pth" --feature_save_path "./features/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 5 --model_name "toy" --model_path "./models/toy_toy_E2_task_0.pth" --feature_save_path "./features/" --data_path "./toy_data_test_inliers"

echo "feature reading for model 0 with task 1 data (color)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 2 --model_name "toy" --model_path "./models/toy_toy_E1_task_0.pth" --feature_save_path "./features/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 4 --model_name "toy" --model_path "./models/toy_toy_E2_task_0.pth" --feature_save_path "./features/" --data_path "./toy_data_test_inliers"



echo "feature reading for model 0 with task 1 data (shape)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 3 --model_name "toy" --model_path "./models2/toy_toy_E1_task_0.pth" --feature_save_path "./features2/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 5 --model_name "toy" --model_path "./models2/toy_toy_E2_task_0.pth" --feature_save_path "./features2/" --data_path "./toy_data_test_inliers"

echo "feature reading for model 0 with task 1 data (color)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 2 --model_name "toy" --model_path "./models2/toy_toy_E1_task_0.pth" --feature_save_path "./features2/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 4 --model_name "toy" --model_path "./models2/toy_toy_E2_task_0.pth" --feature_save_path "./features2/" --data_path "./toy_data_test_inliers"



echo "feature reading for model 0 with task 1 data (shape)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 3 --model_name "toy" --model_path "./models3/toy_toy_E1_task_0.pth" --feature_save_path "./features3/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 5 --model_name "toy" --model_path "./models3/toy_toy_E2_task_0.pth" --feature_save_path "./features3/" --data_path "./toy_data_test_inliers"

echo "feature reading for model 0 with task 1 data (color)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 2 --model_name "toy" --model_path "./models3/toy_toy_E1_task_0.pth" --feature_save_path "./features3/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 4 --model_name "toy" --model_path "./models3/toy_toy_E2_task_0.pth" --feature_save_path "./features3/" --data_path "./toy_data_test_inliers"



echo "feature reading for model 0 with task 1 data (shape)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 3 --model_name "toy" --model_path "./models4/toy_toy_E1_task_0.pth" --feature_save_path "./features4/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 5 --model_name "toy" --model_path "./models4/toy_toy_E2_task_0.pth" --feature_save_path "./features4/" --data_path "./toy_data_test_inliers"

echo "feature reading for model 0 with task 1 data (color)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 2 --model_name "toy" --model_path "./models4/toy_toy_E1_task_0.pth" --feature_save_path "./features4/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 4 --model_name "toy" --model_path "./models4/toy_toy_E2_task_0.pth" --feature_save_path "./features4/" --data_path "./toy_data_test_inliers"



echo "feature reading for model 0 with task 1 data (shape)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 3 --model_name "toy" --model_path "./models5/toy_toy_E1_task_0.pth" --feature_save_path "./features5/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 5 --model_name "toy" --model_path "./models5/toy_toy_E2_task_0.pth" --feature_save_path "./features5/" --data_path "./toy_data_test_inliers"

echo "feature reading for model 0 with task 1 data (color)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 2 --model_name "toy" --model_path "./models5/toy_toy_E1_task_0.pth" --feature_save_path "./features5/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 4 --model_name "toy" --model_path "./models5/toy_toy_E2_task_0.pth" --feature_save_path "./features5/" --data_path "./toy_data_test_inliers"




echo "feature reading for model 0 with task 1 data (shape)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 3 --model_name "toy" --model_path "./models6/toy_toy_E1_task_0.pth" --feature_save_path "./features6/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 5 --model_name "toy" --model_path "./models6/toy_toy_E2_task_0.pth" --feature_save_path "./features6/" --data_path "./toy_data_test_inliers"

echo "feature reading for model 0 with task 1 data (color)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 2 --model_name "toy" --model_path "./models6/toy_toy_E1_task_0.pth" --feature_save_path "./features6/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 4 --model_name "toy" --model_path "./models6/toy_toy_E2_task_0.pth" --feature_save_path "./features6/" --data_path "./toy_data_test_inliers"



echo "feature reading for model 0 with task 1 data (shape)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 3 --model_name "toy" --model_path "./models7/toy_toy_E1_task_0.pth" --feature_save_path "./features7/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 5 --model_name "toy" --model_path "./models7/toy_toy_E2_task_0.pth" --feature_save_path "./features7/" --data_path "./toy_data_test_inliers"

echo "feature reading for model 0 with task 1 data (color)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 2 --model_name "toy" --model_path "./models7/toy_toy_E1_task_0.pth" --feature_save_path "./features7/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 4 --model_name "toy" --model_path "./models7/toy_toy_E2_task_0.pth" --feature_save_path "./features7/" --data_path "./toy_data_test_inliers"



echo "feature reading for model 0 with task 1 data (shape)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 3 --model_name "toy" --model_path "./models8/toy_toy_E1_task_0.pth" --feature_save_path "./features8/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 5 --model_name "toy" --model_path "./models8/toy_toy_E2_task_0.pth" --feature_save_path "./features8/" --data_path "./toy_data_test_inliers"

echo "feature reading for model 0 with task 1 data (color)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 2 --model_name "toy" --model_path "./models8/toy_toy_E1_task_0.pth" --feature_save_path "./features8/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 4 --model_name "toy" --model_path "./models8/toy_toy_E2_task_0.pth" --feature_save_path "./features8/" --data_path "./toy_data_test_inliers"



echo "feature reading for model 0 with task 1 data (shape)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 3 --model_name "toy" --model_path "./models9/toy_toy_E1_task_0.pth" --feature_save_path "./features9/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 5 --model_name "toy" --model_path "./models9/toy_toy_E2_task_0.pth" --feature_save_path "./features9/" --data_path "./toy_data_test_inliers"

echo "feature reading for model 0 with task 1 data (color)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 2 --model_name "toy" --model_path "./models9/toy_toy_E1_task_0.pth" --feature_save_path "./features9/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 4 --model_name "toy" --model_path "./models9/toy_toy_E2_task_0.pth" --feature_save_path "./features9/" --data_path "./toy_data_test_inliers"



echo "feature reading for model 0 with task 1 data (shape)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 3 --model_name "toy" --model_path "./models10/toy_toy_E1_task_0.pth" --feature_save_path "./features10/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 5 --model_name "toy" --model_path "./models10/toy_toy_E2_task_0.pth" --feature_save_path "./features10/" --data_path "./toy_data_test_inliers"

echo "feature reading for model 0 with task 1 data (color)"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 2 --model_name "toy" --model_path "./models10/toy_toy_E1_task_0.pth" --feature_save_path "./features10/" --data_path "./toy_data_test_inliers"
python3 Toy_features.py --task_idx_data 1 --task_idx_model 0 --experiment_idx 4 --model_name "toy" --model_path "./models10/toy_toy_E2_task_0.pth" --feature_save_path "./features10/" --data_path "./toy_data_test_inliers"

#echo "feature reading for task1"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 2 --model_name "toy" --model_path "./models/toy_toy_E3_task_1.pth" --feature_save_path "./features/"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 3 --model_name "toy" --model_path "./models/toy_toy_E4_task_1.pth" --feature_save_path "./features/"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 4 --model_name "toy" --model_path "./models/toy_toy_E5_task_1.pth" --feature_save_path "./features/"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 1 --experiment_idx 5 --model_name "toy" --model_path "./models/toy_toy_E6_task_1.pth" --feature_save_path "./features/"


#echo "feature reading for task2"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 2 --experiment_idx 2 --model_name "toy" --model_path "./models/toy_toy_E3_task_2.pth" --feature_save_path "./features/"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 2 --experiment_idx 3 --model_name "toy" --model_path "./models/toy_toy_E4_task_2.pth" --feature_save_path "./features/"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 2 --experiment_idx 4 --model_name "toy" --model_path "./models/toy_toy_E5_task_2.pth" --feature_save_path "./features/"
#python3 Toy_features.py --task_idx_data 0 --task_idx_model 2 --experiment_idx 5 --model_name "toy" --model_path "./models/toy_toy_E6_task_2.pth" --feature_save_path "./features/"

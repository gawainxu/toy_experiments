import pickle
import numpy as np
import matplotlib.pyplot as plt


distance_save_path2 = "D://projects//open_cross_entropy//save//version1//toy_model_distance2"
with open(distance_save_path2, "rb") as f:
        (similarity2_11, similarity2_41, similarity2_51, similarity2_61, 
         similarity2_22, similarity2_42, similarity2_52, similarity2_62) = pickle.load(f)


distance_save_path3 = "D://projects//open_cross_entropy//save//version1//toy_model_distance3"
with open(distance_save_path3, "rb") as f:
        (similarity3_11, similarity3_41, similarity3_51, similarity3_61, 
         similarity3_22, similarity3_42, similarity3_52, similarity3_62,
         similarity3_33, similarity3_43, similarity3_53, similarity3_63) = pickle.load(f)

"""
S_mean_cb2 = np.mean(np.abs(similarity2_11))
S_mean_cb3 = np.mean(np.abs(similarity3_11))

S_mean_tr2 = np.mean(np.abs(similarity2_22))
S_mean_tr3 = np.mean(np.abs(similarity3_22))

print("S_mean_cb2", S_mean_cb2)
print("S_mean_cb3", S_mean_cb3)
print("S_mean_tr2", S_mean_tr2)
print("S_mean_tr3", S_mean_tr3)
"""

"""
S_mean_cb2 = np.mean(np.abs(similarity2_11))
S_mean_cb3 = np.mean(np.abs(similarity3_11))

S_mean_cg2 = np.mean(np.abs(similarity2_51))
S_mean_cg3 = np.mean(np.abs(similarity3_51))

print("S_mean_cb2", S_mean_cb2)
print("S_mean_cb3", S_mean_cb3)
print("S_mean_tb2", S_mean_cg2)
print("S_mean_tb3", S_mean_cg3)

print("abs substraction class 2", np.abs(S_mean_cb2-S_mean_cg2))
print("abs substraction class 3", np.abs(S_mean_cb3-S_mean_cg3))
"""


S_mean_tr2 = np.mean(np.abs(similarity2_22))
S_mean_tr3 = np.mean(np.abs(similarity3_22))

S_mean_tg2 = np.mean(np.abs(similarity2_61))
S_mean_tg3 = np.mean(np.abs(similarity3_61))

print("S_mean_cb2", S_mean_tr2)
print("S_mean_cb3", S_mean_tr3)
print("S_mean_tb2", S_mean_tg2)
print("S_mean_tb3", S_mean_tg3)

print("abs substraction class 2", np.abs(S_mean_tr2-S_mean_tg2))
print("abs substraction class 3", np.abs(S_mean_tr3-S_mean_tg3))


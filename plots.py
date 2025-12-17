import matplotlib.pyplot as plt

"""
steps = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]

E1 = [0.5, 0.59, 0.77, 0.81, 0.77, 0.77, 0.75, 0.75, 0.76, 0.76, 0.76, 0.75, 0.76, 0.75, 0.75, 0.75]
E2 = [0.666666667, 0.665551839,	0.666666667, 0.666666667, 0.666666667, 0.706666667,	0.766666667, 0.773333333, 
      0.846666667,	0.766666667, 0.86,	0.766666667, 0.84,	0.806666667, 0.766666667, 0.879598662]	

import pickle

path2 = "D:\projects\open_cross_entropy\save1\losses_model3_E1"
path3 = "D:\projects\open_cross_entropy\save2\losses_model3_E2"

with open(path2, "rb") as f:
        losses2 = pickle.load(f)

with open(path3, "rb") as f:
        losses3 = pickle.load(f)

_, acc2, _ = losses2
_, acc3, _ = losses3


acc2 = [acc2[i-1] for i in steps]
acc3 = [acc3[i-1] + 0.02 for i in steps]

fig,ax = plt.subplots()
l11, = ax.plot(steps, E1, label='Histogram Distance in E1', color="red", linewidth=2.5)
l12, = ax.plot(steps, E2, label='Histogram Distance in E2', color="orange", linewidth=2.5)

ax.set_xlabel("Epochs")
ax.set_ylabel("Histogram Distance", color="red", fontsize=12)

ax2=ax.twinx()
l21, = ax2.plot(steps, acc2, label='Experiment 1 Accuracy', color="blue", linewidth=2.5)
l22, = ax2.plot(steps, acc3, label='Experiment 2 Accuracy', color="purple", linewidth=2.5)
ax2.set_ylabel("Classification Accuracy", color="blue", fontsize=12)

plt.title("Histogram Distance and Classification Accuracy", fontsize=14)
plt.legend([l11, l12, l21, l22], ['E1 Histogram Distance', 'E2 Histogram Distance', 'E1 Accuracy', 'E2 Accuracy'],  prop={'size': 10})
plt.savefig("D:\projects\open_cross_entropy\code\differences_acc.png")
"""


steps = [0, 20, 40, 60, 80, 100]
rectangle_blue_e1 = [0.2613, 0.7136, 0.8342, 0.8643, 0.8543, 0.8643]
rectangle_blue_e2 = [0.2915, 0.9146, 0.9346, 0.9548, 0.9848, 0.9548]

circle_green_e1 = [0.9246, 0.9246, 0.9548, 0.9648,	0.9648,	0.9648]
circle_green_e2 = [0.9246, 1, 1, 1, 1, 1]

rectangle_green_e1 = [0.3317, 0.402, 0.402, 0.4523, 0.4422, 0.4422]
rectangle_green_e2 = [0.412, 0.4824, 0.4522, 0.402,	0.4221,	0.4221]

ellipse_blue_e1 = [0.4422, 0.7839, 0.8442, 0.8342, 0.8442, 0.8442]
ellipse_blue_e2 = [0.412, 0.8844, 0.8945, 0.9045, 0.9146, 0.8945]

ellipse_pink_e1 = [0.503, 0.9749, 1, 1,	1, 1]
ellipse_pink_e2 = [0.6533, 1, 1, 1,	1, 1]


plt.plot(steps, rectangle_blue_e1, color="blue", marker='s', linestyle='--', linewidth=2.5, label="Blue Rectangle E1")
plt.plot(steps, rectangle_blue_e2, color="blue", marker='s', linestyle='-', linewidth=2.5, label="Blue Rectangle E2")

plt.plot(steps, circle_green_e1, color="green", marker='o', linestyle='--', linewidth=2.5, label="Green Circle E1")
plt.plot(steps, circle_green_e2, color="green", marker='o', linestyle='-', linewidth=2.5, label="Green Circle E2")

plt.plot(steps, rectangle_green_e1, color="green", marker='s', linestyle='--', linewidth=2.5, label="Green Rectangle E1")
plt.plot(steps, rectangle_green_e2, color="green", marker='s', linestyle='-', linewidth=2.5, label="Green Rectangle E2")

plt.plot(steps, ellipse_blue_e1, color="blue", marker='*', linestyle='--', linewidth=2.5, label="Blue Ellipse E1")
plt.plot(steps, ellipse_blue_e2, color="blue", marker='*', linestyle='-', linewidth=2.5, label="Blue Ellipse E2")

plt.plot(steps, ellipse_pink_e1, color="pink", marker='*', linestyle='--', linewidth=2.5, label="Pink Ellipse E1")
plt.plot(steps, ellipse_pink_e2, color="pink", marker='*', linestyle='-', linewidth=2.5, label="Pink Ellipse E2")


plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Distance", fontsize=15)
plt.title("Histogram Distances between Close- and Open-sets", fontsize=17)
plt.legend()
#plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
#plt.tight_layout()
plt.savefig("Histogram Distances.pdf")

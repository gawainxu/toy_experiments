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

values1 = []

with open("results1.txt", "r") as f:
    lines = f.readlines()

for line in lines[2::2]:
    values1.append(float(line.rstrip()))


values2 = []

with open("results2.txt", "r") as f:
    lines = f.readlines()

for line in lines[2::2]:
    values2.append(float(line.rstrip()))

plt.plot(values1, label="E1")
plt.plot(values2, label="E2")
plt.legend()
plt.savefig("./plots/his_dis.pdf")
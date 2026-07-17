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
"""


import pickle
import matplotlib.pyplot as plt


losses_file1 = "./losses/toy_toy_E1"
with open(losses_file1, "rb") as f1:
    losses1, accs1 = pickle.load(f1)
    

losses_file2 = "./losses/toy_toy_E2"
with open(losses_file2, "rb") as f2:
    losses2, accs2 = pickle.load(f2)


losses_file3 = "./losses/toy_toy_E3"
with open(losses_file3, "rb") as f3:
    losses3, accs3 = pickle.load(f3)
    

losses_file4 = "./losses/toy_toy_E4"
with open(losses_file4, "rb") as f4:
    losses4, accs4 = pickle.load(f4)
    
    
losses_file5 = "./losses/toy_toy_E5"
with open(losses_file5, "rb") as f5:
    losses5, accs5 = pickle.load(f5)
    
    
losses_file6 = "./losses/toy_toy_E6"
with open(losses_file6, "rb") as f6:
    losses6, accs6 = pickle.load(f6)
    

losses_file7 = "./losses/toy_toy_E7"
with open(losses_file7, "rb") as f7:
    losses7, accs7 = pickle.load(f7)
    

losses_file8 = "./losses/toy_toy_E8"
with open(losses_file8, "rb") as f8:
    losses8, accs8 = pickle.load(f8)
    

#plt.plot(losses3, color="blue", label="losses E3")
#plt.plot(losses6, color="orange", label="losses E6")

#plt.plot(losses4, color="navy", label="losses E4")
#plt.plot(losses7, color="darkorange", label="losses E7")

#plt.plot(losses5, color="purple", label="losses E5")
#plt.plot(losses8, color="peru", label="losses E8")

#print("mean losses1 losses 2", sum(losses1)/len(losses1), sum(losses2)/len(losses2))
#print("mean losses3 losses 6", sum(losses3)/len(losses3), sum(losses6)/len(losses6))
#print("mean losses4 losses 7", sum(losses4)/len(losses4), sum(losses7)/len(losses7))
#print("mean losses5 losses 8", sum(losses5)/len(losses5), sum(losses8)/len(losses8))

#plt.plot(accs3, color="blue", label="accs E3")
#plt.plot(accs6, color="orange", label="accs E6")

#plt.plot(accs4, color="navy", label="accs E4")
#plt.plot(accs7, color="darkorange", label="accs E7")

#plt.plot(accs5, color="purple", label="accs E5")
#plt.plot(accs8, color="peru", label="accs E8")
    
    
#losses_file13 = "./losses/toy_toy_E13"
#with open(losses_file13, "rb") as f13:
#    losses13, accs13 = pickle.load(f13)
    
#losses_file14 = "./losses/toy_toy_E14"
#with open(losses_file14, "rb") as f14:
#    losses14, accs14 = pickle.load(f14)
    
    
#losses_file15 = "./losses/toy_toy_E15"
#with open(losses_file15, "rb") as f15:
#    losses15, accs15 = pickle.load(f15)
    

#losses_file26 = "./losses/toy_toy_E26"
#with open(losses_file26, "rb") as f26:
#    losses26, accs26 = pickle.load(f26)
    
    
#losses_file27 = "./losses/toy_toy_E27"
#with open(losses_file27, "rb") as f27:
#    losses27, accs27 = pickle.load(f27)


#losses_file28 = "./losses/toy_toy_E28"
#with open(losses_file28, "rb") as f28:
#    losses28, accs28 = pickle.load(f28)


#plt.plot(losses13, color="blue", label="losses E13")
#plt.plot(losses26, color="orange", label="losses E26")

#plt.plot(losses14, color="navy", label="losses E14")
#plt.plot(losses27, color="darkorange", label="losses E27")

#plt.plot(losses15, color="purple", label="losses E15")
#plt.plot(losses28, color="peru", label="losses E28")


#print("mean losses13 losses 26", sum(losses13)/len(losses13), sum(losses26)/len(losses26))
#print("mean losses14 losses 27", sum(losses14)/len(losses14), sum(losses27)/len(losses27))
#print("mean losses15 losses 28", sum(losses15)/len(losses15), sum(losses28)/len(losses28))

#plt.plot(accs13, color="blue", label="accs E13")
#plt.plot(accs26, color="orange", label="accs E26")

#plt.plot(accs14, color="navy", label="accs E14")
#plt.plot(accs27, color="darkorange", label="accs E27")

#plt.plot(accs15, color="purple", label="accs E15")
#plt.plot(accs28, color="peru", label="accs E28")


losses_file31 = "./losses/toy_toy_E31"
with open(losses_file31, "rb") as f31:
    losses31, accs31 = pickle.load(f31)
    
losses_file41 = "./losses/toy_toy_E41"
with open(losses_file41, "rb") as f41:
    losses41, accs41 = pickle.load(f41)
    
    
losses_file51 = "./losses/toy_toy_E51"
with open(losses_file51, "rb") as f51:
    losses51, accs51 = pickle.load(f51)
    

losses_file62 = "./losses/toy_toy_E62"
with open(losses_file62, "rb") as f62:
    losses62, accs62 = pickle.load(f62)
    
    
losses_file72 = "./losses/toy_toy_E72"
with open(losses_file72, "rb") as f72:
    losses72, accs72 = pickle.load(f72)


losses_file82 = "./losses/toy_toy_E82"
with open(losses_file82, "rb") as f82:
    losses82, accs82 = pickle.load(f82)
    
    
#print("mean losses31 losses 62", sum(losses31)/len(losses31), sum(losses62)/len(losses62))
#print("mean losses41 losses 72", sum(losses41)/len(losses41), sum(losses72)/len(losses72))
#print("mean losses51 losses 82", sum(losses51)/len(losses51), sum(losses82)/len(losses82))


#plt.plot(losses31, color="blue", label="lossess E31")
#plt.plot(losses62, color="orange", label="lossess E62")

#plt.plot(losses41, color="navy", label="lossess E41")
#plt.plot(losses72, color="darkorange", label="lossess E72")

plt.plot(losses51, color="purple", label="lossess E51")
plt.plot(losses82, color="peru", label="lossess E82")



plt.legend()
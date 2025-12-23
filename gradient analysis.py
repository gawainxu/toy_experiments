import matplotlib.pyplot as plt
import numpy as np

s_ip = np.arange(0.8, 1.0, 0.001)
s_in = np.arange(0, 0.8, 0.001)
tau = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]

fig, axes = plt.subplots(
    1, 2, figsize=(17, 5))

ax1, ax2 = axes

for t in tau:

    A = np.sum(np.exp(s_ip/t)) + np.sum(np.exp(s_in/t))
    P = np.exp(s_ip/t)
    P = P / A
    g = -1/t + 1/t*P
    print(t, g)
    ax1.plot(s_ip, g, label=str(t), linewidth=2.5)

ax1.set_xlabel(r'$s_{ip}$', fontsize=20)
ax1.set_ylabel(r'$\frac{\partial{L_{SupCon}}}{s_{ip}}$', fontsize=20)
ax1.tick_params(axis='both', labelsize=20)

for t in tau:

    A = np.sum(np.exp(s_ip/t)) + np.sum(np.exp(s_in/t))
    N = np.exp(s_in/t)
    N = N / A
    g = 1/t*N
    print(t, g)
    ax2.plot(s_in, g, label=str(t), linewidth=2.5)

ax2.set_xlabel(r'$s_{in}$', fontsize=20)
ax2.set_ylabel(r'$\frac{\partial{L_{SupCon}}}{s_{in}}$', fontsize=20)
ax2.tick_params(axis='both', labelsize=20)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc = 'upper center', fontsize=18, ncol=len(tau))

fig.subplots_adjust(
    top=0.86,   # space for legend
    wspace=0.30 # space between subplots (fixes y-label overlap)
)

plt.savefig("./plots/sip.pdf", bbox_inches="tight")
#plt.show()



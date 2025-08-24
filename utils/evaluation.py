import matplotlib.pyplot as plt
import os

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_curves(all_curves, title, savepath=None):
    plt.figure()
    for label, rewards in all_curves.items():
        plt.plot(rewards, label=label)
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    if savepath:
        ensure_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=150)
    plt.show()

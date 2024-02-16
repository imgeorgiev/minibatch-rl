import torch
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.func import vmap, grad
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sns.set()
torch.manual_seed(0)


def f(x, v, th, a, t):
    return x + v * torch.cos(th) * t + 1 / 2 * a * t**2


# g, h, w = -9.81, 0.0, 10


# def f(x, v, th, a, t):
#     ty = (-v * torch.cos(th) + (v**2 * torch.cos(th) ** 2 + a * w) ** 0.5) / a
#     y = v * torch.sin(th) * ty + g / 2 * ty**2
#     out = x + v * torch.cos(th) * t + 1 / 2 * a * t**2
#     out = torch.where((h > y) & (ty < t), w, out)
#     return out


fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax = ax.flatten()

# simulation variables
xx = torch.linspace(-torch.pi, torch.pi, 100)
x, v, a, t = 0, 10, 1, 2
std = 0.1  # noise for policy
N = 5000  # data samples
iters = 300  # for optimization
lr = 5e-2

print("Plotting the ball example")
ax[0].scatter(x, 0, label="start", color="tab:red", s=1000)
ax[0].annotate(
    "", xy=(1, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:blue")
)
ax[0].set_xlim((-1, 10))
ax[0].set_ylim((-1, 10))

# if True:
#     ax[0].vlines(w, -6.0, 6 + h, color="black", lw=10)
# ax[0].axis("equal")


print("Plotting the problem landscape")
ax[1].plot(xx, -f(x, v, xx, a, t), label=r"$\sigma=0.0$")
# ax[2].plot(xx, -f(x, v, xx, a, t), label=r"$J(\theta)$")
ax[1].set_xlabel(r"$\theta$")
ax[2].set_xlabel(r"$\theta$")

# now also plot smoothed
for std in [0.1, 0.5, 1.0]:
    yy = torch.zeros_like(xx)
    for i in range(len(xx)):
        yy[i] = -f(x, v, xx[i] + torch.normal(0, std, (N,)), a, t).mean()
    ax[1].plot(xx, yy, label=f"$\\sigma={std:.1f}$")

std = 0.5
# now also plot smoothed
for N in [1, 10, 100, 1000]:
    yy = torch.zeros_like(xx)
    for i in range(len(xx)):
        yy[i] = -f(x, v, xx[i] + torch.normal(0, std, (N,)), a, t).mean()
    ax[2].plot(xx, yy, label=f"$N={N}$")

ax[1].legend()
ax[2].legend()
plt.tight_layout()
plt.savefig("ball_noise.pdf", bbox_inches="tight")

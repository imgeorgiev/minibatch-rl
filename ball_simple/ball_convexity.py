import torch
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.func import vmap, grad
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sns.set()
torch.manual_seed(0)


def f(x, v, th, a, t, g, h, w):
    ty = (-v * torch.cos(th) + (v**2 * torch.cos(th) ** 2 + a * w) ** 0.5) / a
    y = v * torch.sin(th) * ty + g / 2 * ty**2
    out = x + v * torch.cos(th) * t + 1 / 2 * a * t**2
    out = torch.where((h > y) & (ty < t), w, out)
    return out


fig, ax = plt.subplots(1, 3, figsize=(10, 2.5))
ax = ax.flatten()

# simulation variables
xx = torch.linspace(-torch.pi, torch.pi, 100)
x, v, a, t, g, h, w = 0, 10, 1, 2, -9.81, -4.0, 10
std = 0.5  # noise for policy
N = 5000  # data samples
iters = 100  # for optimization
lr = 5e-2
th_init = -2.0

det_color = "tab:brown"
sto_color = "tab:green"


print("Plotting the ball example")
# ax[0].scatter(x, 0, label="start", color="tab:red", s=1000)
circle = plt.Circle((0, 0), 0.5, color="tab:red")
ax[0].add_patch(circle)
ax[0].annotate(
    "", xy=(1, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:blue")
)
ax[0].set_xlim((-1, 10))
ax[0].set_ylim((-1, 10))
ax[0].vlines(w, -6.0, 6 + h, color="black", lw=10)
ax[0].axis("equal")

print("Plotting the problem landscape")
ax[1].plot(xx, -f(x, v, xx, a, t, g, h, w), label=r"$R_H$")
ax[1].set_xlabel(r"$\theta$")

# now also plot smoothed
yy = torch.zeros_like(xx)
for i in range(len(xx)):
    yy[i] = -f(x, v, xx[i] + torch.normal(0, std, (N,)), a, t, g, h, w).mean()
ax[1].plot(xx, yy, label=r"$\mathbb{E}[R_H]$")

print("Deterministic optimizaiton")
mu = Variable(torch.Tensor([th_init]), requires_grad=True)
optimizer = torch.optim.Adam([mu], lr=lr)

dt_thetas = []
dt_losses = []

for i in tqdm(range(iters)):
    act = mu
    loss = -f(x, v, act, a, t, g, h, w).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # for logging
    dt_thetas.append(mu.item())
    dt_losses.append(loss.mean().item())

ax[1].plot(
    dt_thetas[:: iters // 5],
    dt_losses[:: iters // 5],
    "-*",
    # label="Determ.",
    color=det_color,
)
ax[2].plot(dt_losses, color=det_color, label="Deterministic")


print("First-order gradients optimization")
mu = Variable(torch.Tensor([th_init]), requires_grad=True)
pi = Normal(mu, std)
optimizer = torch.optim.Adam([mu], lr=lr)

fo_thetas = []
fo_losses = []
loss_fn = lambda act: -f(x, v, act, a, t, g, h, w).mean()

for i in tqdm(range(iters)):
    act = pi.rsample((N, 1))
    loss = -f(x, v, act, a, t, g, h, w).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # for logging
    fo_thetas.append(mu.item())
    fo_losses.append(loss.mean().item())

ax[1].plot(
    fo_thetas[:: iters // 5],
    fo_losses[:: iters // 5],
    "-*",
    # label="Stochastic",
    color=sto_color,
)
ax[2].plot(fo_losses, color=sto_color, label=f"Stochastic")


ax[1].legend()
ax[2].set_xlabel("Iterations")
ax[2].set_ylabel("Loss")
ax[2].legend()
plt.tight_layout()
plt.savefig("ball_stochastic.pdf", bbox_inches="tight")

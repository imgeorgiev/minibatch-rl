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


fig, ax = plt.subplots(2, 2, figsize=(10, 6))
ax = ax.flatten()

# simulation variables
xx = torch.linspace(-torch.pi, torch.pi, 100)
x, v, a, t = 0, 10, 1, 2
std = 0.1  # noise for policy
N = 100  # data samples
iters = 300  # for optimization
lr = 5e-2


print("Plotting the ball example")
ax[0].scatter(x, 0, label="start", color="tab:red", s=1000)
ax[0].annotate(
    "", xy=(1, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:blue")
)
ax[0].set_xlim((-1, 10))
ax[0].set_ylim((-1, 10))

print("Plotting the problem landscape")
ax[1].plot(xx, -f(x, v, xx, a, t), label=r"$J(\theta)$")
ax[1].set_xlabel(r"$\theta$")

# now also plot smoothed
yy = torch.zeros_like(xx)
for i in range(len(xx)):
    yy[i] = -f(x, v, xx[i] + torch.normal(0, std, (N,)), a, t).mean()
ax[1].plot(xx, yy, label=r"$E[J(\theta)]$")


print("Policy gradients optimization")
mu = Variable(torch.Tensor([3 / 4 * torch.pi]), requires_grad=True)
pi = Normal(mu, std)
optimizer = torch.optim.Adam([mu], lr=lr)

pg_thetas = []
pg_losses = []
pg_var = []


# wrapper functions for vmap to make per-sample gradient compute efficient
loss_fn = lambda act, logp: torch.mean(-f(x, v, act, a, t) * logp)
ft_compute_grad = grad(loss_fn)
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=0)


for i in tqdm(range(iters)):
    act = pi.sample((N, 1))
    y = -f(x, v, act, a, t)
    logps = pi.log_prob(act)
    loss = loss_fn(act, logps)
    ft_per_sample_grads = ft_compute_sample_grad(act, logps)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # for logging
    pg_thetas.append(mu.item())
    pg_losses.append(y.mean().item())
    pg_var.append(torch.var(ft_per_sample_grads).item())

ax[1].plot(pg_thetas[:: iters // 10], pg_losses[:: iters // 10], "-*", label="PG")
ax[2].plot(pg_losses, label=f"PG={pg_losses[-1]:.3f}")
ax[3].plot(pg_var, label="PG")


print("First-order gradients optimization")
mu = Variable(torch.Tensor([3 / 4 * torch.pi]), requires_grad=True)
pi = Normal(mu, std)
optimizer = torch.optim.Adam([mu], lr=lr)

fo_thetas = []
fo_losses = []
fo_var = []


# wrapper functions for vmap to make per-sample gradient compute efficient
loss_fn = lambda act: -f(x, v, act, a, t).mean()
ft_compute_grad = grad(loss_fn)
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=0)

for i in tqdm(range(iters)):
    act = pi.rsample((N, 1))
    loss = -f(x, v, act, a, t).mean()
    ft_per_sample_grads = ft_compute_sample_grad(act)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # for logging
    fo_thetas.append(mu.item())
    fo_losses.append(loss.mean().item())
    fo_var.append(torch.var(ft_per_sample_grads).item())

ax[1].plot(fo_thetas[:: iters // 10], fo_losses[:: iters // 10], "-*", label="FO")
ax[2].plot(fo_losses, label=f"FO={fo_losses[-1]:.3f}")
ax[3].plot(fo_var, label="FO")


ax[1].legend()
ax[2].set_xlabel("Iterations")
ax[2].set_ylabel("Loss")
ax[2].legend()
ax[3].legend()
ax[3].set_yscale("log")
ax[3].set_xlabel("Iterations")
ax[3].set_ylabel("Variance")
plt.tight_layout()
plt.show()

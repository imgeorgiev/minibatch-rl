import torch
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.func import vmap, grad
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import scipy.integrate as integrate

sns.set()
torch.manual_seed(0)


g, h, w = -9.81, 0.0, 10


def f(x, v, th, a, t):
    ty = (-v * torch.cos(th) + (v**2 * torch.cos(th) ** 2 + a * w) ** 0.5) / a
    y = v * torch.sin(th) * ty + g / 2 * ty**2
    out = x + v * torch.cos(th) * t + 1 / 2 * a * t**2
    out = torch.where((h > y) & (ty < t), w, out)
    return out


def f_grad(x, v, th, a, t, std):
    vals = np.empty_like(th)
    for i in range(len(th)):
        vals[i] = (
            v * t / 2**0.5 / 4 * np.exp(-(std**2) / 4) * np.sin(th[i])
            - integrate.quad(
                lambda w: np.exp(-(w**2) / std**2) * np.cos(th[i] + w), -1.418, 0.248
            )[0]
        )
        # print(v)
        # vals[i] = v
    return vals


fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
ax = ax.flatten()

# simulation variables
xx = torch.linspace(-torch.pi, torch.pi, 100)
ns = torch.arange(1, 1000, 50)
x, v, a, t = 0, 10, 1, 2
std = 0.1  # noise for policy
N = 1000  # data samples
iters = 300  # for optimization
lr = 5e-2
stds = torch.linspace(0.01, 1.0, 50)
# stds = torch.Tensor([0.01])


# first pre-compute the true gradient
print("Pre-computing true gradients")
n = 100000
# wrapper functions for vmap to make per-sample gradient compute efficient
loss_fn = lambda act, logp: torch.mean(-f(x, v, act, a, t) * logp)


for i, n in enumerate([1, 10, 50, 1000]):

    dual_axis = ax[i].twinx()

    print(f"Policy gradients optimization; N={n}")

    pg_bias = []
    pg_var = []

    # wrapper functions for vmap to make per-sample gradient compute efficient
    loss_fn = lambda act, logp: torch.mean(-f(x, v, act, a, t) * logp)
    ft_compute_grad = grad(loss_fn)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=0)

    for std in tqdm(stds):
        grad_est = []
        var_per_x = []
        true_grad = f_grad(x, v, xx.numpy(), a, t, std.numpy())
        for j, a in enumerate(xx):
            mu = Variable(torch.Tensor([a]), requires_grad=True)
            pi = Normal(mu, std)
            act = pi.sample((n, 1))
            baseline = -f(x, v, pi.mean, a, t)
            y = -f(x, v, act, a, t)
            y = y - baseline
            logps = -pi.log_prob(act)
            ft_per_sample_grads = ft_compute_sample_grad(act, logps)
            var_per_x.append(torch.var(ft_per_sample_grads).item())
            grad_est.append(ft_per_sample_grads.mean())

        bias = torch.mean((torch.Tensor(grad_est) - true_grad) ** 2).item()
        if bias < 0:
            print("Negative bias")
        pg_bias.append(bias)
        pg_var.append(torch.mean(torch.Tensor(var_per_x)).item())

    ax[i].plot(stds, pg_bias, label="PG", color="tab:blue")
    dual_axis.plot(
        stds,
        pg_var,
        "--",
        color="tab:blue",
    )

    print(f"First-order gradients optimization; N={n}")

    fo_bias = []
    fo_var = []

    # wrapper functions for vmap to make per-sample gradient compute efficient
    loss_fn = lambda act: -f(x, v, act, a, t).mean()
    ft_compute_grad = grad(loss_fn)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=0)

    for std in tqdm(stds):
        grad_est = []
        var_per_x = []
        true_grad = f_grad(x, v, xx, a, t, std)
        for j, a in enumerate(xx):
            mu = Variable(torch.Tensor([a]), requires_grad=True)
            pi = Normal(mu, std)
            act = pi.rsample((n, 1))
            ft_per_sample_grads = ft_compute_sample_grad(act)
            grad_est.append(ft_per_sample_grads.mean())
            var_per_x.append(torch.var(ft_per_sample_grads).item())

        bias = torch.mean((torch.Tensor(grad_est) - true_grad) ** 2).item()
        if bias < 0:
            print("Negative bias")
        fo_bias.append(bias)
        fo_var.append(torch.mean(torch.Tensor(var_per_x)).item())

    ax[i].plot(stds, fo_bias, label="FO", color="tab:orange")
    dual_axis.plot(stds, fo_var, "--", color="tab:orange")
    ax[i].legend()
    ax[i].set_xlabel(r"$\sigma$")
    ax[i].set_ylabel("Bias")
    dual_axis.set_ylabel("Variance (dashed)")
    ax[i].set_yscale("log")
    if i != 0:
        dual_axis.set_yscale("log")

    if i != 3:
        dual_axis.set_yticks([])
    ax[i].set_xscale("log")
    ax[i].set_title(f"N={n}")

# ax[1].legend()
# ax[2].set_xlabel("Iterations")
# ax[2].set_ylabel("Loss")
# ax[2].legend()
# ax[3].legend()
# ax[3].set_yscale("log")
# ax[3].set_xlabel("Iterations")
# ax[3].set_ylabel("Variance")
plt.tight_layout()
plt.savefig("ball_wall_bias_variance.pdf", bbox_inches="tight")
plt.show()

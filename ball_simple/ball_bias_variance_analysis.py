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


fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax = ax.flatten()

# simulation variables
xx = torch.linspace(-torch.pi, torch.pi, 100)
ns = torch.arange(1, 1000, 50)
x, v, a, t = 0, 10, 1, 2
std = 0.1  # noise for policy
N = 1000  # data samples
iters = 300  # for optimization
lr = 5e-2

for i, std in enumerate([0.1, 0.5, 1.0]):

    dual_axis = ax[i].twinx()

    print(f"Policy gradients optimization; std={std:.2f}")
    mu = Variable(torch.Tensor([3 / 4 * torch.pi]), requires_grad=True)
    pi = Normal(mu, std)

    pg_bias = []
    pg_var = []

    # wrapper functions for vmap to make per-sample gradient compute efficient
    loss_fn = lambda act, logp: torch.mean(-f(x, v, act, a, t) * logp)
    ft_compute_grad = grad(loss_fn)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=0)

    for n in tqdm(ns):
        bias_per_x = []
        var_per_x = []
        for a in xx:
            mu = Variable(torch.Tensor([a]), requires_grad=True)
            pi = Normal(mu, std)
            act = pi.sample((n, 1))
            y = -f(x, v, act, a, t)
            logps = pi.log_prob(act)
            ft_per_sample_grads = ft_compute_sample_grad(act, logps)
            grad_est = ft_per_sample_grads.mean()
            (-f(x, v, mu, a, t)).mean().backward()
            var_per_x.append(torch.var(ft_per_sample_grads).item())
            bias_per_x.append(torch.norm(grad_est - mu.grad).item())
            mu.grad.zero_()
        pg_bias.append(torch.mean(torch.Tensor(bias_per_x)).item())
        pg_var.append(torch.mean(torch.Tensor(var_per_x)).item())

    ax[i].plot(ns, pg_bias, label="PG")
    dual_axis.plot(ns, pg_var, color="tab:blue", alpha=0.6)

    print(f"First-order gradients optimization; std={std:.2f}")
    mu = Variable(torch.Tensor([3 / 4 * torch.pi]), requires_grad=True)
    pi = Normal(mu, std)

    fo_bias = []
    fo_var = []

    # wrapper functions for vmap to make per-sample gradient compute efficient
    loss_fn = lambda act: -f(x, v, act, a, t).mean()
    ft_compute_grad = grad(loss_fn)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=0)

    for n in tqdm(ns):
        bias_per_x = []
        var_per_x = []
        for a in xx:
            mu = Variable(torch.Tensor([a]), requires_grad=True)
            pi = Normal(mu, std)
            act = pi.rsample((n, 1))
            # loss = -f(x, v, act, a, t).mean()
            ft_per_sample_grads = ft_compute_sample_grad(act)
            grad_est = ft_per_sample_grads.mean()

            (-f(x, v, mu, a, t)).mean().backward()

            # for logging
            bias_per_x.append(torch.norm(grad_est - mu.grad).item())
            var_per_x.append(torch.var(ft_per_sample_grads).item())
            mu.grad.zero_()
        fo_bias.append(torch.mean(torch.Tensor(bias_per_x)).item())
        fo_var.append(torch.mean(torch.Tensor(var_per_x)).item())

    ax[i].plot(ns, fo_bias, label="FO")
    dual_axis.plot(ns, fo_var, color="tab:orange", alpha=0.6)
    ax[i].legend()
    ax[i].set_xlabel("N")
    ax[i].set_ylabel("Bias")
    dual_axis.set_ylabel("Variance")
    dual_axis.set_yscale("log")
    # ax[i].set_xscale("log")
    ax[i].set_title(f"std={std:.2f}")

# ax[1].legend()
# ax[2].set_xlabel("Iterations")
# ax[2].set_ylabel("Loss")
# ax[2].legend()
# ax[3].legend()
# ax[3].set_yscale("log")
# ax[3].set_xlabel("Iterations")
# ax[3].set_ylabel("Variance")
plt.tight_layout()
plt.show()

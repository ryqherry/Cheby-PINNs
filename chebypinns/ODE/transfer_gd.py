import torch
import torch.nn as nn
import torch.nn.functional as F

from neurodiffeq import diff
from neurodiffeq.generators import Generator1D


class LinearHead(nn.Module):
    def __init__(self, d=256, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d, dtype=dtype, device=device))
        self.b = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))

    def forward(self, latent_512):
        out1 = latent_512[:, :256]
        out2 = latent_512[:, 256:]
        x = out1 @ self.w + self.b
        y = out2 @ self.w + self.b
        return torch.stack([x, y], dim=1)

def _sample_t(domain, sample_size, eps=1e-3, dtype=torch.float32, device="cpu"):
    t = (domain[1] - domain[0]) * Generator1D(size=sample_size, method="equally-spaced-noisy").get_examples() + domain[0]
    t[t < domain[0]] = domain[0] + eps
    t[t > domain[1]] = domain[1] - eps
    t = t.to(dtype=dtype, device=device).view(-1, 1)
    t.requires_grad_(True)
    return t

def _freeze_trunk(model):
    # Freeze everything in the pretrained network
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

@torch.no_grad()
def init_head_from_W(head: LinearHead, W):
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().view(-1)
    else:
        import numpy as np
        W = torch.tensor(np.array(W).reshape(-1), dtype=head.w.dtype)

    if W.numel() != 257:
        raise ValueError(f"Expected W to have 257 params (256 weights + bias). Got {W.numel()}")

    head.w.copy_(W[:256].to(head.w.device, head.w.dtype))
    head.b.copy_(W[256:257].to(head.b.device, head.b.dtype))

def fit_head_gd_duffing(model, forcing, alpha, delta, beta, u0, v0, domain=(0.0, 5.0), sample_size=256,
    max_iter=4000, lr=1e-3, ode_weight=1.0, ic_weight=1.0, eps_t=1e-6, early_stop_ode_mse=None, step_size=100, 
    gamma=0.92, dtype=torch.float32, device="cpu", W_init=None, verbose_every=200):
    _freeze_trunk(model)

    head = LinearHead(d=256, dtype=dtype, device=device)
    if W_init is not None:
        init_head_from_W(head, W_init)

    opt = torch.optim.Adam(head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    loss_trace = []
    ode_trace = []
    ic_trace = []

    alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
    delta_t = torch.tensor(delta, dtype=dtype, device=device)
    beta_t = torch.tensor(beta, dtype=dtype, device=device)

    reach_max = False

    for it in range(1, max_iter + 1):
        opt.zero_grad()

        t = _sample_t(domain, sample_size, eps=eps_t, dtype=dtype, device=device)
        _, latent = model(t)

        pred = head(latent)
        x = pred[:, 0:1]
        y = pred[:, 1:2]

        dxdt = diff(x, t)
        dydt = diff(y, t)

        # nonlinearity term
        gx = torch.cos(x)
        f_val = forcing(t).to(dtype=dtype, device=device).view(-1, 1)

        r1 = dxdt - y
        r2 = dydt + delta_t * y + alpha_t * x + beta_t * gx - f_val

        ode_loss = (r1.pow(2).mean() + r2.pow(2).mean())
        mse_loss = r2.pow(2).mean()

        # IC loss at t0
        t0 = torch.tensor([[domain[0]]], dtype=dtype, device=device, requires_grad=False)
        _, latent0 = model(t0)
        pred0 = head(latent0).view(-1)
        ic_target = torch.tensor([u0, v0], dtype=dtype, device=device)
        ic_loss = F.mse_loss(pred0, ic_target)

        total_loss = ode_weight * ode_loss + ic_weight * ic_loss
        total_loss.backward()
        opt.step()
        scheduler.step()

        loss_trace.append(total_loss.item())
        ode_trace.append(ode_loss.item())
        ic_trace.append(ic_loss.item())

        if mse_loss.item() <= early_stop_ode_mse:
            if verbose_every is not None:
                print(f"Early stop at iter {it} (ode_mse={mse_loss.item()} <= {early_stop_ode_mse})")
            break

        if verbose_every is not None and it % verbose_every == 0:
            print(f"Gradient Descent iter {it:5d}: total={total_loss.item()} "
                  f"ode={ode_loss.item()} ic={ic_loss.item()}")
        
        if it == max_iter:
            reach_max = True

    traces = {"total": loss_trace, "ode": ode_trace, "ic": ic_trace, "reach_max": reach_max}
    return head, traces

def fit_head_gd_duffing_inv_sq(model, forcing, alpha, delta, beta, u0, v0, domain=(0.0, 5.0), sample_size=256,
    max_iter=4000, lr=1e-3, ode_weight=1.0, ic_weight=1.0, eps_t=1e-6, early_stop_ode_mse=None, step_size=100, gamma=0.92, 
    dtype=torch.float32, device="cpu", W_init=None, verbose_every=200):
    _freeze_trunk(model)

    head = LinearHead(d=256, dtype=dtype, device=device)
    if W_init is not None:
        init_head_from_W(head, W_init)

    opt = torch.optim.Adam(head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    loss_trace = []
    ode_trace = []
    ic_trace = []

    alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
    delta_t = torch.tensor(delta, dtype=dtype, device=device)
    beta_t = torch.tensor(beta, dtype=dtype, device=device)

    reach_max = False

    for it in range(1, max_iter + 1):
        opt.zero_grad()

        t = _sample_t(domain, sample_size, eps=eps_t, dtype=dtype, device=device)
        _, latent = model(t)

        pred = head(latent)
        x = pred[:, 0:1]
        y = pred[:, 1:2]

        dxdt = diff(x, t)
        dydt = diff(y, t)

        # nonlinearity term
        gx = 1.0 / (x ** 2)
        f_val = forcing(t).to(dtype=dtype, device=device).view(-1, 1)

        r1 = dxdt - y
        r2 = dydt + delta_t * y + alpha_t * x + beta_t * gx - f_val

        ode_loss = (r1.pow(2).mean() + r2.pow(2).mean())
        mse_loss = r2.pow(2).mean()

        # IC loss at t0
        t0 = torch.tensor([[domain[0]]], dtype=dtype, device=device, requires_grad=False)
        _, latent0 = model(t0)
        pred0 = head(latent0).view(-1)
        ic_target = torch.tensor([u0, v0], dtype=dtype, device=device)
        ic_loss = F.mse_loss(pred0, ic_target)

        total_loss = ode_weight * ode_loss + ic_weight * ic_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()

        loss_trace.append(total_loss.item())
        ode_trace.append(ode_loss.item())
        ic_trace.append(ic_loss.item())

        if mse_loss.item() <= early_stop_ode_mse:
            if verbose_every is not None:
                print(f"Early stop at iter {it} (ode_mse={mse_loss.item()} <= {early_stop_ode_mse})")
            break

        if verbose_every is not None and it % verbose_every == 0:
            print(f"Gradient Descent iter {it:5d}: total={total_loss.item()} "
                  f"ode={ode_loss.item()} ic={ic_loss.item()}")
        
        if it == max_iter:
            reach_max = True

    traces = {"total": loss_trace, "ode": ode_trace, "ic": ic_trace, "reach_max": reach_max}
    return head, traces

@torch.no_grad()
def predict_with_head(model, head: LinearHead, t_grid, dtype=torch.float32, device="cpu"):
    _freeze_trunk(model)
    t = torch.tensor(t_grid, dtype=dtype, device=device).view(-1, 1)
    _, latent = model(t)
    pred = head(latent)
    return pred[:, 0].detach().cpu(), pred[:, 1].detach().cpu()
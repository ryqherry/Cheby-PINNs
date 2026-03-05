import torch
import torch.nn as nn
import torch.nn.functional as F
from neurodiffeq import diff

from helper import generate_interior_tensor, generate_boundary_tensor  # :contentReference[oaicite:1]{index=1}


class LinearHead(nn.Module):
    def __init__(self, d=256, device="cpu"):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d, device=device))
        self.b = nn.Parameter(torch.zeros(1, device=device))

    def forward(self, latent_512):
        out1 = latent_512[:, :256]
        u = out1 @ self.w + self.b
        return u.view(-1, 1)

    def as_W(self):
        return torch.cat([self.w.view(-1), self.b.view(-1)], dim=0)

def _freeze_trunk(model):
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

def fit_head_gd_reaction_diffusion(model, forcing, r, delta, b0, D=1.0, I=60, B=200, max_iter=20000, lr=1e-3, 
    pde_weight=1.0, bc_weight=1.0, early_stop_pde_mse=None, step_size=100, gamma=0.92, device="cpu", 
    W_init=None, verbose_every=200):

    _freeze_trunk(model)

    head = LinearHead(d=256, device=device)
    if W_init is not None:
        init_head_from_W(head, W_init)

    opt = torch.optim.Adam(head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    loss_trace, pde_trace, bc_trace = [], [], []

    r_t = torch.tensor(r, device=device)
    delta_t = torch.tensor(delta, device=device)
    D_t = torch.tensor(D, device=device)
    b0_t = torch.tensor(float(b0), device=device)
    reach_max = False

    for it in range(1, max_iter + 1):
        opt.zero_grad()

        x, t, interior_tensor = generate_interior_tensor(I=I, require_grad=True)
        xb, tb, boundary_tensor = generate_boundary_tensor(B=B, require_grad=True)

        # Interior prediction
        _, latent = model(interior_tensor)
        u = head(latent)

        ut = diff(u, t)
        uxx = diff(diff(u, x), x)

        f_val = forcing(x, t).to(device=device).view(-1, 1)

        # PDE residual
        residual = ut - D_t * uxx - r_t * (u / (u + 1.0) - delta_t * u) - f_val
        pde_loss = (residual.pow(2)).mean()

        # Boundary prediction and loss
        _, latent_b = model(boundary_tensor)
        u_b = head(latent_b)
        bc_loss = F.mse_loss(u_b, b0_t * torch.ones_like(u_b))

        total_loss = pde_weight * pde_loss + bc_weight * bc_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()

        loss_trace.append(total_loss.item())
        pde_trace.append(pde_loss.item())
        bc_trace.append(bc_loss.item())

        # Early stop on PDE MSE
        if early_stop_pde_mse is not None and pde_loss.item() <= early_stop_pde_mse:
            if verbose_every is not None:
                print(f"Early stop at iter {it} (pde_mse={pde_loss.item()} <= {early_stop_pde_mse})")
            break

        if verbose_every is not None and it % verbose_every == 0:
            print(f"Gradient Descent iter {it:5d}: total={total_loss.item()} "
                  f"pde={pde_loss.item()} bc={bc_loss.item()}")
        
        if it == max_iter:
            print(f"Reached max iterations ({max_iter}). Final PDE MSE: {pde_loss.item()}")
            reach_max = True

    traces = {"total": loss_trace, "pde": pde_trace, "bc": bc_trace, "W257": head.as_W().detach().cpu(), "reach_max": reach_max}
    return head, traces

@torch.no_grad()
def predict_u_with_head(model, head: LinearHead, x_grid, t_grid, device="cpu"):
    _freeze_trunk(model)
    xt = torch.tensor(
        torch.stack([torch.tensor(x_grid), torch.tensor(t_grid)], dim=1), device=device
    )
    _, latent = model(xt)
    u = head(latent)
    return u.detach().cpu()
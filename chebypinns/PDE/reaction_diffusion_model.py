import torch
import torch.nn as nn
import torch.nn.functional as F

from neurodiffeq import diff
from neurodiffeq.neurodiffeq import unsafe_diff as unsafe_diff
from neurodiffeq.generators import Generator1D, Generator2D

import numpy as np
from tqdm.auto import tqdm

class Multihead_model(nn.Module):
	def __init__(self, k, act = nn.Tanh(), bias=False):
		super().__init__()
		self.act = act
		self.linear1 = nn.Linear(2, 128)
		self.linear2 = nn.Linear(128, 128)
		self.linear3 = nn.Linear(128, 256)
		self.linear4 = nn.Linear(256, 256)
		self.linear5 = nn.Linear(256, 512)
		self.final_layers = nn.ModuleList(
				[nn.Linear(256, 1, bias=bias) for _ in range(k)]
		)
		self.k = k

	def forward(self, x):
		fc1 = self.act(self.linear1(x))
		out = self.act(self.linear2(fc1))
		out = self.act(self.linear3(out))
		out = self.act(self.linear4(out))
		out = self.act(self.linear5(out))
		out1 = out[:, :256]
		out2 = out[:, 256:]
		output = []
		for i in range(self.k):
			first = self.final_layers[i](out1)
			second = self.final_layers[i](out2)
			concat = torch.cat((first, second), axis=1)
			output.append(concat)
		return torch.stack(output), out

def train(model, optimizer, lossfn, num_iter, Forcing_functions, truth_functions, boundary_value, D = 1, interior_grid=(30, 30), 
		  x_boundary_num=100, t_boundary_num=100, every=100, pde_weight=1, bc_weight=1, data_weight=1, scheduler=None, method='chebyshev'):
	loss_trace = []; pde_loss_trace = []; bc_loss_trace = []; data_loss_trace = []
	for i in tqdm(range(num_iter)):
		optimizer.zero_grad()

		total, pde, bc, data = loss(model, interior_grid, x_boundary_num, t_boundary_num, boundary_value, Forcing_functions, 
							  truth_functions, D = D, pde_weight=pde_weight, bc_weight=bc_weight, data_weight=data_weight, method=method)
		
		total.backward()
		optimizer.step()
		if scheduler is not None:
			scheduler.step()

		loss_trace.append(total.item())
		pde_loss_trace.append(pde.item())
		bc_loss_trace.append(bc.item())
		data_loss_trace.append(data.item())

		if (i+1)%every == 0:
			print("{}th Iter: total {}, pde {}, bc {}, data {}".format(i+1, total.item(), pde.item(), bc.item(), data.item()))
		
	return loss_trace, pde_loss_trace, bc_loss_trace, data_loss_trace

def loss(model, interior_grid, x_boundary_num, t_boundary_num, boundary_value, Forcing_functions, 
		 truth_functions, D=1, pde_weight=1, bc_weight=1, data_weight=1, method='chebyshev'):
    generator= Generator2D(grid=interior_grid, method=method)
    samples = generator.get_examples()
    x = samples[0].unsqueeze(1)
    t = samples[1].unsqueeze(1)
    x[x<0] = 0; x[x>1] = 1
    x.requires_grad_()
    t.requires_grad_()
    input_tensor = torch.cat([x, t], dim=1)

    # Evaluate the PINN on these sample points
    output_tensor, _ = model(input_tensor)

    # Separate the u, y, z parts of the output of the network
    u = output_tensor[:,:,0].T
    y = output_tensor[:,:,1].T

    # Compute the gradients
    dudt = torch.cat([diff(u[:,i].reshape(-1, 1), t) for i in range(u.shape[1])], dim=1)
    dudx = torch.cat([diff(u[:,i].reshape(-1, 1), x) for i in range(u.shape[1])], dim=1)
    dydx = torch.cat([diff(y[:,i].reshape(-1, 1), x) for i in range(y.shape[1])], dim=1)
    
    # Compute the forcing function on N data points across k heads
    force = torch.cat([Forcing_functions[i](x, t) for i in range(len(Forcing_functions))], dim=1)

    # Compute the pde residual
    residual = torch.cat(
        [
            (dudt - D*dydx - force).unsqueeze(2),
            (dudx - y).unsqueeze(2),
        ],
        dim=2
    )
    # Compute the pde loss
    pde_loss = F.mse_loss(residual, torch.zeros_like(residual))

    ##forward pass for the BC condition
    ##sample points from the boundary first
    x_samples = Generator1D(size=x_boundary_num, method='chebyshev').get_examples()
    t_samples = Generator1D(size=t_boundary_num, method='chebyshev').get_examples()
    x_boundary = torch.cat([
        torch.zeros_like(t_samples),
        torch.ones_like(t_samples),
        x_samples,
    ]).unsqueeze(1)
    t_boundary = torch.cat([
        t_samples,
        t_samples,
        torch.zeros_like(x_samples),
    ]).unsqueeze(1)
    x_boundary.requires_grad_()
    t_boundary.requires_grad_()
    input_boundary = torch.cat([x_boundary, t_boundary], dim=1)
    # Evaluate the neural network at the boundary
    output_boundary, _ = model(input_boundary)
    u0_boundary = output_boundary[:,:,0]
    
    if isinstance(boundary_value, (int, float)):
      truth_boundary = torch.ones_like(u0_boundary)*boundary_value
    elif isinstance(boundary_value[0], (int, float)):
      truth_boundary = torch.ones_like(u0_boundary) * torch.tensor(np.array(boundary_value)[:, np.newaxis])
      truth_boundary.to(u0_boundary.device)
    # if the boundary_value is a list of functions
    else: 
      truth_boundary = torch.stack([truth(input_boundary[:,0], input_boundary[:, 1]) for truth in boundary_value])
    bc_loss = F.mse_loss(u0_boundary, truth_boundary)

    # Compute the data loss
    truth = torch.cat([truth_functions[i](x, t) for i in range(len(truth_functions))], dim=1)
    data_loss = F.mse_loss(u, truth)

    # Sum the weighted loss
    total_loss = pde_weight*pde_loss + bc_weight*bc_loss + data_weight*data_loss
    return total_loss, pde_loss, bc_loss, data_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurodiffeq import diff
from neurodiffeq.neurodiffeq import unsafe_diff as unsafe_diff
from neurodiffeq.generators import Generator1D

from tqdm.auto import tqdm

class Multihead(nn.Module):
	def __init__(self, k, act = nn.Tanh(), dtype=torch.float32):
		super().__init__()
		self.act = act
		self.linear1 = nn.Linear(1, 128, dtype=dtype)
		self.linear2 = nn.Linear(128, 128, dtype=dtype)
		self.linear3 = nn.Linear(128, 256, dtype=dtype)
		self.linear4 = nn.Linear(256, 512, dtype=dtype)

		self.final_layers = nn.ModuleList(
				[nn.Linear(256, 1, bias=True, dtype=dtype) for _ in range(k)]
		)
		self.k = k

	def forward(self, x):
		out = self.act(self.linear1(x))
		out = self.act(self.linear2(out))
		out = self.act(self.linear3(out))
		out = self.act(self.linear4(out))
		out1 = out[:, :256]
		out2 = out[:, 256:]
		output = []
		for i in range(self.k):
			first = self.final_layers[i](out1)
			second = self.final_layers[i](out2)
			concat = torch.cat((first, second), axis=1)
			output.append(concat)
		return torch.stack(output), out

def train(model, optimizer, lossfn, num_iter, para_dict, sample_size=100, domain=(0, 5), every=100, 
		  ode_weight=1, ic_weight=1, scheduler=None, epsilon=1e-3, dtype=torch.float32, verbose=True):
	loss_trace = []; ode_loss_trace = []; ic_loss_trace = []
	Forcing_functions = para_dict['Forcing_functions']
	initial_values = para_dict['initial_values']
	initial_velocities = para_dict['initial_velocities']
	delta_list = para_dict['delta_list']
	alpha_list = para_dict['alpha_list']

	for i in tqdm(range(num_iter)):
		optimizer.zero_grad()
		loss_dict = lossfn(model, Forcing_functions, delta_list, alpha_list, initial_values, initial_velocities, domain=domain, 
					 sample_size=sample_size, epsilon=epsilon, alpha_ode = ode_weight, alpha_ic = ic_weight, dtype=dtype)

		loss_dict['total_loss'].backward()
		optimizer.step()
		if scheduler is not None:
			scheduler.step()
		
		loss_trace.append(loss_dict['total_loss'].item())
		ode_loss_trace.append(loss_dict['ode_loss'].item())
		ic_loss_trace.append(loss_dict['ic_loss'].item())

		if verbose and (i+1)%every == 0:
			print("{}th Iter: total {}, ode {}, ic {}".format(i+1, loss_dict['total_loss'].item(), loss_dict['ode_loss'].item(), loss_dict['ic_loss'].item()))
	return loss_trace, ode_loss_trace, ic_loss_trace

def loss(model, forcing, delta_list, alpha_list, initial_values, initial_velocities, domain=(0, 5), 
		 sample_size=100, epsilon=1e-3, dtype=torch.float32, alpha_ode = 1, alpha_ic = 1, device='cpu'):
  	# Generate samples of time points
	t_samples = (domain[1]-domain[0])*Generator1D(size=sample_size, method='equally-spaced-noisy').get_examples() + domain[0]
	t_samples[t_samples < domain[0]] = domain[0] + epsilon
	t_samples[t_samples > domain[1]] = domain[1] - epsilon

	t_samples = t_samples.to(dtype)
	t_samples = t_samples.view(-1, 1).to(device)
	t_samples = t_samples.requires_grad_()

	# Evaluate the model at sample points
	output, _ = model(t_samples)
	x = output[:,:,0]
	y = output[:,:,1]

	# Compute the derivatives
	dxdt = torch.cat([diff(x[i].view(-1, 1), t_samples) for i in range(x.shape[0])], dim=1)
	dydt = torch.cat([diff(y[i].view(-1, 1), t_samples) for i in range(y.shape[0])], dim=1)

	# Compute forcing at sample points
	force = torch.cat([forcing[i](t_samples) for i in range(len(forcing))], dim=1)
	force = force.to(dtype)

	# Compute the ODE residual loss
	alpha_list = torch.tensor(alpha_list, dtype=dtype, device=device)
	delta_list = torch.tensor(delta_list, dtype=dtype, device=device)
	alpha_x = (alpha_list.view(-1, 1)*x).T
	delta_dxdt = delta_list * dxdt
	delta_y = (delta_list.view(-1, 1)*y).T
	residual = torch.cat(
			[
				(dxdt - y.T).unsqueeze(2),
				(dydt + delta_y + alpha_x - force).unsqueeze(2),
			],
			dim=2
		)
	ode_loss = F.mse_loss(residual, torch.zeros_like(residual))

	# Compute the initial condition loss
	output0, _ = model(torch.tensor([[domain[0]]], dtype=dtype, device=device))
	u0 = torch.stack([torch.tensor(initial_values, dtype=dtype, device=device),
			torch.tensor(initial_velocities, dtype=dtype, device=device)], dim=1).view(-1, 1, 2)
	ic_loss = F.mse_loss(u0, output0)

	# Compute the total loss
	total_loss = alpha_ode * ode_loss + alpha_ic * ic_loss

	return {'total_loss': total_loss, 'ode_loss': ode_loss, 'ic_loss': ic_loss}
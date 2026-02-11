import torch
import torch.nn as nn
import torch.nn.functional as F

from neurodiffeq import diff
from neurodiffeq.neurodiffeq import unsafe_diff as unsafe_diff
from neurodiffeq.generators import Generator1D, Generator2D

import numpy as np
from matplotlib import pyplot as plt


# Define the grids where the solution will be evaluated
N = 60
x_grid = np.arange(0, 1 + .5/N, 1/N)
t_grid = np.arange(0, 1 + .5/N, 1/N)
X_grid = []
for t in t_grid:
	for x in x_grid:
		X_grid.append([x, t])

X_grid = torch.Tensor(np.array(X_grid))

def forcing_decorator(A):
	def force(x, t):
		return 2*A*t
	return force

def function_decorator(A, b = 0):
	def func(x, t):
		return A*t*x*(x-1) + b
	return func

def forcing_decorator2(A, c):
	def force(x, t):
		return 2*A*(x**2 + t**2 - c*t - x)
	return force

def function_decorator2(A, b, c):
	def func(x, t):
		return A*t*x*(x-1)*(t-c) + b
	return func

def forcing_decorator_trig(A, k1, k2, D):
	def force(x, t):
		Sink1 = torch.sin(k1*torch.pi*x)
		Sink2 = torch.sin(k2*torch.pi*t)
		Cosk2 = torch.cos(k2*torch.pi*t)
		return A*k2*torch.pi*Sink1*Cosk2 + A*D*(torch.pi**2)*(k1**2)*Sink1*Sink2
	return force

def truth_decorator_trig(a, k1, k2, b=0):
	def force(x, t):
		return a*torch.sin(torch.pi*k1*x)*torch.sin(torch.pi*k2*t) + b
	return force
	
def plot_solutions16(solutions, title, savfig = None, figsize=(16, 3.5), subtitle_list = None):
	fig, ax = plt.subplots(2, 8, figsize=figsize)

	for i in range(16):
		global_min = solutions[i].min()
		global_max = solutions[i].max()
		j = i // 8
		k = i % 8
		cmap = plt.get_cmap('viridis')
		im = ax[j][k].imshow(solutions[i][::-1, :], cmap=cmap, vmin=global_min, vmax=global_max)
		ax[j][k].axis('off')
		cbar = fig.colorbar(im, ax=ax[j][k], shrink=0.9, aspect=8)
		cbar.ax.tick_params(labelsize=6)
		if subtitle_list:
			ax[j][k].set_title(subtitle_list[i], fontdict={'family': 'serif', 'color': 'darkred', 'weight': 'bold', 'size': 10})

	plt.suptitle(title)
	plt.subplots_adjust(top=0.93)
	if savfig is not None:
		plt.savefig(savfig, format='eps')

def plot_loss(loss_trace, pde_trace, bc_trace, data_trace=None, figsize=(15, 5), path=None):
	"""
	Plots the loss trace.
	"""
	num_iter = len(loss_trace)
	_, ax = plt.subplots(1, 2, figsize=figsize)
	ax[0].plot(range(1, num_iter+1), np.log10(loss_trace), label='Total Loss')
	ax[1].plot(range(1, num_iter+1), np.log10(pde_trace), label='pde Loss')
	ax[1].plot(range(1, num_iter+1), np.log10(bc_trace), label="BC Loss")
	if data_trace is not None:
		ax[1].plot(range(1, num_iter+1), np.log10(data_trace), label='data Loss')

	ax[0].set_xlabel("Number of iterations")
	ax[0].set_ylabel("Log loss")
	ax[0].set_title("Log Total Loss Value vs. Iteration")
	ax[0].grid()
	ax[0].legend()
	ax[1].set_xlabel("Number of iterations")
	ax[1].set_ylabel("Log loss")
	if data_trace is not None:
		ax[1].set_title("Log pde,data and BC Loss Value vs. Iteration")
	else:
		ax[1].set_title("Log pde, BC Loss Value vs. Iteration")
	ax[1].grid()
	ax[1].legend()
	if path is not None:
		plt.savefig(path, format='eps')

def relative_error(NN_solution, truth_solution):
  	return abs(np.array(NN_solution) - np.array(truth_solution)).mean() / abs(np.array(truth_solution)).mean()

def generate_interior_tensor(I=60, require_grad = True):
	"""
	Generate interior tensors on which we evaluate hidden representation H.
	"""
	N_i = I**2
	generator= Generator2D(grid=(I, I), method='equally-spaced',
							xy_min=(1/200, 1/200), xy_max = (1-1/200, 1-1/200))
	samples = generator.get_examples()
	x = samples[0].unsqueeze(1)
	t = samples[1].unsqueeze(1)
	x = x.cpu(); t = t.cpu()
	if require_grad:
		x.requires_grad_()
		t.requires_grad_()
	interior_tensor = torch.cat([x, t], dim=1)
	return (x, t, interior_tensor)

def generate_boundary_tensor(B=200, require_grad=True, method='uniform'):
	"""
	Generate boundary tensors on which we evaluate hidden representation H.
	"""
	N_b = 3*B
	x_samples = Generator1D(size=B, method=method).get_examples()
	t_samples = Generator1D(size=B, method=method).get_examples()
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
	x_boundary = x_boundary.cpu(); t_boundary = t_boundary.cpu()
	if require_grad:
		x_boundary.requires_grad_()
		t_boundary.requires_grad_()
	boundary_tensor = torch.cat([x_boundary, t_boundary], dim=1)
	return (x_boundary, t_boundary, boundary_tensor)

def after_loss(H_dict, W, forcing_f, b0, x, t, x_boundary=None, t_boundary=None):
	H = H_dict['H']; H_b = H_dict['H_b']; H_star = H_dict['H_star']
	N_b = 3*H_dict['B']
	f_values = forcing_f(x, t).detach().numpy()
	F_values = np.concatenate([f_values.T, np.zeros((f_values.T).shape)]).T.reshape(-1, 1)
	if isinstance(b0, (int, float)):
		b_vec = (np.ones((1, N_b))*b0).reshape(-1, 1)
	else:
		if x_boundary is None or t_boundary is None:
			raise ValueError("x_boundary and t_boundary cannot be none")
		b_vec = b0(x_boundary, t_boundary).detach().numpy() 
	boundary_values = (H_b@W)[[2*i for i in range(N_b)]]
	return ((H_star@W - F_values)**2).mean() + ((boundary_values - b_vec)**2).mean()

def after_loss_from_fvalues(H_dict, W, f_values, b0):
	H = H_dict['H']; H_b = H_dict['H_b']; H_star = H_dict['H_star']
	N_b = 3*H_dict['B']
	F_values = np.concatenate([f_values.T, np.zeros((f_values.T).shape)]).T.reshape(-1, 1)
	b_vec = (np.ones((1, N_b))*b0).reshape(-1, 1)
	boundary_values = (H_b@W)[[2*i for i in range(N_b)]]
	return ((H_star@W - F_values)**2).mean() + ((boundary_values - b_vec)**2).mean()
  
def reaction_diffusion_loss(model, W, r, delta, forcing_function, b0, D = 1, I=60, B=200, bias=True):
	"""
	Compute the Reaction-Diffusion Equation Loss (with the specific nonlinear reaction term).
	"""
	device = torch.device("cpu")
	x, t, interior_tensor = generate_interior_tensor(I=I, require_grad = True)
	x_boundary, t_boundary, boundary_tensor = generate_boundary_tensor(B=B, require_grad=True)

	# Compute the values at interior points
	_, H  = model(interior_tensor)
	H = H.reshape(2*H.shape[0], -1)
	H = H[[2*i for i in range(I**2)]] # Only pick the entries that corresponds to u
	if bias: H = torch.hstack((H, torch.ones((H.shape[0], 1), device=device)))
	u = H@torch.as_tensor(W, device=device)

	# Compute the second order derivatives
	ut = diff(u, t)
	uxx = diff(diff(u, x), x)
	
	f_term = forcing_function(x, t)
	residual = ut - D * uxx - r*(u/(u+1)-delta*u) - f_term
	pde_loss = F.mse_loss(residual, torch.zeros_like(residual, device=device)).item()
	# Compute the relative PDE loss
	f_term_mean = abs(f_term).mean().item()
	residual_mean = abs(residual).mean().item()
	if f_term_mean == 0:
		relative_pde_loss = 'NaN'
	else:
		relative_pde_loss = residual_mean/f_term_mean

	# Compute the boundary loss
	_, H0  = model(boundary_tensor)
	H0 = H0.reshape(2*H0.shape[0], -1)
	H0 = H0[[2*i for i in range(3*B)]]
	if bias: H0 = torch.hstack((H0, torch.ones((H0.shape[0], 1), device=device)))

	# Compute the boundary loss
	u0 = H0@torch.as_tensor(W, device=device)
	bc_loss = F.mse_loss(u0, b0*torch.ones_like(u0, device=device)).item()
	if b0 == 0:
		relative_bc_loss = 'NaN'
	else:
		relative_bc_loss = bc_loss/b0

	result = {
		'total_loss': pde_loss + bc_loss,
		'pde_loss': pde_loss,
		'bc_loss': bc_loss, 
		'relative_pde_loss': relative_pde_loss,
		'relative_bc_loss': relative_bc_loss
	}
	return result
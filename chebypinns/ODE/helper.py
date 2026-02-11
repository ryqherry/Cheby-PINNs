import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from neurodiffeq import diff
from neurodiffeq.neurodiffeq import unsafe_diff as unsafe_diff

def plot_trace(loss_trace, ode_trace, ic_trace):
	"""
	Plot the training loss traces.
	"""
	fig, ax = plt.subplots(1, 2, figsize=(12, 5))
	num_iter = len(loss_trace)
	ax[0].set_yscale('log')
	ax[0].plot(range(1, num_iter+1), loss_trace, label='Total Loss')
	ax[1].set_yscale('log')
	ax[1].plot(range(1, num_iter+1), ode_trace, label='ODE Loss')
	ax[1].plot(range(1, num_iter+1), ic_trace, label="Initial Condition Loss")

	ax[0].set_xlabel("Number of iterations")
	ax[0].set_ylabel("Loss")
	ax[0].set_title("Total Loss Value vs. Iteration")
	ax[0].grid()
	ax[0].legend()
	ax[1].set_xlabel("Number of iterations")
	ax[1].set_ylabel("Loss")
	ax[1].set_title("ODE and IC Loss Value vs. Iteration")
	ax[1].grid()
	ax[1].legend()

def forcing_decorator(gamma, omega):
	def force(t):
		return gamma*torch.cos(omega*t)
	return force

def forcing_decorator_inv_square(gamma):
	def force(t):
		return gamma-torch.exp(-t)
	return force

def forcing_decorator_zero():
	def force(t):
		N = t.shape[0]
		return torch.zeros(N, 6)
	return force

def generate_parameters(k, gamma_domain = (0.5, 3), omega_domain = (0.5, 3), alpha_domain = (0.5, 4.5), delta_domain = (0.5, 4.5), 
						initial_domain = (-3, 3), velocity_domain = (-1, 1), seed = 42):
	np.random.seed(seed)
	gamma_list = np.random.uniform(*gamma_domain, k)
	omega_list = np.random.uniform(*omega_domain, k)
	alpha_list = np.random.uniform(*alpha_domain, k)
	delta_list = np.random.uniform(*delta_domain, k)
	initial_values = np.random.uniform(*initial_domain, k)
	initial_velocities = np.random.uniform(*velocity_domain, k)

	forcing_list = []
	for i in range(k):
		forcing_list.append(forcing_decorator(gamma_list[i], omega_list[i]))

	return {'gamma_list': gamma_list, 'omega_list': omega_list, 'alpha_list': alpha_list, 
			'delta_list': delta_list, 'initial_values': initial_values, 'initial_velocities': initial_velocities, 
			'Forcing_functions': forcing_list}

def generate_parameters_inv_square(k, gamma_domain = (2, 4), alpha_domain = (0.5, 4.5), delta_domain = (0.5, 4.5), 
									initial_domain = (-3, 3), velocity_domain = (-1, 1), seed = 42):
	np.random.seed(seed)
	gamma_list = np.random.uniform(*gamma_domain, k)
	alpha_list = np.random.uniform(*alpha_domain, k)
	delta_list = np.random.uniform(*delta_domain, k)
	initial_values = np.random.uniform(*initial_domain, k)
	initial_velocities = np.random.uniform(*velocity_domain, k)

	forcing_list = []
	for i in range(k):
		forcing_list.append(forcing_decorator_inv_square(gamma_list[i]))

	return {'gamma_list': gamma_list, 'alpha_list': alpha_list, 'delta_list': delta_list, 
			'initial_values': initial_values, 'initial_velocities': initial_velocities, 
			'Forcing_functions': forcing_list}

def RHS_decorator(gamma, w, alpha, delta):
	def func(t, y):
		y = np.array(y)
		A_mat = np.array([[0, -1], [alpha, delta]])
		return -A_mat@y + np.array([0, gamma*np.cos(w*t)])
	return func
	
def compute_MSE(NN_sol, numerical_sol):
	NN_sol = np.array([NN_sol[i].detach().numpy().T for i in range(NN_sol.shape[0])])
	numerical_sol = np.array([ele.y for ele in numerical_sol])
	return ((NN_sol - numerical_sol)**2).mean()

def compute_Ht(H, t):
	"""
	Compute the gradient of H w.r.t. t.
	"""
	output = []
	for i in range(H.shape[1]):
		output.append(diff(H[:,i].reshape(-1, 1), t).detach().numpy())
	return np.concatenate(output, axis=1)

def compute_Htt(H, t):
	"""
	Compute the second order gradient of H w.r.t. t.
	"""
	output = []
	for i in range(H.shape[1]):
		output.append(diff(H[:,i].reshape(-1, 1), t, order=2).detach().numpy())
	return np.concatenate(output, axis=1)

def solve_duffing_cos(delta, alpha, beta, f, u0, domain, t_eval):
	"""
	Solve the ODE with cosine forcing numerically.
	"""
	def F(t, y):
		return [y[1], -delta*y[1]- alpha*y[0] - beta*np.cos(y[0]) + f(t)]
	solution = solve_ivp(F, domain, u0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
	return solution

def compute_duffing_loss_cos(H, Ht, Htt, H0, W, delta, alpha, beta, f_value, u0, v0):
	"""
	Compute the loss for the ODE with cosine forcing.
	"""
	H_total = Htt + delta*Ht + alpha*H
	LHS = H_total @ W
	N = int(LHS.shape[0]/2)
	x = LHS[range(0, 2*N, 2)].flatten() + beta*np.cos((H@W)[range(0, 2*N, 2)].flatten())
	f_value = f_value.flatten()
	ode_loss = ((x - f_value)**2).mean()
	model_initials = H0@W.flatten()
	initials = np.array([u0, v0])
	IC_loss = ((model_initials - initials)**2).mean()
	return {
		'total_loss': IC_loss + ode_loss, 'ode_loss': ode_loss, 'ic_loss': IC_loss
	}

def solve_duffing_inv_square(delta, alpha, beta, f, u0, domain, t_eval):
	"""
	Solve the ODE with inverse square forcing numerically.
	"""
	def F(t, y):
		return [y[1], -delta*y[1]- alpha*y[0] - beta/(y[0]**2) + f(t)]
	solution = solve_ivp(F, domain, u0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
	return solution

def compute_duffing_loss_inv_square(H, Ht, Htt, H0, W, delta, alpha, beta, f_value, u0, v0):
	"""
	Compute the loss for the ODE with inverse square forcing.
	"""
	H_total = Htt + delta*Ht + alpha*H
	LHS = H_total @ W
	N = int(LHS.shape[0]/2)
	x = LHS[range(0, 2*N, 2)].flatten() + beta/(((H@W)[range(0, 2*N, 2)].flatten())**2)
	f_value = f_value.flatten()
	ode_loss = ((x - f_value)**2).mean()
	model_initials = H0@W.flatten()
	initials = np.array([u0, v0])
	IC_loss = ((model_initials - initials)**2).mean()
	return {
		'total_loss': IC_loss + ode_loss, 'ode_loss': ode_loss, 'ic_loss': IC_loss
	}
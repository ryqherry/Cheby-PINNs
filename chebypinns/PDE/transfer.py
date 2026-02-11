from helper import generate_boundary_tensor, generate_interior_tensor

from neurodiffeq import diff
from neurodiffeq.neurodiffeq import unsafe_diff as unsafe_diff

import numpy as np
import time

def compute_Hx(H, x):
	output = []
	for i in range(H.shape[1]):
		output.append(diff(H[:,i].reshape(-1, 1), x).detach().numpy())
	return np.concatenate(output, axis=1)

def compute_Ht(H, t):
	output = []
	for i in range(H.shape[1]):
		output.append(diff(H[:,i].reshape(-1, 1), t).detach().numpy())
	return np.concatenate(output, axis=1)

def compute_AH(A, H):
	N, W_size = H.shape
	A_reshaped = A.reshape(1, 2, 2)
	H_reshaped = H.reshape(-1, 2, W_size)
	AH = np.matmul(A_reshaped, H_reshaped)
	AH = AH.reshape(-1, W_size)
	return AH

def compute_H_dict(model, I, B, bias, D = 1):
	"""
	Compute the latent representation H and other relevant variables for transfer learning.
	"""
	model.to('cpu')
	x, t, interior_tensor = generate_interior_tensor(I=I, require_grad = True)
	x_boundary, t_boundary, boundary_tensor = generate_boundary_tensor(B=B, require_grad=True, method='equally-spaced')
	_, H = model(interior_tensor)
	_, H_b = model(boundary_tensor)

	print("Differentiating H w.r.t. x now...")
	Hx = compute_Hx(H, x)
	print("Finished computing Hx.")
	print("Differentiating H w.r.t. t now...")
	Ht = compute_Ht(H, t)
	print("Finished computing Ht")

	print("Differentiating H_b w.r.t. x now...")
	Hx_b = compute_Hx(H_b, x_boundary)
	print("Finished computing Hx_b.")
	print("Differentiating H_b w.r.t. t now...")
	Ht_b = compute_Ht(H_b, t_boundary)
	print("Finished computing Ht_b")
	
	H = H.detach().numpy()
	H_b = H_b.detach().numpy()
	H = H.reshape(2*H.shape[0], -1)
	H_b = H_b.reshape(2*H_b.shape[0], -1)
	Hx = Hx.reshape(2*Hx.shape[0], -1)
	Ht = Ht.reshape(2*Ht.shape[0], -1)
	Hx_b = Hx_b.reshape(2*Hx_b.shape[0], -1)
	Ht_b = Ht_b.reshape(2*Ht_b.shape[0], -1)

	if bias:
		H = np.hstack((H, np.ones((H.shape[0], 1))))
		H_b = np.hstack((H_b, np.ones((H_b.shape[0], 1))))
		Hx = np.hstack((Hx, np.zeros((Hx.shape[0], 1))))
		Ht = np.hstack((Ht, np.zeros((Ht.shape[0], 1))))
		Hx_b = np.hstack((Hx_b, np.zeros((Hx_b.shape[0], 1))))
		Ht_b = np.hstack((Ht_b, np.zeros((Ht_b.shape[0], 1))))
	
	A1 = np.array([[0, -D], [1, 0]])
	A2 = np.array([[1, 0], [0, 0]])
	A3 = np.array([[0, 0], [0, 1]])
	A1Hx = compute_AH(A1, Hx)
	A2Ht = compute_AH(A2, Ht)
	A3H = compute_AH(A3, H)
	H_star = A1Hx + A2Ht - A3H
	H_dict =  {'H': H, 'H_b': H_b, 'Hx': Hx, 'Ht': Ht, 'Hx_b': Hx_b, 'Ht_b': Ht_b, 'I': I, 'B': B, 'A1Hx': A1Hx, 'A2Ht': A2Ht, 'A3H': A3H, 'H_star': H_star}
	return H_dict

def compute_M(H_dict):
	I = H_dict['I']; B = H_dict['B']
	H_b_0 = H_dict['H_b'][[2*i for i in range(3*B)]]
	N_i = I**2; N_b = 3*B
	M = (H_dict['H_star'].T@H_dict['H_star'])/N_i + (H_b_0.T@H_b_0)/N_b
	Minv = np.linalg.pinv(M)
	return M, Minv

def compute_TLW(b0, forcing_f, Minv, H_dict, x, t, x_boundary=None, t_boundary=None):
	H_b_0 = H_dict['H_b'][[2*i for i in range(3*H_dict['B'])]]
	f_values = forcing_f(x, t).detach().numpy()
	F_values = np.concatenate([f_values.T, np.zeros((f_values.T).shape)]).T.reshape(-1, 1)
	N_i = H_dict['I']**2; N_b = 3*H_dict['B']
	if isinstance(b0, (int, float)): # constant boundary condition
		R = (H_dict['H_star'].T@F_values)/N_i +  ((b0*H_b_0.T).sum(axis=1)/N_b).reshape(-1, 1)
	else:
		if x_boundary is None or t_boundary is None:
			raise ValueError('x_boundary and t_boundary cannot be None when the boundary condition is not constant')
		b0 = b0(x_boundary, t_boundary).detach().numpy() 
		R = (H_dict['H_star'].T@F_values)/N_i + (H_b_0.T@b0)/N_b
	W = Minv@R
	return W

def compute_TLW_from_fvalues(b0, f_values, Minv, H_dict, x_boundary=None, t_boundary=None):
	H_b_0 = H_dict['H_b'][[2*i for i in range(3*H_dict['B'])]]
	F_values = np.concatenate([f_values.T, np.zeros((f_values.T).shape)]).T.reshape(-1, 1)
	N_i = H_dict['I']**2; N_b = 3*H_dict['B']
	if isinstance(b0, (int, float)): # constant boundary condition 
		R = (H_dict['H_star'].T@F_values)/N_i +  ((b0*H_b_0.T).sum(axis=1)/N_b).reshape(-1, 1)
	else:
		if x_boundary is None or t_boundary is None:
			raise ValueError('x_boundary and t_boundary cannot be None when the boundary condition is not constant')
		b0 = b0(x_boundary, t_boundary).detach().numpy() 
		R = (H_dict['H_star'].T@F_values)/N_i + (H_b_0.T@b0)/N_b
	W = Minv@R
	return W

def compute_solution(H, W, I):
	H_ = H[[2*i for i in range(round(H.shape[0]/2))],:]
	return (H_@W).reshape(I, -1)

def combine_W(W_list, r):
	W = W_list[0].copy()
	for i in range(1, len(W_list)):
		W += (r**i)*W_list[i]
	return W

def compute_each_b(b, r, p):
	deno = 1
	for i in range(1, p+1):
		deno += r**i
	return b/deno

def cheb_mapping_params(u_min, u_max):
	"""
	Compute the mapping for chebyshev.
	"""
	alpha = 2.0 / (u_max - u_min)
	beta  = -(u_min + u_max) / (u_max - u_min)
	return alpha, beta

def rd_cheb_coeffs(u_min, u_max, K, M, delta):
	"""
	Compute the Chebyshev coefficients ak for the reaction term in the reaction-diffusion equation.
	"""
	j = np.arange(1, M + 1)
	theta = (2*j - 1) * np.pi / (2.0 * M)
	s = np.cos(theta)
	u = 0.5 * (u_max - u_min) * s + 0.5 * (u_min + u_max)
	g = u / (1.0 + u) - delta * u

	c = np.zeros(K + 1)
	c[0] = (1.0 / M) * np.sum(g)
	if K >= 1:
		cos_k_theta = np.cos(np.outer(np.arange(1, K + 1), theta))
		c[1:] = (2.0 / M) * (cos_k_theta @ g)
	return c

def rd_C_0(S0, K):
	"""
	Builds the first column of chebyshev polynomials C_{k,0} = T_k(S0) for k=0..K
	"""
	S0 = np.asarray(S0)
	Ck0 = [None] * (K + 1)
	Ck0[0] = np.ones_like(S0)
	if K >= 1:
		Ck0[1] = S0.copy()
		for k in range(1, K):
			Ck0[k + 1] = 2.0 * S0 * Ck0[k] - Ck0[k - 1]
	return Ck0

def rd_C_update(C_cols, S_cols, j_new, K):
	"""
	Builds a new column of chebyshev polynomials C_{k,j_new} for k=0..K
	"""
	S_cols = [np.asarray(S) for S in S_cols]
	Cj = [None] * (K + 1)
	Cj[0] = np.zeros_like(S_cols[0])
	Cj[1] = S_cols[j_new].copy()

	for k in range(1, K):
		acc = np.zeros_like(S_cols[0])
		for p in range(0, j_new + 1):
			C_k_j_minus_p = C_cols[j_new - p][k]
			acc += S_cols[p] * C_k_j_minus_p
		Cj[k + 1] = 2.0 * acc - Cj[k - 1]

	C_cols.append(Cj)
	return C_cols

def rd_build_rhs_from_C(a, C_col_j):
	"""
	Builds the right-hand side R_j = sum_{k=0}^K a_k * C_{k,j}.
	"""
	a = np.asarray(a)
	G = a[0] * C_col_j[0]
	for k in range(1, len(a)):
		G += a[k] * C_col_j[k]
	return G

def reaction_diffusion_solver(H_dict, Minv, c, u_min, u_max, f0_values, b, epsilon, H, p=11, display=False):
	"""
	Solves the reaction-diffusion equation per order via chebyshev and perturbation expansions.
	"""
	t0 = time.time()
	alpha, beta = cheb_mapping_params(u_min, u_max)
	b_each = compute_each_b(b, epsilon, p)
	W_list = []; u_list = []; f_list = []

	# Solve the 0th order PDE
	W0 = compute_TLW_from_fvalues(b_each, f0_values, Minv, H_dict)
	u0 = compute_solution(H_dict['H'], W0, I=H_dict['I'])
	W_list.append(W0); u_list.append(u0); f_list.append(f0_values)

	K = len(c) - 1
	S_list = []
	S0 = alpha * u0.reshape(-1) + beta
	S_list.append(S0)
	C_cols = [rd_C_0(S0, K)]

	# Solve for higher orders
	for i in range(1, p+1):
		j = i - 1
		C_cols = rd_C_update(C_cols, S_list[: j + 1], j, K)
		C_j = C_cols[j]
		R_j = rd_build_rhs_from_C(c, C_j)
		R_j = R_j.reshape(-1, 1)
		W_i = compute_TLW_from_fvalues(b_each, R_j, Minv, H_dict)
		u_i = compute_solution(H_dict['H'], W_i, I=H_dict['I'])
		f_list.append(R_j.copy()); W_list.append(W_i); u_list.append(u_i)
		S_list.append(alpha * u_i.reshape(-1))
	
	t1 = time.time()
	W = combine_W(W_list, epsilon)
	sol = compute_solution(H, W, I=round((H.shape[0]/2)**.5))

	result = {'W_list': W_list, 'u_list': u_list, 'f_list': f_list, 'W': W, 'sol': sol}
	if display:
		print("All PDEs solved in {} seconds. On average, each PDE is solved using {} seconds".format(round(t1-t0, 6), round((t1-t0)/(p+1), 6)))

	return result
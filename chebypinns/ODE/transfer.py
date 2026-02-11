import torch
import numpy as np
from numpy.polynomial.chebyshev import chebval
from math import factorial

def compute_AH(A, H):
	A_reshaped = A.reshape(1, 2, 2)
	H_reshaped = H.reshape(-1, 2, H.shape[-1])
	AH = np.matmul(A_reshaped, H_reshaped)
	AH = AH.reshape(-1, AH.shape[-1])
	return AH

def compute_M(H, Ht, AH, H0):
	N = H.shape[0]/2
	HtAH = (Ht.T @ AH)
	HAHt = (AH).T @ Ht
	HtHt = Ht.T @ Ht
	HAAH = (AH).T @ AH
	H0tH0 = H0.T @ H0
	M = (HtAH + HAHt + HtHt + HAAH)/N + H0tH0
	Minv = np.linalg.pinv(M)
	return M, Minv

def compute_TLW(f, initial_value, initial_velocity, Ht, AH, H0, Minv, t_grids):
	"""
	Computes and returns the transfer learning head W.
	"""
	u0 = np.array([[initial_value], [initial_velocity]])
	f_values = f(t_grids).detach().numpy()
	F = np.concatenate([np.zeros((f_values.T).shape), f_values.T]).T.reshape(-1, 1)
	N = t_grids.shape[0]
	R = H0.T @ u0 + (Ht.T @ F)/N + (AH.T @ F)/N
	W = Minv @ R
	return W

def compute_TLW_fromfvalues(f_values, initial_value, initial_velocity, Ht, AH, H0, Minv):
	u0 = np.array([[initial_value], [initial_velocity]])
	F = np.concatenate([np.zeros((f_values.T).shape), f_values.T]).T.reshape(-1, 1)
	N = f_values.shape[0]
	R = H0.T @ u0 + (Ht.T @ F)/N + (AH.T @ F)/N
	W = Minv @ R
	return W

def compute_TLsolution(W, H = None, t_grids = None, model=None, dtype=torch.float32):
	if H is None and t_grids is None:
		raise ValueError("H and t_grids cannot both be None");
	result_dict = {}
	if H is not None:
		result = (H @ W).reshape(-1, 2)
		result_dict['x'] = result[:,0]
		result_dict['y'] = result[:,1]
	else:
		if model is None: raise ValueError("model cannot be None");
		t_grids = torch.tensor(t_grids, dtype=dtype).view(-1, 1)
		_, H = model(t_grids)
		H = np.hstack((H.reshape(2*H.shape[0], -1).detach().numpy(), np.ones((2*H.shape[0], 1))))
		result = (H @ W).reshape(-1, 2)
		result_dict['x'] = result[:,0]
		result_dict['y'] = result[:,1]
	return result_dict

def One_Shot_solve(f, H, Ht, H0, alpha, delta, initial_value, initial_velocity, t_grids):
	"""
	Computes the solution of an unseen linear ODE system.
	"""
	A = np.array([[0, -1], [alpha, delta]])
	AH = compute_AH(A, H)
	M, Minv = compute_M(H, Ht, AH, H0)
	W = compute_TLW(f, initial_value, initial_velocity, Ht, AH, H0, Minv, t_grids)
	result = compute_TLsolution(W, H)
	result['W'] = W
	return result

def combine_W(W_list, beta):
	W = W_list[0].copy()
	for i in range(1, len(W_list)):
		W += (beta**i)*W_list[i]
	return W

def compute_each_iv(iv, beta, p):
	deno = 1
	for i in range(1, p+1):
		deno += beta**i
	return iv/deno

def generate_valid_multiindices(target, p, q):
	"""
	Generates valid multiindices for the inner sum in the perturbation expansion.
	"""
	total = 2 * q
	result = []
	i = [0] * (p + 1)

	def backtrack(idx, sum_i, sum_w):
		rem = total - sum_i
		if idx == p:
			val = rem
			if sum_w + idx * val == target:
				i[idx] = val
				result.append(tuple(i))
			return
		
		lb, ub = 0, rem
		if idx > 0:
			ub = min(ub, (target - sum_w) // idx)
		if idx < p:
			num = target - sum_w - p * rem
			den = idx - p
			ub = min(ub, num // den)

		lb = max(lb, sum_w + (idx + 1) * rem - target)
		if lb > ub:
			return
		
		for x in range(lb, ub + 1):
			i[idx] = x
			backtrack(idx + 1, sum_i + x, sum_w + idx * x)

	backtrack(0, 0, 0)
	return result

def compute_C_kj_cos(k, j, y_series):
	"""
	Computes the Chebyshev convolution coefficients for the cosine forcing case.
	"""
	p = len(y_series) - 1
	if k == 0:
		return np.ones_like(y_series[0]) if j == 0 else np.zeros_like(y_series[0])
	if k == 1:
		return y_series[j] if j <= p else np.zeros_like(y_series[0])
		
	# Initialize T_{n-1} and T_n series arrays for n=1
	T_nm1 = [np.zeros_like(y_series[0]) for _ in range(p+1)]
	T_nm1[0] = np.ones_like(y_series[0])
	# T1(epsilon) = y_series
	T_n = [y_series[m].copy() for m in range(p+1)]
		
	for n in range(1, k):
		# Series convolution conv = y_series * T_n
		conv = [np.zeros_like(y_series[0]) for _ in range(p+1)]
		for m in range(p+1):
			ym = y_series[m]
			for l in range(p+1-m):
				conv[m+l] += ym * T_n[l]
		# Build T_{n+1} = 2 * conv - T_{n-1}
		T_np1 = [np.zeros_like(y_series[0]) for _ in range(p+1)]
		for m in range(p+1):
			T_np1[m] = 2 * conv[m] - T_nm1[m]
		
		T_nm1, T_n = T_n, T_np1
		
	return T_n[j] if j <= p else np.zeros_like(y_series[0])

def compute_chebyshev_ak_cos(N, a, b, M=500):
	"""
	Computes the Chebyshev coefficients a_k for the cosine forcing case.
	"""
	# Gauss–Chebyshev nodes t_j in [-1, 1]
	j = np.arange(M)
	t = np.cos((2*j + 1) * np.pi / (2*M))
	x = 0.5*(a+b) + 0.5*(b-a)*t
	f = np.cos(x)
		
	# Compute each a_k via weighted sum
	ak = np.empty(N + 1)
	for k in range(N+1):
		T_k = chebval(t, [0]*k + [1])
		factor = (1/M) if k == 0 else (2/M)
		ak[k] = factor * np.dot(f, T_k)
		
	return ak

def compute_C_kj_inv_square(k, j, y_series):
	"""
	Computes the Chebyshev convolution coefficients for the inverse square forcing case.
	"""
	p = len(y_series) - 1
	if k == 0:
		return y_series[0]*0 + (1 if j == 0 else 0)
	if k == 1:
		return y_series[j] if j <= p else y_series[0]*0

	# Initialize T0 and T1 as series in ε
	T_nm1 = [y_series[0]*0 + 1] + [y_series[0]*0 for _ in range(p)]
	T_n   = [ys.copy() for ys in y_series]

	for n in range(1, k):
		# Convolution y_series * T_n to degree p
		conv = [y_series[m] * T_n[l]
				for m in range(p+1)
				for l in range(p+1-m)]
		# Reorganize into length-(p+1) list
		conv_series = [sum(conv[m*(p+1):m*(p+1)+p+1-m]) 
						if m < p+1 else 0 
						for m in range(p+1)]
		# Build T_{n+1} = 2*conv_series - T_nm1
		T_np1 = [2*c - T_nm1[idx] for idx, c in enumerate(conv_series)]
		T_nm1, T_n = T_n, T_np1

	return T_n[j] if j <= p else y_series[0]*0

def compute_chebyshev_ak_inv_square(N, a, b, M=1000):
	"""
	Computes the Chebyshev coefficients a_k for the inverse square forcing case.
	"""
	# Gauss–Chebyshev nodes t_j in [-1, 1]
	j = np.arange(M)
	t_j = np.cos((2*j + 1) * np.pi / (2 * M))
		
	# Map nodes to [a, b]
	x = 0.5 * (b + a) + 0.5 * (b - a) * t_j
		
	# Evaluate f(x) = 1/x^2 at these points
	f_j = 1.0 / (x**2)
		
	# Compute each a_k via weighted sum
	ak = np.zeros(N + 1)
	for k in range(N + 1):
		# Evaluate T_k at t_j
		T_k = chebval(t_j, [0]*k + [1])
		weight = (1.0 / M) if k == 0 else (2.0 / M)
		ak[k] = weight * np.dot(f_j, T_k)
		
	return ak

def p_shot_solve(f0_values, beta, alpha, delta, initial_value, initial_velocity, H, Ht, H0, p=12, n=10):
	"""
	Computes the solution of a nonlinear ODE using perturbation expansion only.
	"""
	A = np.array([[0, -1], [alpha, delta]])
	AH = compute_AH(A, H)
	M, Minv = compute_M(H, Ht, AH, H0)
	# Set initial conditions for all p+1 ODE systems
	ini_val = compute_each_iv(initial_value, beta, p)
	ini_vel = compute_each_iv(initial_velocity, beta, p)
	W_list = []; x_list = []; f_list = []; y_list = []
	# Compute the 0th order ODE
	W0 = compute_TLW_fromfvalues(f0_values, ini_val, ini_vel, Ht, AH, H0, Minv)
	u0 = compute_TLsolution(W0, H)
	W_list.append(W0); x_list.append(u0['x']); f_list.append(f0_values); y_list.append(u0['y'])
		
	# Solve the remaining ODEs
	for i in range(1, p+1):
		target = i-1
		fi_values = np.zeros(x_list[0].shape)
		for q in range(n+1):
			indices = generate_valid_multiindices(target, p, q)
			for combo in indices:
				coeff = ((-1)**q) / np.prod([factorial(k_i) for k_i in combo])
				term = np.ones_like(x_list[0])
				for idx, k_i in enumerate(combo):
					if k_i > 0:
						term *= x_list[idx]**k_i
				fi_values -= coeff * term

		fi_values = fi_values.reshape(-1, 1)
		# Solve the ith ode
		Wi = compute_TLW_fromfvalues(fi_values, ini_val, ini_vel, Ht, AH, H0, Minv)
		ui = compute_TLsolution(Wi, H) 
		W_list.append(Wi); x_list.append(ui['x']); f_list.append(fi_values); y_list.append(ui['y'])
	
	W = combine_W(W_list, beta)
	x = combine_W(x_list, beta)
	y = combine_W(y_list, beta)
	dict = {
			'W': W, 'x': x, 'y': y, 'W_list': W_list, 'x_list': x_list, 'y_list': y_list, 'f_list': f_list
	}
	return dict

def p_shot_solve_cheb_cos(f0_values, beta, alpha, delta, initial_value, initial_velocity, H, Ht, H0, p=12, n=10, a=-4, b=4):
	"""
	Computes the solution of the nonlinear ODE with cosine forcing via Chebyshev and perturbation expansions.
	"""
	A = np.array([[0, -1], [alpha, delta]])
	AH = compute_AH(A, H)
	M, Minv = compute_M(H, Ht, AH, H0)
	a_k = compute_chebyshev_ak_cos(n, a, b)
	# Set initial conditions for all p+1 ODE systems
	ini_val = compute_each_iv(initial_value, beta, p)
	ini_vel = compute_each_iv(initial_velocity, beta, p)
	W_list = []; x_list = []; f_list = []; y_list = []
	# Compute the 0th order ODE solution
	W0 = compute_TLW_fromfvalues(f0_values, ini_val, ini_vel, Ht, AH, H0, Minv)
	u0 = compute_TLsolution(W0, H)
	W_list.append(W0); x_list.append(u0['x']); f_list.append(f0_values); y_list.append(u0['y'])
		
	# Solve the remaining ODE solutions
	for i in range(1, p+1):
		target = i-1
		fi_values = np.zeros(x_list[0].shape)
		y_series = [(2*x_list[m] - (a + b)) / (b - a) if m==0 else (2*x_list[m] / (b - a)) for m in range(i)]
		for q in range(n+1):
			C_kj_minus1 = compute_C_kj_cos(q, i-1, y_series)
			fi_values -= a_k[q] * C_kj_minus1
		fi_values = fi_values.reshape(-1, 1)
		# Solve the ith ode
		Wi = compute_TLW_fromfvalues(fi_values, ini_val, ini_vel, Ht, AH, H0, Minv)
		ui = compute_TLsolution(Wi, H) 
		W_list.append(Wi); x_list.append(ui['x']); f_list.append(fi_values); y_list.append(ui['y'])

	W = combine_W(W_list, beta)
	x = combine_W(x_list, beta)
	y = combine_W(y_list, beta)
	dict = {
			'W': W, 'x': x, 'y': y, 'W_list': W_list, 'x_list': x_list, 'y_list': y_list, 'f_list': f_list
	}
	return dict

def p_shot_solve_cheb_inv_square(f0_values, beta, alpha, delta, initial_value, initial_velocity, H, Ht, H0, p=12, n=10, a=0.1, b=5, M_nodes=1000):
	"""
	Computes the solution of the nonlinear ODE with inverse square forcing via Chebyshev and perturbation expansions.
	"""
	A = np.array([[0, -1], [alpha, delta]])
	AH = compute_AH(A, H)
	M, Minv = compute_M(H, Ht, AH, H0)
	a_k = compute_chebyshev_ak_inv_square(n, a, b, M_nodes)
	# Set initial conditions for all p+1 ODE systems
	ini_val = compute_each_iv(initial_value, beta, p)
	ini_vel = compute_each_iv(initial_velocity, beta, p)
	W_list = []; x_list = []; f_list = []; y_list = []
	# Compute the 0th order ODE solution
	W0 = compute_TLW_fromfvalues(f0_values, ini_val, ini_vel, Ht, AH, H0, Minv)
	u0 = compute_TLsolution(W0, H)
	W_list.append(W0); x_list.append(u0['x']); f_list.append(f0_values); y_list.append(u0['y'])
	
	# Solve the remaining ODE solutions
	for i in range(1, p+1):
		target = i-1
		fi_values = np.zeros(x_list[0].shape)
		y_series = [(2*x_list[m] - (a + b)) / (b - a) if m==0 else (2*x_list[m] / (b - a)) for m in range(i)]
		for q in range(n+1):
			C_kj_minus1 = compute_C_kj_inv_square(q, i-1, y_series)
			fi_values -= a_k[q] * C_kj_minus1
		fi_values = fi_values.reshape(-1, 1)
		# Solve the ith ODE
		Wi = compute_TLW_fromfvalues(fi_values, ini_val, ini_vel, Ht, AH, H0, Minv)
		ui = compute_TLsolution(Wi, H) 
		W_list.append(Wi); x_list.append(ui['x']); f_list.append(fi_values); y_list.append(ui['y'])
	
	W = combine_W(W_list, beta)
	x = combine_W(x_list, beta)
	y = combine_W(y_list, beta)
	dict = {
			'W': W, 'x': x, 'y': y, 'W_list': W_list, 'x_list': x_list, 'y_list': y_list, 'f_list': f_list
	}
	return dict
from numpy.random import random, choice, shuffle
from numpy import dot, zeros, diag, fill_diagonal, cos, sin, array, amax, argsort, shape, nonzero, append, linspace
from math import exp, pi, floor
from itertools import product

from helper import *

import matplotlib.pyplot as plt

# Dwave
from dwave_sapi2.remote import RemoteConnection
from dwave_sapi2.core import async_solve_ising, await_completion
from dwave_sapi2.util import get_hardware_adjacency
from dwave_sapi2.embedding import embed_problem, unembed_answer
from minorminer import find_embedding
import dwave_networkx as dnx


from time import sleep

#from gurobipy import Model, GRB, QuadExpr

# # Branch and Bound
# from branchandbound_tools import *
# from sympy import S
# from spin_models import *
# import numpy as np
# import chompack
# from ncpol2sdpa.chordal_extension import find_variable_cliques
# from ncpol2sdpa import generate_variables, flatten, SdpRelaxation, get_monomials


#===============
#  DWAVE INFO  #
#===============

solver_name = 'SOLVER NAME'
url = 'URL'
token = 'TOKEN'
remote_connection = 0
solver = 0
adj = 0

def sign_in():

	global solver
	global adj

	print('Connecting to DWave')
	remote_connection = RemoteConnection(url, token)
	solver = remote_connection.get_solver(solver_name)
	adj = list(get_hardware_adjacency(solver))


#==========
# Solvers #
#==========

# Description: Brute force scan over all solutions
# Inputs:	- pot: dictionary describing potential and solver
#			- states: possible states for each variable to be in
# Outputs:	- energy of solution and the solution itself 
# Description: Search over all of state space and find the minimum
#=============================
def brute_force(pot, states, show_spec = False):

	J, const = dict_2_mat(pot)
	h = diag(J).copy()
	fill_diagonal(J, 0.0)

	# Scan over all possible states and calculate spectrum
	res = []
	for st in product(states, repeat = pot['num vars']):
		st = array(st)
		res += [h.dot(st) + st.dot(J.dot(st.transpose())) + const]

	# Find ordering of states according to increasing energy
	res = array(res)
	inds = argsort(res)

	if show_spec:
		plt.scatter(range(2**pot['num vars']), res[inds], marker = '.')
		plt.show()

	# Convert state in "index format" to actual state (e.g. inds[0] = 2 => state = 000..010)
	r = inds[0]
	state = [0 for i in range(pot['num vars'])]
	for i in range(pot['num vars'])[::-1]:
		state[i] = r % 2
		r = int(r/2)

	state = array(state)
	if -1 in states: state = 2*state - 1

	# Reorder according to the mapping constructed in dict_2_mat
	state = [state[pot['mapping'][i]] for i in range(pot['num vars'])]

	return res[inds[0]], state



# Description: Spin Vector Monte Carlo, effectively simulated annealing with a phase component (arXiv: 1401.7087)
# Inputs:	- pot: the potential/hamiltonian other than the transverse field... dictionary
#			- states: which states the variables can be in (right now will convert to spin)... list
#			- temp: temperature to set the scale for the metropolis updates... float
#			- num_cycles: how many iterations/cycles to run for... int
#			- A: the transverse field schedule... list with an entry for each iteration/cycle
#			- B: problem hamiltonian schedule... list with an entry for each iteration/cycle
# Outputs:	- the answer according to svmc
#=====================
def svmc(pot_, states = [-1, 1], temp = 5.0, num_cycles = 100, num_moves = 1000, A = None, B = None):

	if states == [0, 1]:
		qubo = True
		pot = qubo_2_ising(pot_)
		states = [-1, 1]
	else: 
		pot = pot_
		qubo = False

	J, const = dict_2_mat(pot)
	h = diag(J).copy()
	fill_diagonal(J, 0.0)
	upper_bound = sum(abs(h)) + sum(sum(abs(J))) + abs(const)


	if B is None: B = linspace(0, 1, num = num_cycles)#[1.0/high*(exp(x*1.0/num_cycles) - 1) for x in range(num_cycles)]
	if A is None: A = B[::-1]*upper_bound # this should put them on roughly equal scales

	# Random initial state
	num_vars = pot_['num vars']
	state = [random()*2.0*pi for i in range(num_vars)]
	updates = range(num_vars)
	new_states = [[random()*2.0*pi for i in range(num_vars)] for move in range(num_moves)]

	for t in range(num_cycles):
		for move in range(num_moves):
			shuffle(updates)
			for spin in updates:

				trans_field = A[t]*sin(new_states[move][spin]) - sin(state[spin])
				prob_ham = B[t]*cos(new_states[move][spin]) - cos(state[spin])*(h[spin] + J[spin, :].dot(cos(state)))
				delta = (trans_field + prob_ham)/temp
				if delta >= -700: #avoid overflow errors
					if random() <= min(1.0, exp(-delta)): state[spin] = new_states[move][spin]
				else: state[spin] = new_states[move][spin]


	# Project onto computational basis
	if qubo: state = {-1: 0, 0: 1}
	else: state = {-1: -1, 0: 1}

	state = [state[floor(cos(state[i]))] for i in range(num_vars)]
	energy = array(h).dot(state) + array(state).dot(J.dot(array(state).transpose())) + const

	return energy, state


# Description: Simulated Annealing
# Inputs:	- pot: potential defining the energy of states... dictionary of tuple: strength
#			- states: the possible states each variable can be in... list of ints
#			- T: the temperature schedule... list of ints (length num_cycles)
#			- num_cycles: the number of iterations to run... int
#			- num_moves: the number of moves to make at each temperature/cycle
# Outputs:	- the energy of the found state (the guess at the ground state energy)
#			- the corresponding state at that energy
#====================================================
def sa(pot, states, T = None, num_cycles = 100, num_moves = 1000):

	J, const = dict_2_mat(pot)
	h = diag(J).copy()
	fill_diagonal(J, 0.0)

	if T is None: 
		T = range(num_cycles - 1, -1, -1) #T = [1.0/high*exp(x*1.0/num_cycles) for x in range(num_cycles)]
		T[-1] = 0.1

	# Random initial state
	num_vars = pot['num vars']
	state = choice(states, num_vars)
	updates = range(num_vars)

	# Anneal
	for temp in T:
		for move in range(num_moves):
			shuffle(updates)
			for spin in updates:

				delta = -2*state[spin]*(h[spin] + J[spin, :].dot(state))/temp
				if delta >= -700: #avoid overflow errors
					if random() <= min(1.0, exp(-delta)): state[spin] = -state[spin]
				else: state[spin] = -state[spin]


			#new_state = 
			#delta = new_state.dot(J.dot(new_state.transpose())) - state.dot(J.dot(state.transpose())) + h.dot(new_state) - h.dot(state)
			#delta /= temp

			#if delta <= 700:  # To avoid overflow errors
			#	if random() <= 1.0/max(1.0, exp(delta)): state = new_state.copy()


	return h.dot(state) + state.dot(J.dot(state.transpose())) + const, state



# Description: submit a potential to dwave to try to find a solution
# Inputs:	- pot: the potential function to be minimized... dictionary
#			- states: not currently used as it's assumed the states are [-1, 1] for this function
# Outputs:	- the minimum energy found
#			- the corresponding state
#=======================================
def dwave(pot, states):

	solved = False
	const = 0
	h_ = []
	J_ = {}
	state = []
	free_state = []

	while not solved:
		try:
			global solver
			global adj

			if solver == 0: sign_in() #should try to make it so there is a pool of pool_size connections that the various threads can use


			# Map problem into a dwave consumable format

			# J_, const = dict_2_mat(pot)
			# h_ = diag(J_).copy()
			# fill_diagonal(J_, 0.0)
			# j = mat_2_dict(J_)

			# if 'num vars' in j: del j['num vars']
			# if 'const' in j: del j['const']
			# if 'mapping' in j: del j['mapping']

			#const, h_, j, free_state = dwave_prepare(pot)

			const, h_, j, prob_adj = dwave_prepare(pot)

			# Find an embedding and prepare to send physical problem to dwave
			embedding = []
			while len(embedding) == 0:
				embedding = find_embedding(prob_adj, adj).values()

			[h, J, chains, embedding] = embed_problem(h_, j, embedding, adj)


			s = 0.75
			h = [a*s for a in h]
			for k in J: J[k] = J[k]*s
			for k in chains:
				if k in J: J[k] += chains[k]
				else: J[k] = chains[k]


			# Submit problem
			#print('submitting problem')

			while True:
				submitted_problems = [async_solve_ising(solver, h, J, num_reads = 1000, num_spin_reversal_transforms = 10, answer_mode = 'histogram', auto_scale = True)]
				await_completion(submitted_problems, len(submitted_problems), float('180'))
				if submitted_problems[0].done(): break

			res = unembed_answer(submitted_problems[0].result()['solutions'], embedding, 'discard')

			if len(res) > 0:
				state = array(res[0])
				solved = True

		except Exception as err:
			print(err)
			solved = False
			sleep(120) # wait about 2 minutes and then retry


	if len(h_) != len(state):
		print(h_, len(h_))
		print(state, len(state))

		print(pot)

	J_, _ = dict_2_mat(j, len(h_))
	energy = h_.dot(state) + state.dot(J_.dot(state.transpose())) + const


	#for v in sorted(free_state):
	#	energy += pot[v]*free_state[v]
	#	state = append(state, free_state[v])

	return energy, state


# Function: gurobi
# Description: solve the given qubo using gurobi
# Inputs:	- Q: qubo... dictionary
#			- num_vars: number of variables... int
# Outputs:	- answer from gurobi... list
#=====================================================
# def gurobi(pot, states):

# 	if states == [-1, 1]:
# 		ising = True
# 		pot_ = qubo_2_ising(pot)
# 		states = [0, 1]
# 	else: 
# 		pot_ = pot
# 		ising = False

# 	Q, const = dict_2_mat(pot_)
# 	num_vars = pot_['num vars']

# 	m = Model('lat')
# 	m.setParam( 'OutputFlag', False )

# 	# Collect the variables and construct the objective function (the qubo)
# 	vs = [m.addVar(name = str(i), vtype = GRB.BINARY) for i in range(num_vars)]
# 	obj = QuadExpr()
# 	for i in range(num_vars):
# 		for j in range(num_vars):
# 			if Q[i, j] != 0: obj += Q[i, j]*vs[i]*vs[j]

# 	# Run optimizer and get solutions
# 	m.setObjective(obj, GRB.MINIMIZE)
# 	m.update()
# 	m.optimize()

# 	state = array(m.X)

# 	if ising: 
# 		state = 2*state - 1
# 		J, const = dict_2_mat(pot)
# 		h = diag(J).copy()
# 		fill_diagonal(J, 0.0)
# 		energy = h.dot(state) + state.dot(J.dot(state.transpose())) + const
# 	else:
# 		energy = state.dot(Q.dot(state.transpose())) + const

# 	return energy, state





def sqa(pot, states): pass


# # Taken from Arxiv 1808.01275
# def cbnb(pot, states):

# 	# setting the precision for the SDP Solver
# 	_precision = 10**(-5)
# 	solverparameters = {
#     'dparam.intpnt_co_tol_rel_gap': _precision,
#     'dparam.intpnt_co_tol_mu_red': _precision,
#     'dparam.intpnt_nl_tol_rel_gap': _precision,
#     'dparam.intpnt_nl_tol_mu_red': _precision,
#     'dparam.intpnt_tol_rel_gap': _precision,
#     'dparam.intpnt_tol_mu_red': _precision,
#     'dparam.intpnt_co_tol_dfeas': _precision,
#     'dparam.intpnt_co_tol_infeas': _precision,
#     'dparam.intpnt_co_tol_pfeas': _precision,
# 	}


# 	# Build up variables and hamiltonian
# 	if 'mapping' not in pot: map_vars(pot)
# 	mapping = pot['mapping']
# 	vs = []
# 	substitutions = {}
# 	for i in pot.keys():
# 		if type(i) == type(0): 
# 			vs.append(generate_variables(str(mapping[i])))
# 			substitutions[vs[mapping[i]]**2] = S.One

# 	ham = 0.0
# 	for i in pot.keys():
# 		if type(i) == type(0): ham += pot[i]*vs[mapping[i]]
# 		if type(i) == type((0, 0)):
# 			a, b = i
# 			ham += pot[i]*vs[mapping[a]]*vs[mapping[b]]

# 	# Solve using CBnB with a given threshold for n_t
# 	threshold = 3
# 	cliques = find_variable_cliques(flatten(vs), ham)
# 	z_low, z_up, state = get_groundBandB(vs, substitutions, ham, cliques, threshold, solverparameters, verbose = 0)


# 	return z_up, state


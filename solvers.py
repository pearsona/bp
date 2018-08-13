from numpy.random import random, choice
from numpy import dot, zeros, diag, fill_diagonal, cos, sin, array, amax
from math import exp, pi, floor
from itertools import product

from helper import *


from dwave_sapi2.remote import RemoteConnection
from dwave_sapi2.core import async_solve_ising, await_completion
from dwave_sapi2.util import get_hardware_adjacency
from dwave_sapi2.embedding import embed_problem, unembed_answer
from minorminer import find_embedding


#===============
#  DWAVE INFO  #
#===============

solver_name = 'C16'#'DW2X'
url = 'https://qfe.nas.nasa.gov/sapi'#'https://usci.qcc.isi.edu/sapi'
token = 'NASA-f73f6a756b922f9ebfcb6127740bec11bf986527'#'RnD-27a8a3832f9f6f8aa1b74c192a3098802c91c7a1'
remote_connection = RemoteConnection(url, token)
solver = remote_connection.get_solver(solver_name)
adj = list(get_hardware_adjacency(solver))



#==========
# Solvers #
#==========

# Inputs:	- pot: dictionary describing potential and solver
#			- states: possible states for each variable to be in
# Outputs:	- energy of solution and the solution itself 


# Description: Search over all of state space and find the minimum
#=============================
def brute_force(pot, states):

	min_so_far = 10e9

	J, const = dict_2_mat(pot)
	h = diag(J).copy()
	fill_diagonal(J, 0.0)

	for st in product(states, repeat = pot['num vars']):
		st = array(st)
		energy = h.dot(st) + st.dot(J.dot(st.transpose())) + const

		if energy < min_so_far:
			min_so_far = energy
			state = st


	return min_so_far, st


# Description: Spin Vector Monte Carlo, effectively simulated annealing with a phase component (arXiv: 1401.7087)
# Inputs:	- pot: the potential/hamiltonian other than the transverse field... dictionary
#			- states: which states the variables can be in (right now will convert to spin)... list
#			- temp: temperature to set the scale for the metropolis updates... float
#			- num_cycles: how many iterations/cycles to run for... int
#			- A: the transverse field schedule... list with an entry for each iteration/cycle
#			- B: problem hamiltonian schedule... list with an entry for each iteration/cycle
# Outputs:	- the answer according to svmc
#=====================
def svmc(pot_, states = [-1, 1], temp = 5.0, num_cycles = 20, A = None, B = None):

	if states == [0, 1]:
		qubo = True
		pot = ising_2_qubo(pot_)
		states = [-1, 1]
	else: 
		pot = pot_
		qubo = False

	J, const = dict_2_mat(pot)
	h = diag(J).copy()
	fill_diagonal(J, 0.0)
	high = max(max(amax(abs(J)), amax(abs(h))) + abs(const), 0.1)

	if B is None: B = [1.0/high*(exp(x*1.0/num_cycles) - 1) for x in range(num_cycles)]
	if A is None: A = B[::-1]

	# Random initial state
	num_spins = pot_['num vars']
	spins = [random()*2.0*pi for i in range(num_spins)]

	for t in range(num_cycles):

		J_ = dot(J, cos(spins))
		new_spins = [random()*2.0*pi for i in range(num_spins)]
		for i in range(num_spins):

			# Compute the energy difference between the new state and the old state
			trans_field = A[t]*sin(new_spins[i]) - sin(spins[i])
			prob_ham = B[t]*((h[i] + J_[i])*cos(new_spins[i]) - cos(spins[i]) + const)

			# Do a metropolis update according to energy difference
			if random() <= 1.0/max(1.0, exp((trans_field + prob_ham)/temp)): spins[i] = new_spins[i]

	# Project onto computational basis
	if qubo: state = {-1: -1, 0: 1}
	else: state = {-1: -1, 0: 1}

	spins = [state[floor(cos(spins[i]))] for i in range(num_spins)]
	energy = array(h).dot(spins) + array(spins).dot(J.dot(array(spins).transpose())) + const

	return energy, spins


# Description: Simulated Annealing
# Inputs:	- pot: potential defining the energy of states... dictionary of tuple: strength
#			- states: the possible states each variable can be in... list of ints
#			- T: the temperature schedule... list of ints (length num_cycles)
#			- num_cycles: the number of iterations to run... int
# Outputs:	- the energy of the found state (the guess at the ground state energy)
#			- the corresponding state at that energy
#====================================================
def sa(pot, states, T = None, num_cycles = 5):

	J, const = dict_2_mat(pot)
	h = diag(J).copy()
	fill_diagonal(J, 0.0)

	if T is None: 
		T = range(num_cycles - 1, -1, -1) #T = [1.0/high*exp(x*1.0/num_cycles) for x in range(num_cycles)]
		T[-1] = 0.1

	# Random initial state
	num_vars = pot['num vars']
	state = choice(states, num_vars)

	# Anneal
	for temp in T:
		new_state = choice(states, num_vars)
		delta = new_state.dot(J.dot(new_state.transpose())) - state.dot(J.dot(state.transpose())) + h.dot(new_state) - h.dot(state)
		delta /= temp

		if delta <= 700:  # To avoid overflow errors
			if random() <= 1.0/max(1.0, exp(delta)): state = new_state.copy()


	return h.dot(state) + state.dot(J.dot(state.transpose())) + const, state



# Description: submit a potential to dwave to try to find a solution
# Inputs:	- pot: the potential function to be minimized... dictionary
#			- states: not currently used as it's assumed the states are [-1, 1] for this function
# Outputs:	- the minimum energy found
#			- the corresponding state
#=======================================
def dwave(pot, states):

	# Map problem into a dwave consumable format
	J_, const = dict_2_mat(pot)
	h_ = diag(J_).copy()
	fill_diagonal(J_, 0.0)
	j = mat_2_dict(J_)

	if 'num vars' in j: del j['num vars']
	if 'const' in j: del j['const']
	if 'mapping' in j: del j['mapping']

	prob_adj = j.keys()
	for k in range(pot['num vars']): prob_adj += [(k, k)]


	# Find an embedding and prepare to send physical problem to dwave
	embedding = find_embedding(prob_adj, adj).values()
	[h, J, chains, embedding] = embed_problem(h_, j, embedding, adj)	

	s = 0.8
	h = [a*s for a in h]
	for k in J: J[k] = J[k]*s
	for k in chains: 
		if k in J: J[k] += chains[k]
		else: J[k] = chains[k]


	# Submit problem
	print('submitting problem')
	submitted_problems = [async_solve_ising(solver, h, J, num_reads = 10000, num_spin_reversal_transforms = 100, answer_mode = 'histogram', auto_scale = True)]
	await_completion(submitted_problems, len(submitted_problems), float('inf'))
	res = unembed_answer(submitted_problems[0].result()['solutions'], embedding, 'discard')

	if len(res) == 0: return 10e3, None
	else: state = array(res[0])


	return h_.dot(state) + state.dot(J_.dot(state.transpose())) + const, state



def sqa(pot, states): pass














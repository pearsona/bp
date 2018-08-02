from numpy.random import random as rand
from numpy import dot, zeros, diag, fill_diagonal, cos, sin, array, amax
from math import exp, pi, floor
from itertools import product

from helper import *


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

	for state in product(states, repeat = pot['num vars']):
		state = array(state)
		energy = h.dot(state) + state.dot(J.dot(state.transpose())) + const

		if energy < min_so_far:
			min_so_far = energy
			best_state = state


	return min_so_far, best_state


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
	high = max(amax(abs(J)), amax(abs(h))) + abs(const)

	if B is None: B = [1.0/high*(exp(x*1.0/num_cycles) - 1) for x in range(num_cycles)]
	if A is None: A = B[::-1]

	# Random initial state
	num_spins = pot_['num vars']
	spins = [rand()*2.0*pi for i in range(num_spins)]

	for t in range(num_cycles):

		J_ = dot(J, cos(spins))
		new_spins = [rand()*2.0*pi for i in range(num_spins)]
		for i in range(num_spins):

			# Compute the energy difference between the new state and the old state
			trans_field = A[t]*sin(new_spins[i]) - sin(spins[i])
			prob_ham = B[t]*((h[i] + J_[i])*cos(new_spins[i]) - cos(spins[i]) + const)

			# Do a metropolis update according to energy difference
			if rand() <= 1.0/max(1.0, exp((trans_field + prob_ham)/temp)): spins[i] = new_spins[i]

	# Project onto computational basis
	if qubo: state = {-1: -1, 0: 1}
	else: state = {-1: -1, 0: 1}

	spins = [state[floor(cos(spins[i]))] for i in range(num_spins)]
	energy = array(h).dot(spins) + array(spins).dot(J.dot(array(spins).transpose()))

	return energy, spins





def sqa(pot, states): pass

def sa(pot, states): pass

def dwave(pot, states): pass




















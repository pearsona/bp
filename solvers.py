from numpy.random import random as rand
from numpy import dot, zeros, diag, fill_diagonal, cos, sin
from math import exp, pi
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

	min_so_far = 100000000000000.0

	if 'mapping' not in pot.keys(): map_vars(pot)
	mapping = pot['mapping']


	for state in product(states, repeat = pot['num vars']):
		energy = 0.0
		for variables in pot.keys():


			if type(variables) == type(""):
				if variables == 'const': energy += pot['const']
				else: continue
			else:

				var_state = pot[variables]
				if type(variables) == type(0): var_state *= state[mapping[variables]]

				else:
					for v in variables:
						var_state *= state[mapping[v]]
						if var_state == 0: break

				energy += var_state

		if energy < min_so_far:
			min_so_far = energy
			best_state = state


	return min_so_far, state


# Description: Spin Vector Monte Carlo, effectively simulated annealing with a phase component (arXiv: 1401.7087)
# Inputs:	- pot: the potential/hamiltonian other than the transverse field... dictionary
#			- states: which states the variables can be in (right now will convert to spin)... list
#			- temp: temperature to set the scale for the metropolis updates... float
#			- num_cycles: how many iterations/cycles to run for... int
#			- A: the transverse field schedule... list with an entry for each iteration/cycle
#			- B: problem hamiltonian schedule... list with an entry for each iteration/cycle
# Outputs:	- the answer according to svmc
#=====================
def svmc(pot_, states = [-1, 1], temp = 10.0, num_cycles = 10, A = None, B = None):

	if B is None: B = range(num_cycles)
	if A is None: A = B[::-1]

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


	# Random initial state
	num_spins = pot_['num vars']
	spins = [rand()*2.0*pi for i in range(num_spins)]

	for t in range(num_cycles):

		J_ = dot(J, cos(spins))
		new_spins = [rand()*2.0*pi for i in range(num_spins)]
		for i in range(num_spins):

			# Compute the energy difference between the new state and the old state
			trans_field = A[t]*sin(new_spins[i]) - sin(spins[i])
			prob_ham = h[i] + J_[i]
			prob_ham *= B[t]*cos(new_spins[i]) - cos(spins[i])
			prob_ham += B[t]*const

			# Do a metropolis update according to energy difference
			if rand() < 1.0/max(1.0, exp((trans_field + prob_ham)/temp)): spins[i] = new_spins[i]

	# Project onto computational basis
	spins = [cos(spins[i]) for i in range(num_spins)]


	if qubo:
		return [(s + 1)/2 for s in spins]
	else:
		return spins






def sqa(pot, states): pass

def sa(pot, states): pass

def dwave(pot, states): pass




















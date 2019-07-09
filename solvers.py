# Math
from numpy.random import random, choice, shuffle
from numpy import dot, zeros, diag, fill_diagonal, cos, sin, array, amax, argsort, shape, nonzero, append, linspace, mean
from math import exp, pi, floor
from itertools import product

# Plotting... not really necessary, but brute foce can plot the full spectrum if requested
import matplotlib.pyplot as plt

# Misc
from helper import *
from time import sleep
import datetime

# Dwave
from dwave_sapi2.remote import RemoteConnection
from dwave_sapi2.core import async_solve_ising, await_completion
from dwave_sapi2.util import get_hardware_adjacency
from dwave_sapi2.embedding import embed_problem, unembed_answer
from minorminer import find_embedding
import dwave_networkx as dnx




#===============
#  DWAVE INFO  #
#===============

solver_name = 'C16'#'DW2X'
url = 'https://qfe.nas.nasa.gov/sapi'#'https://usci.qcc.isi.edu/sapi'
token = 'NASA-f73f6a756b922f9ebfcb6127740bec11bf986527'#'RnD-27a8a3832f9f6f8aa1b74c192a3098802c91c7a1'
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
	print('Connected to DWave')



#==========
# Solvers #
#==========

# Description: Brute force scan over all solutions
# Inputs:	- pot: dictionary describing potential and solver
#			- states: possible states for each variable to be in
#			- show_spec: whether to plot the full spectrum (requires matplotlib)
# Outputs:	- energy of solution and the solution itself 
# Description: Search over all of state space and find the minimum
#=============================
def brute_force(pot, states, show_spec = False):

	if pot['num vars'] > 0:

		J, const = dict_2_mat(pot)
		h = diag(J).copy()
		fill_diagonal(J, 0.0)

			
		if len(h) == 1:

			best = 1e9
			state = [0]
			for s in states:
				if h[0]*s < best:
					best = h[0]*s
					state = s

			return best, [state]

		else:

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

			#print(pot)

			# Reorder according to the mapping constructed in dict_2_mat
			#estate = [state[pot['mapping'][v]] for v in pot['mapping']]

			return res[inds[0]], state

	else:
		if 'const' in pot: return pot['const'], states[0]
		else: return 0, states[0]



# Description: Spin Vector Monte Carlo, effectively simulated annealing with a phase component (arXiv: 1401.7087)
# Inputs:	- pot: the potential/hamiltonian other than the transverse field... dictionary
#			- states: which states the variables can be in (right now will convert to spin)... list
#			- temp: temperature to set the scale for the metropolis updates... float
#			- num_cycles: how many iterations/cycles to run for... int
#			- A: the transverse field schedule... list with an entry for each iteration/cycle
#			- B: problem hamiltonian schedule... list with an entry for each iteration/cycle
# Outputs:	- the answer according to svmc
#=====================
def svmc(pot_, states = [-1, 1], temp = 1.0, num_cycles = 100, num_moves = 1, num_samples = 1000, A = None, B = None):

	if pot_['num vars'] > 0:


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

		if len(h) == 1:
			
			best = 1e9
			state = [0]
			for s in states:
				if h[0]*s < best:
					best = h[0]*s
					state = s

			return best, [state]


		else:
			upper_bound = sum(abs(h)) + sum(sum(abs(J)))



		if B is None: B = linspace(0, 1, num = num_cycles)
		if A is None: A = B[::-1]*upper_bound # this should put them on roughly equal scales

		
		num_vars = pot_['num vars']

		best = 1e9
		solution = []
		new_states = 2.0*pi*random((num_samples, num_cycles, num_moves, num_vars)) # randomly draw all states to be used in the following runs

		# How many samples (i.e. runs of SVMC) to consider
		for sample in range(num_samples):
			# The initial guess for this run of SVMC
			state = [random()*2.0*pi for i in range(num_vars)]
			updates = range(num_vars)
			
			# A round of SVMC that has a schedule broken into num_cycles points
			for t in range(num_cycles):
				# A thermalization of sorts at this point in the schedule
				for move in range(num_moves):
					# Pick a random order of checking if we want to change a variable state
					shuffle(updates)
					for spin in updates:

						# Calculate the change in "transverse field" and problem hamiltonian components at this point in the schedule
						trans_field = A[t]*(sin(new_states[sample][t][move][spin]) - sin(state[spin]))
						prob_ham = B[t]*(cos(new_states[sample][t][move][spin]) - cos(state[spin])*(h[spin] + J[spin, :].dot(cos(state))))
						delta = (trans_field + prob_ham)/temp

						# DMetropolis update
						if delta >= -700: #avoid overflow errors
							if random() <= min(1.0, exp(-delta)): state[spin] = new_states[sample][t][move][spin]
						else: state[spin] = new_states[sample][t][move][spin]


			# Project onto computational basis and compute the energy of this state
			if qubo: proj = {-1: 0, 0: 1}
			else: proj = {-1: -1, 0: 1}
			state = [proj[floor(cos(state[i]))] for i in range(num_vars)]
			energy = array(h).dot(state) + array(state).dot(J.dot(array(state).transpose())) + const

			# Determine if this sample is better than the previous ones
			if energy < best:
				best = energy
				solution = state


		return best, solution
	
	else:
		if 'const' in pot_: return pot_['const'], states[0]
		else: return 0, states[0]


# Description: Simulated Annealing
# Inputs:	- pot: potential defining the energy of states... dictionary of tuple: strength
#			- states: the possible states each variable can be in... list of ints
#			- T: the temperature schedule... list of ints (length num_cycles)
#			- num_cycles: the number of iterations to run... int
#			- num_moves: the number of moves to make at each temperature/cycle
# Outputs:	- the energy of the found state (the guess at the ground state energy)
#			- the corresponding state at that energy
#====================================================
def sa(pot, states, T = None, num_cycles = 10, num_moves = 100, num_samples = 10):

	if pot['num vars'] > 0:

		J, const = dict_2_mat(pot)
		h = diag(J).copy()
		fill_diagonal(J, 0.0)

		if T is None: 
			T = range(num_cycles - 1, -1, -1) #T = [1.0/high*exp(x*1.0/num_cycles) for x in range(num_cycles)]
			T[-1] = 0.1

		# Random initial state
		num_vars = pot['num vars']
		updates = range(num_vars)

		best = 1e9
		solution = []

		# How many samples (i.e. runs of SA) to consider
		for sample in range(num_samples):

			# Random initial state
			state = choice(states, num_vars)
			
			# Anneal
			for temp in T:
				# Thermalize at this temperature
				for move in range(num_moves):
					shuffle(updates)
					# Update each spin state
					for spin in updates:

						delta = -2*state[spin]*(h[spin] + J[spin, :].dot(state))/temp

						# Metropolis update
						if delta >= -700: #avoid overflow errors
							if random() <= min(1.0, exp(-delta)): state[spin] = -state[spin]
						else: state[spin] = -state[spin]


			# Compute the energy of this sample and keep if it's better than what we've seen so far
		 	energy = h.dot(state) + state.dot(J.dot(state.transpose())) + const
		 	if energy < best:
		 		best = energy
		 		solution = state


		return best, solution

	else:
		if 'const' in pot: return pot['const'], states[0]
		else: return 0, states[0]



# Description: submit a potential to dwave to try to find a solution
# Inputs:	- pot: the potential function to be minimized... dictionary
#			- states: not currently used as it's assumed the states are [-1, 1] for this function
# Outputs:	- the minimum energy found
#			- the corresponding state
#=======================================
def dwave(pot, states):

	if pot['num vars'] > 0:

		solved = False
		const = 0
		h_ = []
		J_ = {}
		state = []
		free_state = []
		embedding = []

		while not solved:
			try:
				#global solver
				#global adj

				#if solver == 0: sign_in() #should try to make it so there is a pool of pool_size connections that the various threads can use
				remote_connection = RemoteConnection(url, token)
				solver = remote_connection.get_solver(solver_name)
				adj = list(get_hardware_adjacency(solver))


							
				if 'embedding' in pot: 
					const, h_, j, prob_adj = dwave_prepare(pot)
					embedding = pot['embedding']
				else:
					
					# if we're doing a new embedding for each f -> v in state i message, then we'll have frozen a variable
					# so we need to remap the variables, since otherwise the h will have a 0 for this variable, but the embedding won't consider it
					map_vars(pot)
					const, h_, j, prob_adj = dwave_prepare(pot)

					while len(embedding) == 0:
						embedding = find_embedding(prob_adj, adj).values()
					
					
				
				[h, J, chains, embedding] = embed_problem(h_, j, embedding, adj)

				s = 0.50
				h = [a*s for a in h]
				for k in J: J[k] = J[k]*s
				for k in chains:
					if k in J: J[k] += chains[k]
					else: J[k] = chains[k]


				# Submit problem
				#print('submitting problem')

				
				submitted_problems = [async_solve_ising(solver, h, J, num_reads = 10000, num_spin_reversal_transforms = 5, answer_mode = 'histogram', auto_scale = True)]
				await_completion(submitted_problems, len(submitted_problems), float('180'))
				res = unembed_answer(submitted_problems[0].result()['solutions'], embedding, 'discard')

				if len(res) > 0:
					state = array(res[0])
					solved = True

			except Exception as err:
				print(err)
				solved = False
				#sleep(30) # wait 30 seconds and retry


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

	else:

		if 'const' in pot: return pot['const'], states[0]
		else: return 0, states[0]


def dwave_embed(pot, overkill = True):

	#global solver
	#global adj

	#if solver == 0: sign_in()

	remote_connection = RemoteConnection(url, token)
	solver = remote_connection.get_solver(solver_name)
	adj = list(get_hardware_adjacency(solver))

	#print('Connecting to DWave')
	#remote_connection = RemoteConnection('https://qfe.nas.nasa.gov/sapi', 'NASA-f73f6a756b922f9ebfcb6127740bec11bf986527')
	#solver = remote_connection.get_solver('C16')
	#adj = list(get_hardware_adjacency(solver))

	const, h_, j, prob_adj = dwave_prepare(pot)

	embedding = []
	if overkill:

		emb = {}
		beta = 2
		max_length = 10e9
		try:
			
			while emb == {} or max_length > 4:

				for i in range(3):

					emb_ = find_embedding(prob_adj, adj, max_beta = beta)

					# only take this embedding if it has a shorter max length
					if emb_ != {} and max([len(c) for c in emb_.values()]) < max_length: 
						emb = emb_.copy()
						max_length = max([len(c) for c in emb.values()])

					if max_length < 4: break

				if beta > 64:
					emb_ = find_embedding(prob_adj, adj, tries = 100)

					if emb_ != {} and max([len(c) for c in emb_.values()]) < max_length: 
						emb = emb_.copy()
						max_length = max([len(c) for c in emb.values()])

					break

				beta = beta*2
				

		except RuntimeError as err:
			print(err)
			emb = find_embedding(prob_adj, adj)


		if emb == {}:
			print('Unable to find embedding for problem')
			return [False]*4
		else: print('Found an embedding')

		embedding = emb.values()

	else:
		
		while len(embedding) == 0:
			embedding = find_embedding(prob_adj, adj).values()




	remote_connection = 0
	solver = 0
	adj = 0

	return embedding




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





# # Taken from Arxiv 1808.01275
# This should really be used for certification of optimality of a good solution, not as a solver
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


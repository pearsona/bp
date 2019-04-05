from bp import *

from numpy import vstack, array
from numpy.random import randint, permutation
from numpy.linalg import norm

from scipy.io import loadmat
from os import listdir
from os.path import isfile, join
from fnmatch import fnmatch

import re
from pickle import dump, load



# LDPC as Factor Graphs
# This code can create instances of low density parity check code decoding (i.e. the check matrix and the syndrome) and turn such an instance into a factor graph (single check per factor)
#=============================



# Description: Run BP on the instances found in the given folder
# Inputs:	- folder: name of the folder to pull instances from... string
#			- solver: which solver to use... function
#			- pool_size: how many threads to start (1 by default, so no concurrency)
#			- verbose: whether to print what's happening or not... bool
# Outputs:	- the solutions found for each instance
#===================================================
def runBP(folder = '../insts', solver = sa, pool_size = 1, verbose = True):

	sols = []
	dist = []
	if verbose: print('Loading instances from: ' + folder)
	problems = load_problems(folder)

	for (fn, problem) in problems:
		graph = problem['graph']

		if verbose: print('Running: ' + fn)
		[bits, rgns, errs] = re.split('bits|rgns|error', fn)[:-1]

		jump_in = False
		fn_ = ''
		for file in listdir('../results'):
			if fnmatch(file, bits + 'bits' + rgns + 'rgns' + errs + 'error_solver=' + solver.__name__ + '*.pickle'):
				jump_in = True
				fn_ = file

		if not jump_in:

			#graph = create_factor_graph(inst['H'], array(inst['y1'].transpose(), dtype = int))#array(inst['H'].dot(inst['y1'].transpose()) % 2, dtype = int))
			graph.solver = solver

			#if num_regions != False: graph = graph.regionalize(recurs = False, num_regions = num_regions if num_regions != -1 else int(rgns))

			#if verbose: print('Created the Factor Graph with (# variables, # factors): ' + str((graph.num_variables, graph.num_factors)))
			sol, num_iters = min_sum_BP(graph, solver, verbose = verbose, pool_size = pool_size)

		else:

			f = open('../results/' + fn)
			x = load(f)
			f.close()

			if verbose: print('Jumping back into this factor graph at iteration: ' + str(x['iterations']))

			graph = x['graph']
			sol, num_iters = min_sum_BP(graph, solver, verbose = verbose, pool_size = pool_size, jump_in = True, max_iter = 1000 - x['iterations'])
			num_iters += x['iterations']

		sol = array(spin_2_bin(sol))
		dist = sum((sol - problem['solution'][0]) % 2)
		save(bits + 'bits' + rgns + 'rgns' + errs + 'error_solver=' + solver.__name__, {'solution': sol, 'true_solution': problem['solution'][0], 'dist': dist, 'graph': graph, 'iterations': num_iters})




# Description: Create a factor for each parity check in the given matrix with an option to bias bits (initialize messages) according to the syndrome
# Inputs:	- H: parity check matrix... np matrix
#			- s: syndrome... np array of -1, 0, +1 (which ones depending on binary or spin representation)
# Outputs:	the factor graph for the instance
# NOTE: the 3 bit parity check potential is the K3,3 from https://doi.org/10.3389/fphy.2014.00056
#=============================
def create_factor_graph(H, s = None, ising = True):

	g = FactorGraph()

	# add in each of the variables involved in the parity checks and initialize according to syndrome
	g.add_variables(len(s), s)

	# Add each check to the factor graph
	for check in H:
		bits = []

		# this could be done with numpy.nonzero(check)
		for bit in range(len(check)):
			if check[bit] == 1: bits += [bit]

		potential, ancs = parity_check_ham(bits, g.num_variables)
		g.add_factor(potential, ancillas = ancs)

	if not ising: g.ising_2_qubo()

	return g



def parity_check_ham(bits, anc_start):
	
	potential = {'num vars':  0}
	ancs = []

	# 2 bits just need to agree
	if len(bits) == 2:
		potential[bits[0], bits[1]] = -1
		potential['num vars'] = 2

	elif len(bits) == 3:
		# Ancilla bit variable number assignments
		ancs = [anc_start, anc_start + 1, anc_start + 2]

		# Create k3,3 potential
		potential[ancs[0]] = -1
		potential[ancs[1]] = 1
		potential[ancs[2]] = -1

		potential[bits[0], ancs[0]] = 1
		potential[bits[0], ancs[1]] = 1
		potential[bits[0], ancs[2]] = 1

		potential[bits[1], ancs[0]] = -1
		potential[bits[1], ancs[1]] = 1
		potential[bits[1], ancs[2]] = 1

		potential[bits[2], ancs[0]] = 1
		potential[bits[2], ancs[1]] = 1
		potential[bits[2], ancs[2]] = -1

		potential['num vars'] = 6


	elif len(bits) == 4:#% 4 == 0: 

		# break each 4 into 2 sets of 3-bit parity checks on 2 decision variables and a shared ancilla
		# a + b + c + d = (a + b + x) + (c + d + x) (mod 2)
		num_checks = len(bits)/2
		ancs = [anc_start + i for i in range(num_checks/2)]

		for a in range(num_checks):
			pot, anc = parity_check_ham(bits[2*a: 2*a + 2] + ancs[int(a/2) : int(a/2) + 1], anc_start + len(ancs))
			ancs += anc
			potential.update(pot)

	elif len(bits) == 5:#% 5 == 0:

		# break into 3 sets of 3-bit parity checks on 2 decision variables and 2 shared ancillas
		# a + b + c + d + e= (a + b + x) + (c + d + y) + (e + x + y) (mod 2)
		ancs = [anc_start, anc_start + 1]

		pot, anc = parity_check_ham(bits[:2] + [anc_start], anc_start + 2)
		ancs += anc
		potential.update(pot)

		pot, anc = parity_check_ham(bits[2:4] + [anc_start + 1], anc_start + len(ancs))
		ancs += anc
		potential.update(pot)

		pot, anc = parity_check_ham(bits[4:] + [anc_start, anc_start + 1], anc_start + len(ancs))
		ancs += anc
		potential.update(pot)

	else:
		print('This size (' + str(len(bits)) + ') parity check is not currently supported')
		return False



	get_num_vars(potential)

	return potential, ancs


def test_checks(num_bits):

	bits = range(num_bits)
	pot, ancs = parity_check_ham(bits, len(bits))

	J, const = dict_2_mat(pot)
	h = diag(J).copy()
	fill_diagonal(J, 0.0)

	res = []
	for st in product([-1, 1], repeat = pot['num vars']):
		st = array(st)
		res += [h.dot(st) + st.dot(J.dot(st.transpose())) + const]

	res = array(res)
	inds = argsort(res)
	lowest = min(res)

	#print(res[inds][:16])

	states = []
	for i in range(len(res)):
		if res[inds[i]] == lowest:
			r = inds[i]
			state = [0 for j in range(pot['num vars'])]
			for j in range(len(state))[::-1]:
				state[j] = r % 2
				r = int(r/2)

			state = [state[pot['mapping'][j]] for j in range(len(state))]
			states += [state[:num_bits]]
		else: break


	return states, [sum(states[i]) % 2 for i in range(len(states))]

# Description: load instances from the given folder
# Inputs:	- folder: name of the folder to pull instances from... string
# Outputs:	- the instances... a list of dictionaries where each entry has H, G, y, y1, and file name
#===================================================
def load_insts(folder):

	insts = []

	for f in listdir(folder):
		if isfile(join(folder, f)) and '.mat' in f:
			inst = loadmat(join(folder, f))
			insts += [{'H': inst['H'], 'G': inst['G'], 'y': inst['y'], 'y1': inst['y1'], 'name': f}]

	return insts


def load_problems(folder):

	graphs = []

	for fn in listdir(folder):
		if isfile(join(folder, fn)) and 'graph.pickle' in fn:
			f = open(join(folder, fn))
			graphs += [(fn, load(f))]
			f.close()

	return graphs


def create_graphs():

	insts = load_insts('../insts')

	for inst in insts:

		print('Running: ' + inst['name'])
		[bits, rgns, errs] = re.split('bits|rgns|error', inst['name'])[:-1]

		graph = create_factor_graph(inst['H'], bin_2_spin(list(inst['y1'][0])))#array(inst['H'].dot(inst['y1'].transpose()) % 2, dtype = int))
		graph = graph.regionalize(recurs = False, num_regions = int(rgns))

		print('Created the Factor Graph with (# variables, # factors): ' + str((graph.num_variables, graph.num_factors)))

		f = open('../insts/' + bits + 'bits' + rgns + 'rgns' + errs + 'error_graph.pickle', 'wb')
		dump({'graph': graph, 'solution': inst['y'][0], 'input': inst['y1'][0]}, f)
		f.close()


def save(name, data):

    i = 0
    while True:
        if isfile('../results/' + name + '_' + str(i) + '.pickle'):
            i += 1
        else:
            f = open('../results/' + name + '_' + str(i) + '.pickle', 'wb')
            break

    dump(data, f)
    f.close()



# # Description: Create a [n, k] LDPC decoding instance (i.e. PC matrix and transmission)
# # Inputs:	- error: the probability of a bit flip... float in [0, 1]
# #			- num_bits: the total number of bits... int  (n + k)
# #			- num_checks: the total number of checks... int (k)
# #			- max_bits: maximum number of bits involved in a check... int
# #			- H: PC matrix... np matrix
# # Outputs:	- the PC matrix and the transmission (z = Gm + e)
# #=============================
# def create_instance(error = 0.1, num_bits = 4, num_checks = 2, max_bits = 3, H = []):

# 	# Create the PC and generator matrix if not given
# 	if H == []:

# 		check = randint(0, 2, num_bits)
# 		H = check
# 		while len(H) < num_checks: H = vstack((H, permutation(check)))

# 	else: num_checks, num_bits = H.shape


# 	# Create the transmission (z = Gm + e... HGm = 0 => Gm in null space of H... z = null_space vector + error)
# 	return H, nullspace(H)[0] + binomial(1, error, num_bits)


# # from https://scipy-cookbook.readthedocs.io/items/RankNullspace.html
# def nullspace(A):

# 	_, s, vh = svd(A)
#     nnz = (s > 0).sum()

#     return vh[nnz:].conj()


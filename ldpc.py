from bp import *

from numpy import vstack, array
from numpy.random import randint, permutation
from numpy.linalg import norm

from scipy.io import loadmat
from os import listdir
from os.path import isfile, join

# LDPC as Factor Graphs
# This code can create instances of low density parity check code decoding (i.e. the check matrix and the syndrome) and turn such an instance into a factor graph (single check per factor)
#=============================



# Description: Run BP on the instances found in the given folder
# Inputs:	- folder: name of the folder to pull instances from... string
#			- solver: which solver to use... function
#			- verbose: whether to print what's happening or not... bool
# Outputs:	- the solutions found for each instance
#===================================================
def runBP(folder = 'insts', solver = sa, verbose = True, regionalize = True, pool_size = 1):

	sols = []
	dist = []
	if verbose: print('Loading instances from: ' + folder)
	insts = load_insts(folder)

	for inst in insts:
		if verbose: print('Running: ' + inst['name'])

		graph = create_factor_graph(inst['H'], array(inst['y1'].transpose(), dtype = int))#array(inst['H'].dot(inst['y1'].transpose()) % 2, dtype = int))
		graph.solver = solver

		if regionalize: graph = graph.regionalize(recurs = False, frac_regions = 0.2)

		if verbose: print('Created the Factor Graph with (# variables, # factors): ' + str((graph.num_variables, graph.num_factors)))
		sol, _ = min_sum_BP(graph, solver, verbose = verbose, pool_size = pool_size)
		sols += [sol]
		dist += [norm(array(sol) - inst['y'])]

	return sols, dist





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
		potential[bits[0], bits[1]] = 1
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
	else: 

		# break into sets of 3-bit parity checks on 2 decision variables and an ancilla 
		num_checks = int(len(bits)/2)
		ancs = [anc_start + i for i in range(num_checks)]

		for a in range(num_checks):
			pot, anc = parity_check_ham(bits[2*a: 2*a + 2] + ancs[a:a+1], anc_start + num_checks + 3*a)
			ancs += anc

			num = potential['num vars'] + pot['num vars']
			potential.update(pot)
			potential['num vars'] = num


		# if the number of bits is odd, we need an extra check for the last variable
		if len(bits) % 2 != 0:
			pot, anc = parity_check_ham([bits[-1], ancs[-2], ancs[-1]], ancs[-1] + 1)
			ancs += anc

			num = potential['num vars'] + pot['num vars']
			potential.update(pot)
			potential['num vars'] = num

	return potential, ancs


# Description: load instances from the given folder
# Inputs:	- folder: name of the folder to pull instances from... string
# Outputs:	- the instances... a list of dictionaries where each entry has H, G, y, y1, and file name
#===================================================
def load_insts(folder):

	insts = []

	for f in listdir(folder):
		if isfile(join(folder, f)):
			inst = loadmat(join(folder, f))
			insts += [{'H': inst['H'], 'G': inst['G'], 'y': inst['y'], 'y1': inst['y1'], 'name': f}]

	return insts




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


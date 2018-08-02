from FactorGraph import *
from solvers import *


#=====================
# Running min-sum BP #
#=====================

# Description: run min-sum BP on the graph given (or from the default if none is given)
# Inputs:	- g: graph to run BP on... FactorGraph (defined above)
#			- solver: which solver to use for updates
#			- init_mess: messages to initialize each node with... dictionary (key = node sending message) of dictionaries (key = node receiving message) of dictionaries (keys = states)... (sorry)
#			- max_iter: maximum number of iterations before termination... int
#			- verbose: print everything or nothing... bool
# Output:	- answer from BP and number of iterations taken to find it
#=====================
def min_sum_BP(g = None, solver = brute_force, init_mess = {}, max_iter = 1000, verbose = False):

	# A default example meant to illustrate the various capabilities and parameters of a factor graph with solution = [1,1,1]
	if g is None:
		g = FactorGraph(solv = solver, states = [0, 1], threshold = 0.1, normalize = False)

		g.add_factor(0, {'const': -1.0, 0: -1.0, 1: -1.0, (0, 1): -1.0, 'num vars': 2})
		g.add_factor(1, {'const': -2.0, 1: 1.0, (1, 2): -3.0, 'num vars': 2})

		g.qubo_2_ising()

		verbose and print('created an ising factor graph '  + str(list(g)))


	g.initialize(init_mess)
	state = []
	verbose and print('initialized')

	# Iteratively run BP
	for i in range(max_iter):

		try:

			verbose and print('\n\niteration: ' + str(i))

			verbose and print('\n1) sending factor -> variable messages')
			g.all_factors_to_variables(verbose)

			verbose and print('\n2) updating beliefs')
			changed = g.update_all_beliefs(verbose)

			if not changed:
				verbose and print('\nWe seem to have found the answer in just ' + str(i + 1) + '/' + str(max_iter) + ' runs')
				return g.get_best_state(), i + 1


			verbose and print('\n3) sending variable -> factor messages')
			g.all_variables_to_factors(verbose)

			old_state = list(state)
			state = g.get_best_state()
			verbose and print('\ncurrent best state: ' + str(state))

		except KeyboardInterrupt:

			state = g.get_best_state()
			verbose and print('\n\nexiting... current best state:' + str(state))
			return state, i + 1


	verbose and print('We ran through all ' + str(max_iter) + ' runs and got the answer: ' + str(state))
	return state, max_iter


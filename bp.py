from FactorGraph import *
from solvers import *

import datetime

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
def min_sum_BP(g = None, solv = brute_force, init_mess = {}, max_iter = 1000, verbose = False, pool_size = 1, jump_in = False):

	# A default example meant to illustrate the various capabilities and parameters of a factor graph with solution = [1,1,1]
	if g is None:
		g = FactorGraph(solver = solv, states = [0, 1], threshold = 0.1)

		#g.add_variables(3, [0, 0, 0]) # to test starting with bad initial conditions

		g.add_factor({'const': -1.0, 0: -1.0, 1: -1.0, (0, 1): -1.0, 'num vars': 2})
		g.add_factor({'const': -2.0, 1: 1.0, (1, 2): -3.0, 'num vars': 2})

		#g.qubo_2_ising()

		if verbose: print('created an ising factor graph '  + str(list(g)))

	if not jump_in:

		g.initialize(init_mess)
		state = g.get_best_state()
		if verbose: print('initialized')

		num_iters = 0
	else:

		state = g.get_best_state()

	fixed_state = 0	



	# Iteratively run BP
	for i in range(max_iter):

		num_iters = i + 1
		try:

			print('\n\n' + str(datetime.datetime.now()) + '\niteration: ' + str(i))

			if verbose: print('\n' + str(datetime.datetime.now()) + '\n1) sending factor -> variable messages')
			g.all_factors_to_variables(verbose, pool_size)
			#raw_input('Press enter to continue: ')

			if verbose: print('\n' + str(datetime.datetime.now()) + '\n2) updating beliefs')
			num = g.update_all_beliefs(verbose, pool_size)
			#raw_input('Press enter to continue: ')

			if num == 0 and i > 0:
				if verbose: print('\nWe seem to have found the answer in just ' + str(i + 1) + '/' + str(max_iter) + ' runs')
				break


			if verbose: print('\n' + str(datetime.datetime.now()) + '\n3) sending variable -> factor messages')
			g.all_variables_to_factors(verbose, pool_size)
			#raw_input('Press enter to continue: ')

			old_state = list(state)
			state = g.get_best_state()
			if verbose: print('\n' + str(datetime.datetime.now()) + '\ncurrent best state: ' + str(state))

			# If the state hasn't changed in 10 iterations, it seems reasonable to exit
			if old_state == state: fixed_state += 1
			else: fixed_state = 0

			if fixed_state == 10: break

		except KeyboardInterrupt as err:

			print(err)

			state = g.get_best_state()
			if verbose: print('\n\n' + str(datetime.datetime.now()) + '\nExiting early at ' + str(i + 1), ' iterations... current best state:' + str(state))
			break

	if verbose: print('\n\n' + str(datetime.datetime.now()) + '\nFinished BP, will now use these results to determine a solution (will require solving each factor once more)...\n')

	return g.finish(pool_size), num_iters


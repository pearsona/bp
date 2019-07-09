# BP Stuff
from FactorGraph import *
from solvers import *

# Used for formatting the output messages
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
#			- pool_size: how many processes to use (default = 1 means no multiprocessing)... int
#			- jump_in: whether or not we are jumping into a previously started run... bool
# Output:	- answer from BP and number of iterations taken to find it
# A nice reference:	https://pdfs.semanticscholar.org/9e00/32f75cc57a8b7b23255964561fab480b7a8b.pdf?_ga=2.252865417.1669990439.1554775842-1496842273.1554775842
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


	# If we're not jumping into a previous run, then we need to initialize the graph and considering the initial messages (i.e. the noisy input)
	if not jump_in:

		g.initialize(init_mess, pool_size = pool_size)
		num_iters = 0
		if verbose: print('initialized')
		

	# Get the initial guess (random if not jumping in and the guess of the previous run otherwise)
	state = g.get_best_state()

	fixed_state = 0	
	g.solver = solv


	# Iteratively run BP
	for i in range(max_iter):

		num_iters = i + 1
		try:


			# Sending factor to variable messages
			if verbose: print('\n\n' + str(datetime.datetime.now()) + '\niteration: ' + str(i))

			if verbose: print('\n' + str(datetime.datetime.now()) + '\n1) sending factor -> variable messages')

			g.all_factors_to_variables(verbose, pool_size)
			#raw_input('Press enter to continue: ')



			# Updating beliefs
			# Stop iterating if no beliefs changed (and we've done at least 1 iteration)
			if verbose: print('\n' + str(datetime.datetime.now()) + '\n2) updating beliefs')
			num = g.update_all_beliefs(verbose, pool_size)
			#raw_input('Press enter to continue: ')

			if num == 0 and i > 0:
				if verbose: print('\nWe seem to have found the answer in just ' + str(i + 1) + '/' + str(max_iter) + ' runs')
				break


			# Sending variable to factor messages
			if verbose: print('\n' + str(datetime.datetime.now()) + '\n3) sending variable -> factor messages')
			g.all_variables_to_factors(verbose, pool_size)
			#raw_input('Press enter to continue: ')


			# Determine the best guess after this round of message passing
			old_state = list(state)
			state = g.get_best_state()
			if verbose: print('\n' + str(datetime.datetime.now()) + '\ncurrent best state: ' + str(state))


			# If the state hasn't changed in 10 iterations, it seems reasonable to exit
			# This was a criterion used in the 2014 work that is not strictly theoretically sound, but seems reasonable (particularly for loopy BP)
			if old_state == state: fixed_state += 1
			else: fixed_state = 0

			if verbose: print('\nWe have had this state for ' + str(fixed_state) + ' iterations')

			if fixed_state == 10: break

		# Allow us to exit BP in the middle, but save our place (have had some issues with this and multiprocessing)
		except (KeyboardInterrupt, TimeoutError) as err:

			print(err)

			state = g.get_best_state()
			if verbose: print('\n\n' + str(datetime.datetime.now()) + '\nExiting early at ' + str(i + 1), ' iterations... current best state:' + str(state))
			break

	if verbose: print('\n\n' + str(datetime.datetime.now()) + '\nFinished BP, will now use these results to determine a solution (will require solving each factor once more)...\n')



	# Call the finish function, which solves each region internally and returns a solution
	return g.finish(pool_size), num_iters


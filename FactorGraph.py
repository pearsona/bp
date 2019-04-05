import networkx as nx
from metis import *

from solvers import *
from helper import *

#from multiprocessing.pool import ThreadPool as Pool
from multiprocessing.pool import Pool
import copy_reg
import types

import heapq

from dwave_sapi2.fix_variables import fix_variables

import os
import psutil
import signal

parent_id = os.getpid()

def worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# def worker_init():
#     def sig_int(signal_num, frame):
#         print('signal: %s' % signal_num)
#         parent = psutil.Process(parent_id)
#         for child in parent.children():
#             if child.pid != os.getpid():
#                 print("killing child: %s" % child.pid)
#                 child.kill()
#         print("killing parent: %s" % parent_id)
#         parent.kill()
#         print("killed: %s" % os.getpid())
#         psutil.Process(os.getpid()).kill()
#     signal.signal(signal.SIGINT, sig_int)


def _pickle_method(method):

	func_name = method.im_func.__name__
	obj = method.im_self
	cls = method.im_class

	return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
	
	for cls in cls.mro():
		try: func = cls.__dict__[func_name]
		except KeyError: pass
		else: break

	return func.__get__(obj, cls)


copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


# Description: Create a factor graph particularly built for min-sum BP using a given solver
#
#	Instance Variables:
#				- num_factors, num_variables: ints to keep count of each
#				- solver: function that will be used to 'solve' (may be heuristic) a given potential dictionary
#				- normalize: boolean determining if the messages and beliefs are to be normalized
#				- states: list of possible states of the variables (e.g. [1, -1], [0, 1], [-1, 0, 1], etc.)
#				- threshold: how much beliefs have to change in order to actually treat them as changed... float
#
# 	Factors: These are nodes labelled 'fx' where x is the factor number, with fields:
#				- pot: a dictionary describing the potential function of the given factor (including how many variables involved)
#					   e.g. {1: 2, (1, 2): 4, 'const': 3, 'num vars': 2} => h[1] = 2, h[2] = 0, J[1, 2] = 4, constant = 3
#				- mess: a dictionary of dictionaries describing the messages TO each variable in its scope
#					   e.g. {1: {1: 1, -1: 1}, 3: {1: 1, -1: 2} => variable 1 equally +/- 1, variable 2 not in scope, variable 3 is +1 w.p. 1/3 and -1 w.p. 2/3
#
#
#	Variables: These are nodes labelled 'vx' where x is the variable number, with fields:
#				- mess: a dictionary of dictionaries in the same form as factors (i.e. TO each factor NOT FROM each factor)
#				- bel: a dictionary describing the current beliefs of this variable
#
#=============================
class FactorGraph(nx.Graph):


	# Description: Constructor for a Factor Graph
	# Inputs:	- solv: which solver to used... function
	#			- states: the possible states each variable can be in... list
	#			- normalize: whether to normalize messages and beliefs... bool
	#			- threshold: how much beliefs have to change in order to actually treat them as changed... float
	# Outputs: none
	#=============================
	def __init__(self, solver = brute_force, states = [-1, 1], normalize = True, threshold = 0.5):
		super(FactorGraph, self).__init__()

		self.num_factors = 0
		self.num_variables = 0
		self.ancillas = []

		self.solver = solver
		self.norm = normalize
		self.states = states
		self.threshold = threshold



	#=====================
	# Graph Construction #
	#=====================


	# Description: Add a factor node to the graph with the given scope or with a scope determined by the 
	#			   potential and add all the necessary variables and edges
	# Inputs:	- num = number to label node... int
	# 			- potential = potential function for this factor... dictionary
	#			- scope = variable nodes connected to this factor graph... list of ints
	#			- messages = messages for each variable in scope... dictionary
	#			- ancillas = which of these variables are ancillas... list
	# Outputs:	none
	#=============================
	def add_factor(self, potential, num = -1, scope = None, ancillas = []):

		#if num < self.num_factors: return 'factor already in graph'
		if num == -1: num = self.num_factors
		self.num_factors += 1
		self.add_node('f' + str(num), pot = potential, mess = {})

		# add edges to each node in the scope
		if scope == None:
			scope = []
			for k in potential.keys():
				if type(k) == type(0):
					if k not in scope: scope += [k]
				elif k not in ['const', 'num vars']:
					for x in k:
						if x not in scope: scope += [x]

		for k in scope:
			if ('v' + str(k)) not in self: self.add_variable(num = k, ancilla = (k in ancillas))
			self.add_edge('f' + str(num), 'v' + str(k))

		if 'num vars' not in self.nodes['f' + str(num)]['pot']: self.nodes['f' + str(num)]['pot']['num vars'] = len(scope)
		if 'const' not in self.nodes['f' + str(num)]['pot']: self.nodes['f' + str(num)]['pot']['const'] = 0.0



	# Description: Add a variable node to the graph
	# Inputs:	- num = number to label node... int
	#			- init_state = what to initialize messages according to (effectively a guess)... element of self.states
	#			- ancilla = if this variable is ancillary (so won't be relevant to output/answer)
	# Outputs:	none or false if the variable has already been added
	#=============================
	def add_variable(self, init_state = None, num = -1, ancilla = False):

		#if num < self.num_variables: return 'variable already in graph'

		#if init_state is None: init_state = self.states[0]
		if num == -1: num = self.num_variables
		self.num_variables += 1
		self.add_node('v' + str(num), mess = {}, bel = {}, init = init_state, done = False)

		if ancilla: self.ancillas += ['v' + str(num)]
		


	# Description: Add a set of variable nodes to the graph
	# Inputs:	- num_vars = number of variables to add... int
	#			- init_states = what to initialize messages according to (effectively a guess)... list of elements of self.states
	#			- ancillas = which of these variables are ancillas... list of ints
	# Outputs:	none or false if the variable has already been added
	#=============================
	def add_variables(self, num_vars, init_states = None, ancillas = []):

		if init_states is None: init_states = [528]*num_vars
		else:
			if 0 in init_states and (self.states == [-1, 1] or self.states == [1, -1]): init_state = bin_2_spin(init_states)
			if -1 in init_states and (self.states == [0, 1] or self.states == [1, 0]): init_state = spin_2_bin(init_states)

		start = self.num_variables
		for var in range(num_vars): self.add_variable(init_state = int(init_states[var]))







	#=============================
	# Min-Sum Belief Propogation #
	#=============================


	# Description: Initialize messages from each variable to the corresponding factors, default is equally 0.0
	# Inputs:	- mess_state = message for each state for each variable ... dictionary of dictionaries e.g. {0: {0: 1.0, 1: 0.0}}
	# Outputs:	none
	# NOTE: the initial messages are just from vars -> facs
	#=============================
	def initialize(self, mess_state = {}):

		for node in self:

			# just vars -> facs
			if 'v' in node:

				node_num = int(node[1:])
				if node not in mess_state.keys():

					init = {}
					init_state = self.nodes[node]['init']
					for state in self.states:

						if state == init_state or init_state == 528 or init_state is None: init[state] = 0.0
						else: init[state] = 500

				else: init = mess_state[node]
				
				#self.nodes[node]['mess'] = {}... don't think this is necessary
				for neighbor in self[node]:

					neigh_num = int(neighbor[1:])
					if neigh_num not in self.nodes[node]['mess']: 
						self.nodes[node]['mess'][neigh_num] = {}
					
					self.nodes[node]['mess'][neigh_num] = init
				
				for state in self.states: self.nodes[node]['bel'][state] = 0.0

			else:
				self.nodes[node]['mess']
				for neighbor in self[node]:
					self.nodes[node]['mess'][int(neighbor[1:])] = {state: 0.0 for state in self.states}



	# Description: Calculate and send the message from the given factor to the given variable
	# Inputs:	- fac_num = which factor is sending the message... int
	#			- var_num = which variable is receiving the message... int
	# Outputs:	- the message
	#			- whether it changed
	#=============================
	def factor_to_variable(self, fac_num, var_num):

		m = {}
		if var_num in self.nodes['f' + str(fac_num)]['mess']:
			m_old = self.nodes['f' + str(fac_num)]['mess'][var_num].copy()
		else:
			m_old = []

		gen_pot = self.nodes['f' + str(fac_num)]['pot'].copy()
		if 'const' not in gen_pot.keys(): gen_pot['const'] = 0.0
		gen_pot['num vars'] -= 1

		# Incoming message contribution...
		# For each other incoming message, add a term for each state that only
		# contributes if the variable is in that state
		# NOTE: for now this assumes only 2 possible states...
		# 		to generalize to n states, must implement a higher order potential of the form:
		#		sum(in_mess[statei]*prod((x - state_j)/(statei - state_j), j != i))...
		#		currently the 0th and 1st order terms are explicitly written out, but this could be wrapped up into a loop...
		for neighbor in self['f' + str(fac_num)]:
			
			neigh_num = int(neighbor[1:])
			if 'v' in neighbor and neigh_num != var_num:
				if neigh_num not in gen_pot.keys(): gen_pot[neigh_num] = 0.0

				in_mess = self.nodes[neighbor]['mess'][fac_num]

				for state_i in self.states:
					for state_j in self.states:
						if state_i != state_j:
							gen_pot['const'] -= in_mess[state_i]*state_j/(state_i - state_j)
							gen_pot[neigh_num] += in_mess[state_i]/(state_i - state_j)


		# Factor contribution...
		# Determine the contribution from freezing this variable to each of its possible states
		# This part should be the same each time... maybe just calculate this once and store
		for state in self.states:
			pot = gen_pot.copy()

			if var_num in pot.keys():
				pot['const'] += pot[var_num]*state
				del pot[var_num]

			for k in gen_pot.keys():
				if type(k) == type((0, 0)):
					if var_num in k:

						(a, b) = k
						if var_num == a: 
							if b in pot.keys():
								pot[b] += pot[k]*state
							else:
								pot[b] = pot[k]*state

						if var_num == b: 
							if a in pot.keys():
								pot[a] += pot[k]*state
							else:
								pot[a] = pot[k]*state

						del pot[k]

			# Now use this instance's solver to attempt to find a solution
			m[state], _ = self.solver(pot, self.states)#), show_spec = True)
		

		#self.nodes['f' + str(fac_num)]['mess'][var_num] = m
		if self.norm: m = self.normalize(m) #self.normalize('f' + str(fac_num), var_num, False)

		#print('f' + str(fac_num) + ' -> v' + str(var_num))
		#print(m_old)
		#print(m)

		return m, m != m_old

	
	# Description: Send message from all factors to all of the variables on their borders (not internal ones)
	# Inputs:	none
	# Outputs:	- whether any messages changed
	#=============================
	def all_factors_to_variables(self, verbose = False, pool_size = 1):


		if pool_size > 1: 
			pool = Pool(pool_size, worker_init)
			res = {}

		num = 0
		for fac_num in range(self.num_factors):
			for neighbor in self['f' + str(fac_num)]:
				# making sure to only do border variables ... should this include ancillary variables?
				if 'v' in neighbor and len(self[neighbor]) > 1:
					var_num = int(neighbor[1:])
					if not self.nodes['v' + str(var_num)]['done']:

						if pool_size > 1:
							res[fac_num, var_num] = pool.apply_async(self.factor_to_variable, (fac_num, var_num))
							#self.nodes['f' + str(fac_num)]['mess'][var_num], ch = pool.apply_async(self.factor_to_variable, (fac_num, var_num)).get()
						else: 
							self.nodes['f' + str(fac_num)]['mess'][var_num], ch = self.factor_to_variable(fac_num, var_num)
							#num += ch

							if verbose: print('f' + str(fac_num) + ' -> ' + neighbor + ': ' + str(self.nodes['f' + str(fac_num)]['mess'][var_num]))

		if pool_size > 1:
			for fac_num, var_num in res.keys():
					self.nodes['f' + str(fac_num)]['mess'][var_num], ch = res[fac_num, var_num].get(18000)
					#num += ch

					if verbose: print('f' + str(fac_num) + ' -> ' + str(var_num) + ': ' + str(self.nodes['f' + str(fac_num)]['mess'][var_num]))

			pool.close()
			pool.join()

		return num


	# Description: Update this variable's belief
	# Inputs:	- var_num = which variable is receiving the message... int
	# Outputs:	- the new belief
	#			- changed: if any beliefs changed... bool
	#=============================
	def update_belief(self, var_num):

		b = {}
		b_old = self.nodes['v' + str(var_num)]['bel'].copy()
		changed = False
		for state in self.states:

			b[state] = 0.0
			for neighbor in self['v' + str(var_num)]:
				if 'f' in neighbor: b[state] += self.nodes[neighbor]['mess'][var_num][state]

			changed = (abs(b[state] - b_old[state]) >= self.threshold) or changed

		
		#self.nodes['v' + str(var_num)]['bel'] = b
		low = 0
		if self.norm: b = self.normalize(b) #self.normalize('v' + str(var_num), bel = True)
		else: low = minimum(b.values())

		#second_low = heapq.nsmallest(2, b.values())[-1]
		#if second_low - low > 1e9: self.nodes['v' + str(var_num)]['done'] = True

		return b, changed



	# Description: Update the beliefs for all variables
	# Inputs:	none
	# Outputs:	- num: how many beliefs changed... int
	#=============================
	def update_all_beliefs(self, verbose = False, pool_size = 1):

		if pool_size > 1: pool = Pool(pool_size, worker_init)

		num = 0
		res = {}
		for var_num in range(self.num_variables):
			if not self.nodes['v' + str(var_num)]['done']:
				if pool_size > 1: 
					res[var_num] = pool.apply_async(self.update_belief, (var_num,))
				else: 
					self.nodes['v' + str(var_num)]['bel'], ch = self.update_belief(var_num)

					num += ch
					if ch == False: self.nodes['v' + str(var_num)]['done'] = True
				#if verbose: print('v' + str(var_num) + ': ' + str(self.nodes['v' + str(var_num)]['bel']))
			#elif verbose: print('v' + str(var_num) + ' is frozen at: ' + str(self.nodes['v' + str(var_num)]['bel']))


		if pool_size > 1:

			for var_num in range(self.num_variables): 
				self.nodes['v' + str(var_num)]['bel'], ch = res[var_num].get(18000)
				num += ch

			pool.close()
			pool.join()


		return num



	# Description: Calculate the message to send from the given factor to the given variable
	# Inputs:	- var_num = which variable is sending the message... int
	#			- fac_num = which factor is receiving the message... int
	# Outputs:	- the message
	# NOTE: Currently this will send a 0 message from any non-border variable... could probably get a small speedup by checking if there are more than 1 factor neighbors and only calculating a message then
	#=============================
	def variable_to_factor(self, var_num, fac_num):

		m = {}
		for state in self.states:

			m[state] = 0.0
			for neighbor in self['v' + str(var_num)]:
				if 'f' in neighbor and neighbor != ('f' + str(fac_num)):
					m[state] += self.nodes[neighbor]['mess'][var_num][state]

			# since a v->f message is the sum over f->v messages, we want to divide out the number of factors. To see why, consider the following example:
			# Suppose each factor sends a message of 100 towards one state, so m[state] = 100*num_factors
			# Then, this will get built into each factor's potential, meaning they will each send this sized message back in the next iteration
			# Then the next v->f message will be order (100*num_factors)*num_factors
			# So, after n iterations, m[state] = 100*num_factors^n
			# Thus, if we divide out num_factors each iteration, we keep the messages bounded
			m[state] /= len(self['v' + str(var_num)])

		#self.nodes['v' + str(var_num)]['mess'][fac_num] = m

		if self.norm: m = self.normalize(m)#self.normalize('v' + str(var_num), fac_num, False)

		return m, m != self.nodes['v' + str(var_num)]['mess'][fac_num]


	# Description: Send message from all variables to all of the factors in their scope
	# Inputs:	none
	# Outputs:	- changed: whether any message changed
	#=============================
	def all_variables_to_factors(self, verbose = False, pool_size = 1):

		if pool_size > 1: pool = Pool(pool_size, worker_init)

		num = 0
		res = {}
		for var_num in range(self.num_variables)[::-1]:  #why backwards...?
			if not self.nodes['v' + str(var_num)]['done']:

				for neighbor in self['v' + str(var_num)]:
					if 'f' in neighbor:

						fac_num = int(neighbor[1:])

						if pool_size > 1: 
							res[var_num, fac_num] = pool.apply_async(self.variable_to_factor, (var_num, fac_num))
						else: 
							self.nodes['v' + str(var_num)]['mess'][fac_num], ch = self.variable_to_factor(var_num, fac_num)
							#num += ch
						#if verbose: print('v' + str(var_num) + ' -> ' + neighbor + ': ' + str(self.nodes['v' + str(var_num)]['mess'][fac_num]))


		if pool_size > 1:

			for var_num, fac_num in res: 
				self.nodes['v' + str(var_num)]['mess'][fac_num], ch = res[var_num, fac_num].get(18000)
				#num += ch

			pool.close()
			pool.join()


		return num



	# Description: Normalize all messages and beliefs in the factor graph s.t the lowest value for each one is 0.0
	# Inputs:	- node: which node to normalize... string
	# Outputs:	- none
	#=============================
	def normalize(self, mes):

		low = min(mes.values())
		for state in self.states: mes[state] -= low

		return mes






	# Description: Using the results of BP and solving each factor for its internal variables, get the final solution
	# Inputs:	- pool_size: how many threads to allow running on this
	# Outputs: - the solutions... list
	#=============================
	def finish(self, pool_size = 1):

		if pool_size > 1: pool = Pool(pool_size, worker_init)
		
		for fac_num in range(self.num_factors):
			if pool_size > 1: 
				pool.apply_async(self.factor_internal_solve, (fac_num, ))
			else: 
				self.factor_internal_solve(fac_num)	


		if pool_size > 1:
			pool.close()
			pool.join()


		return self.get_best_state()



	# Description: Solve this factor for the internal variables by fixing the border variables
	# Inputs:	fac_num: which factor to solve
	# Outputs:	none (beliefs of internal variables will be updated)
	#=============================
	def factor_internal_solve(self, fac_num):

		neighbors = self['f' + str(fac_num)]
		pot = self.nodes['f' + str(fac_num)]['pot'].copy()

		# Incoming message contribution...
		# For each other incoming message, add a term for each state that only
		# contributes if the variable is in that state
		# NOTE: for now this assumes only 2 possible states...
		# 		to generalize to n states, must implement a higher order potential of the form:
		#		sum(in_mess[statei]*prod((x - state_j)/(statei - state_j), j != i))...
		#		currently the 0th and 1st order terms are explicitly written out, but this could be wrapped up into a loop...
		for neighbor in self['f' + str(fac_num)]:
			
			neigh_num = int(neighbor[1:])
			if 'v' in neighbor:
				if neigh_num not in pot.keys(): pot[neigh_num] = 0.0

				in_mess = self.nodes[neighbor]['mess'][fac_num]

				for state_i in self.states:
					for state_j in self.states:
						if state_i != state_j:
							pot['const'] -= in_mess[state_i]*state_j/(state_i - state_j)
							pot[neigh_num] += in_mess[state_i]/(state_i - state_j)
		

		# Freezing each border variable
		for neighbor in neighbors:
			var_num = int(neighbor[1:])

			if 'v' in neighbor and len(self[neighbor]) > 1 and var_num not in self.ancillas:
				pot['num vars'] -= 1

				# Determine which state to freeze this variable to
				state = 5.28
				low = 10e9
				for s in self.states:
					b = self.nodes[neighbor]['bel'][s]
					if b < low:
						low = b
						state = s

				# Freeze each variable in the potential
				if var_num in pot.keys():
					pot['const'] += pot[var_num]*state
					del pot[var_num]

				for k in pot.keys():
					if type(k) == type((0, 0)):
						if var_num in k:

							(a, b) = k
							if var_num == a: 
								if b in pot.keys():
									pot[b] += pot[k]*state
								else:
									pot[b] = pot[k]*state

							if var_num == b: 
								if a in pot.keys():
									pot[a] += pot[k]*state
								else:
									pot[a] = pot[k]*state

							del pot[k]

		map_vars(pot)

		# Now use this instance's solver to find a solution
		_, solution = self.solver(pot, self.states)#), show_spec = True)

		# Use this solution to threshold (i.e. set really extreme/certain) the beliefs for the internal decision variables
		for neighbor in neighbors:
			var_num = int(neighbor[1:])

			if 'v' in neighbor and len(self[neighbor]) == 1 and var_num not in self.ancillas:
				for state in self.states:
					if solution[pot['mapping'][var_num]] == state: self.nodes[neighbor]['bel'][state] = 0
					else: self.nodes[neighbor]['bel'][state] = 1e9



	# Description: Find the best state based on the beliefs
	# Inputs: none
	# Outputs: - the best state... list
	# NOTE: Maybe the actual calculation of this state can be done while updating beliefs...
	#=============================
	def get_best_state(self):

		state = [5.28]*self.num_variables

		for var_num in range(self.num_variables):

			var = 'v' + str(var_num)
			if var not in self.ancillas:

				low = 10e9
				for s in self.states:
					b = self.nodes[var]['bel'][s]
					if b < low:
						low = b
						state[var_num] = s


		return [s for s in state if s != 5.28]




	



	#===================
	# Helper Functions #
	#===================


	# Description: Take this factor graph with potentials in qubo form and turn it into an equivalent ising factor graph
	# Inputs: - none (since this function will act on the instance calling it)
	# Outputs: - none (since it will internally change it)
	#=====================
	def qubo_2_ising(self):

		if self.states == [-1, 1]: return 'this graph is already in ising form'

		for node in self:

			if 'f' in node: 
				self.nodes[node]['pot'] = qubo_2_ising(self.nodes[node]['pot'])
			else:
				if self.nodes[node]['bel'] != {} and 0 in self.nodes[node]['bel'].keys():
					tmp = self.nodes[node]['bel'][0]
					del self.nodes[node]['bel'][0]
					self.nodes[node]['bel'][-1] = tmp

				
			for neighbor in self[node]:
				neigh_num = int(neighbor[1:])

				if self.nodes[node]['mess'] != {} and 0 in self.nodes[node]['mess'][neigh_num].keys():
					tmp = self.nodes[node]['mess'][neigh_num][0]
					del self.nodes[node]['mess'][neigh_num][0]
					self.nodes[node]['mess'][neigh_num][-1] = tmp

		self.states = [-1, 1]



	# Description: Take this factor graph with potentials in ising form and turn it into an equivalent qubo factor graph
	# Inputs: - none (since this function will act on the instance calling it)
	# Outputs: - none (since it will internally change it)
	#=====================
	def ising_2_qubo(self):

		if self.states == [0, 1]: return 'this graph is already in qubo form'

		for node in self:

			if 'f' in node: 
				self.nodes[node]['pot'] = ising_2_qubo(self.nodes[node]['pot'])
			else:
				if self.nodes[node]['bel'] != {}:
					tmp = self.nodes[node]['bel'][-1]
					del self.nodes[node]['bel'][-1]
					self.nodes[node]['bel'][0] = tmp

				
			for neighbor in self[node]:
				neigh_num = int(neighbor[1:])

				if self.nodes[node]['mess'] != {}:
					tmp = self.nodes[node]['mess'][neigh_num][-1]
					del self.nodes[node]['mess'][neigh_num][-1]
					self.nodes[node]['mess'][neigh_num][0] = tmp

		self.states = [0, 1]












	# Description: Return a graph of just the factors with edge weights corresponding to the number of shared variables
	# Inputs:	- frac_regions: fraction of regions to aim for (e.g. 4 factors and frac_regions = 0.5 => 2 regions)... float between 0 and 1
	# Outputs:	- graph 
	#===================
	def regionalize(self, num_regions = 35, recurs = True):

		_, parts = part_graph(self.shared_variable_graph(), max(num_regions, 2), recursive = recurs)
		parts = [abs(max(parts) - p) for p in parts]
		new_potentials = {f: {} for f in range(max(parts) + 1)}

		for i, p in enumerate(parts):
			old_potential = self.nodes['f' + str(i)]['pot']
			for k in old_potential:
				if k != 'num vars':
					if k in new_potentials[p].keys(): new_potentials[p][k] += old_potential[k]
					else: new_potentials[p][k] = old_potential[k]

		g = FactorGraph()
		for v in range(self.num_variables): 
			if 'init' in self.nodes['v' + str(v)]: init = self.nodes['v' + str(v)]['init']
			else: init = None
			g.add_variable(init_state = init)
		for f in range(max(parts) + 1): g.add_factor(new_potentials[f])


		g.ancillas = list(self.ancillas)
		g.solver = self.solver
		g.norm = self.norm
		g.states = self.states
		g.threshold = self.threshold

		return g



	# Description: Create a graph describing the overlap of variables amongst factors
	# Inputs:	- none
	# Outputs:	- the graph
	#=======================
	def shared_variable_graph(self):

		g = nx.Graph()
		g.graph['edge_weight_attr'] = 'weight'

		for fac_num in range(self.num_factors):
			if not g.has_node(fac_num): g.add_node(fac_num)
			for fac_num2 in range(fac_num + 1, self.num_factors):
				for neighbor in self['f' + str(fac_num)]:
					if neighbor in self['f' + str(fac_num2)]: 
						if not g.has_edge(fac_num, fac_num2): g.add_edge(fac_num, fac_num2, weight = 0)
						g[fac_num][fac_num2]['weight'] += 1

		return g



		# # Description: Combine a list of factors (labelled by the lowest factor number)
	# # Inputs:	- list of which factors to combine... list of ints
	# # Outputs:	- none
	# #==================
	# def combine_factors(self, fac_nums):

	# 	old = self.nodes['f' + str(fac_num)]['mess'][var_num]

	# 	potential = {}
	# 	mess = {}
	# 	for fac in fac_nums:
	# 		for k in self.nodes['f' + str(fac)]['pot'].keys():
	# 			if k in potential: potential[k] += self.nodes['f' + str(fac)]['pot'][k]
	# 			else: potential[k] = self.nodes['f' + str(fac)]['pot'][k]

	# 		for neigh_num in self.nodes['f' + str(fac)]['mess']:
	# 			for state in self.states:
	# 				if neigh_num not in mess: mess[neigh_num] = {}
	# 				if state not in mess[neigh_num]: mess[neigh_num][state] = 0.0

	# 				mess[neigh_num][state] += self.nodes['f' + str(fac)]['mess']

	# 		self.remove_node['f' + str(fac)]


	# 	self.add_factor(potential, num = min(fac_nums), messages = mess)
	# 	self.normalize('f' + str(min(fac_nums)), False)



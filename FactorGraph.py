import networkx as nx
from solvers import *
from helper import *

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
	def __init__(self, solv = brute_force, states = [-1, 1], normalize = True, threshold = 0.01):
		super().__init__()

		self.num_factors = 0
		self.num_variables = 0
		self.ancillas = []

		self.solver = solv
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
	def add_factor(self, num, potential, scope = None, messages = {}, ancillas = []):

		#if num < self.num_factors: return 'factor already in graph'

		self.num_factors += 1
		self.add_node('f' + str(num), pot = potential, mess = messages)

		# add edges to each node in the scope
		if scope == None:
			scope = []
			for k in potential.keys():
				if type(k) == type(0): scope += [k]
				elif k not in ['const', 'num vars']:
					for x in k: scope += [x]

		for k in scope:
			if ('v' + str(k)) not in self: self.add_variable(k, ancilla = (k in ancillas))
			self.add_edge('f' + str(num), 'v' + str(k))



	# Description: Add a variable node to the graph
	# Inputs:	- num = number to label node... int
	#			- init_state = what to initialize messages according to (effectively a guess)... element of self.states
	#			- ancilla = if this variable is ancillary (so won't be relevant to output/answer)
	# Outputs:	none or false if the variable has already been added
	#=============================
	def add_variable(self, num, init_state = None, ancilla = False):

		#if num < self.num_variables: return 'variable already in graph'

		#if init_state is None: init_state = self.states[0]
		self.num_variables += 1
		self.add_node('v' + str(num), mess = {}, bel = {}, init = init_state)

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
		for var in range(num_vars): self.add_variable(start + var, init_state = int(init_state[var]))







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

						if state == init_state: init[state] = 0.0
						else: init[state] = 10.0 #this value should definitely be tweaked...

				else: init = mess_state[node]
				
				#self.nodes[node]['mess'] = {}... don't think this is necessary
				for neighbor in self[node]:

					neigh_num = int(neighbor[1:])
					if neigh_num not in self.nodes[node]['mess']: 
						self.nodes[node]['mess'][neigh_num] = {}
					
					self.nodes[node]['mess'][neigh_num] = init
				
				for state in self.states: self.nodes[node]['bel'][state] = 0.0





	# Description: Calculate and send the message from the given factor to the given variable
	# Inputs:	- fac_num = which factor is sending the message... int
	#			- var_num = which variable is receiving the message... int
	# Outputs:	- if any of the messages changed... bool
	# NOTE: Sending message equates to saving it in the given variable
	#=============================
	def factor_to_variable(self, fac_num, var_num):

		m = {}
		if var_num in self.nodes['f' + str(fac_num)]['mess']:
			m_old = self.nodes['f' + str(fac_num)]['mess'][var_num].copy()
		else:
			m_old = []

		gen_pot = self.nodes['f' + str(fac_num)]['pot'].copy()
		if 'const' not in gen_pot.keys(): gen_pot['const'] = 0.0
		

		# Incoming message contribution...
		# For each other incoming message, add a term for each state that only
		# contributes if the variable is in that state
		# NOTE: for now this assumes only 2 possible states...
		# 		to generalize to n states, must implement a higher order potential of the form:
		#		sum(in_mess[statei]*prod((x - state_j)/(statei - state_j), j != i))...
		#		currently the 0th and 1st order terms are explicitly written out, but this could be wrapped up into a loop...
		for neighbor in self['f' + str(fac_num)]:
			if 'v' in neighbor and neighbor != ('v' + str(var_num)):
				neigh_num = int(neighbor[1:])
				if neigh_num not in gen_pot.keys(): gen_pot[neigh_num] = 0.0

				in_mess = self.nodes[neighbor]['mess'][fac_num]
	
				for state_i in self.states:
					for state_j in self.states:
						if state_i != state_j:
							gen_pot['const'] -= in_mess[state_i]*state_j/(state_i - state_j)
							gen_pot[neigh_num] += in_mess[state_i]/(state_i - state_j)


		# Factor contribution...
		# Determine the contribution from freezing this variable to each of its possible states
		for state in self.states:
			pot = gen_pot.copy()

			if var_num in pot.keys():
				pot['const'] += pot[var_num]
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
			m[state], _ = self.solver(pot, self.states)
		

		self.nodes['f' + str(fac_num)]['mess'][var_num] = m
		if self.norm: m = self.normalize('f' + str(fac_num), False)


		return m != m_old

	
	# Description: Send message from all factors to all of the variables in their scopes
	# Inputs:	none
	# Outputs:	- changed: if any message changed... bool
	#=============================
	def all_factors_to_variables(self, verbose = False):


		changed = False
		for fac_num in range(self.num_factors):
			for neighbor in self['f' + str(fac_num)]:
				if 'v' in neighbor:
					changed = self.factor_to_variable(fac_num, int(neighbor[1:])) or changed
					verbose and print('f' + str(fac_num) + ' -> ' + neighbor + ': ' + str(self.nodes['f' + str(fac_num)]['mess'][int(neighbor[1:])]))

		return changed



	# Description: Calculate and send the message from the given factor to the given variable
	# Inputs:	- var_num = which variable is receiving the message... int
	# Outputs:	- changed: if any beliefs changed... bool
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

		
		self.nodes['v' + str(var_num)]['bel'] = b
		if self.norm: b = self.normalize('v' + str(var_num), True)

		return changed



	# Description: Update the beliefs for all variables
	# Inputs:	none
	# Outputs:	- changed: if any beliefs changed... bool
	#=============================
	def update_all_beliefs(self, verbose = False):

		changed = False
		
		for var_num in range(self.num_variables):
			changed = self.update_belief(var_num) or changed
			verbose and print('v' + str(var_num) + ': ' + str(self.nodes['v' + str(var_num)]['bel']))

		return changed



	# Description: Calculate the message to send from the given factor to the given variable
	# Inputs:	- var_num = which variable is sending the message... int
	#			- fac_num = which factor is receiving the message... int
	# Outputs:	- if any of the messages changed... bool
	#=============================
	def variable_to_factor(self, var_num, fac_num):

		m = {}
		for state in self.states:

			m[state] = 0.0
			for neighbor in self['v' + str(var_num)]:
				if 'f' in neighbor and neighbor != ('f' + str(fac_num)):
					m[state] += self.nodes[neighbor]['mess'][var_num][state]

		
		m_old = self.nodes['v' + str(var_num)]['mess'][fac_num].copy()
		self.nodes['v' + str(var_num)]['mess'][fac_num] = m
		if self.norm: m = self.normalize('v' + str(var_num), False)

		return m != m_old


	# Description: Send message from all variables to all of the factors in their scope
	# Inputs:	none
	# Outputs:	- changed: if any message changed... bool
	#=============================
	def all_variables_to_factors(self, verbose = False):

		changed = False
		for var_num in range(self.num_variables):
			for neighbor in self['v' + str(var_num)]:

				if 'f' in neighbor:
					changed = self.variable_to_factor(var_num, int(neighbor[1:])) or changed
					verbose and print('v' + str(var_num) + ' -> ' + neighbor + ': ' + str(self.nodes['v' + str(var_num)]['mess'][int(neighbor[1:])]))

		return changed




	# Description: Normalize all messages and beliefs in the factor graph s.t the lowest value for each one is 0.0
	# Inputs:	- node: which node to normalize... string
	#			- bel: whether to normalize the belief (or the messages)... bool
	# Outputs:	- none
	#=============================
	def normalize(self, node, bel):

		if bel:
			low = min(self.nodes[node]['bel'].values())
			for state in self.states: self.nodes[node]['bel'][state] -= low

			return self.nodes[node]['bel']

		else:
			ms = self.nodes[node]['mess']
			for m in ms:
				low = min(ms[m].values())
				for state in self.states: self.nodes[node]['mess'][m][state] -= low

			return self.nodes[node]['mess']



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







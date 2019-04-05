from numpy import zeros, nonzero, shape

# Description: Take a qubo description and turn it into the equivalent ising description
# Inputs: - qubo: the qubo potential to be transformed... dictionary (key = variable(s), value = potential)
# Outputs: - ising: the potential in ising form
#=====================
def qubo_2_ising(qubo):

	ising = {'const': 0.0}

	# E0 = qubo_const + sum(diag(Q))/2.0 + sum(offdiag(Q))/4.0
	# h_i = Q_ii/2.0 + sum((Q_ij + Q_ji), j != i)/4.0
	# J_ij = (Q_ij + Q_ji)/4.0
	# Here it is assumed single valued keys (e.g. 1) are the diags and double valued keys (e.g. (1, 2)) are the off diags
	for variables in qubo:

		if variables not in ising: ising[variables] = 0.0

		if type(variables) == type(""):
			if variables == 'const': ising['const'] += qubo['const']
			else:  ising[variables] = qubo[variables]

		elif type(variables) == type(0): 

			# h
			ising[variables] += qubo[variables]/2.0

			# E0
			ising['const'] += qubo[variables]/2.0
		
		else:
			(i, j) = variables
			val = qubo[(i, j)]/4.0

			# J... Make upper triangular
			if i < j: ising[(i, j)] += val
			else: 
				if (j, i) in ising: ising[(j, i)] += val
				else: ising[(j, i)] = val

			# h
			if i in ising: ising[i] += val
			else: ising[i] = val

			if j in ising: ising[j] += val
			else: ising[j] = val

			# E0
			ising['const'] += val

	# remove all 0 elements to make sparse
	var_list = list(ising.keys())
	for variables in var_list: 
		if ising[variables] == 0.0: del ising[variables]


	return ising


# Description: Take an ising description and turn it into the equivalent qubo
# Inputs:	- ising: the ising potential to be transformed... dictionary
# Outputs:	- qubo: the potential in qubo form... dictionary
#=====================
def ising_2_qubo(ising):

	qubo = {'const': 0.0}


	# const = E0 + sum(J) - sum(h)
	# v_i = 2*h_i - 2*sum(J_ij + J_ji, j != i)
	# w_ij = 4*J_ij
	for variables in ising:

		if variables not in qubo: qubo[variables] = 0.0

		if type(variables) == type(""): 
			if variables == 'const': qubo['const'] += ising['const']
			else: qubo[variables] = ising[variables]

		elif type(variables) == type(0):

			# v
			qubo[variables] += 2.0*ising[variables]

			# const
			qubo['const'] -= ising[variables]

		else:
			(i, j) = variables
			val = 4.0*ising[(i, j)]

			# w... make upper triangular
			if i < j: qubo[(i, j)] += val
			else: 
				if (j, i) in qubo: qubo[(j, i)] += val
				else: qubo[(j, i)] = val

			# v
			if i in qubo: qubo[i] -= val/2.0
			else: qubo[i] = -val/2.0

			if j in qubo: qubo[j] -= val/2.0
			else: qubo[j] = -val/2.0

			# const
			qubo['const'] += val/4.0

	# remove all 0 elements to make sparse
	var_list = list(qubo.keys())
	for variables in var_list: 
		if qubo[variables] == 0.0: del qubo[variables]

	return qubo



# Description: Take a binary vector and turn it into a spin vector
# Inputs:	- vec: vector to be converted
# Outputs:	- spin version of vec
#====================
def bin_2_spin(vec):

	if type(vec) == type([]): 
		spin = vec[:]
		for i in range(len(vec)): spin[i] = 2*vec[i] - 1

	else: spin = 2*vec - 1

	return spin


# Description: Take a spin vector and turn it into a binary vector
# Inputs:	- vec: vector to be converted
# Outputs:	- binary version of vec
#====================
def spin_2_bin(vec):

	if type(vec) == type([]): 
		bin = vec[:]
		for i in range(len(vec)): bin[i] = (vec[i] + 1)/2

	else: bin = (vec + 1)/2

	return bin



# Description: Take a truth table potential and turn it into a potential dictionary as used in this file
# Inputs:	- table: a description of the truth table potential... dictionary (key = state of variables, value = potential)
# Outputs:	- the qubo potential in terms of the variables... dictionary
#=====================
def truth_table_2_qubo(table): pass




# Description: Turn a potential dictionary into a matrix equivalent
# Inputs:	- pot: a dictionary to be converted... dictionary
#			- num_vars: how many variables, in case this isn't provided by the potential... int
# Outputs:	- mat: matrix containing the potential... np matrix
#			- const: the constant of the potential... float
#=====================
def dict_2_mat(pot, num_vars = -1):

	if 'mapping' not in pot.keys(): map_vars(pot)
	mapping = pot['mapping']

	if num_vars == -1: num_vars = pot['num vars']
	mat = zeros((num_vars, num_vars))
	const = 0.0

	for variables in pot:

		if type(variables) == type(""):
			if variables == 'const': const = pot['const']
			else: continue
		elif type(variables) == type(0): mat[mapping[variables], mapping[variables]] = pot[variables]
		else: 
			(i, j) = variables
			mat[(mapping[i], mapping[j])] = pot[variables]


	return mat, const



# Description: Turn a matrix and constant potential into a dictionary
# Inputs:	- mat: matrix containing the potential... np matrix
#			- const: the constant of the potential... float
# Outputs:	- the potential dictionary in "upper triangular" form... dictionary
#=====================
def mat_2_dict(mat, const = 0.0):

	pot = {'const': const, 'num vars': len(mat)}

	for i in range(len(mat)):
		for j in range(len(mat)):
			if mat[i, j] != 0.0:
				if i == j: pot[i] = mat[i, i]
				else: 
					if (i, j) in pot: pot[(i, j)] += mat[i, j]
					else: pot[(i, j)] = mat[i, j]


	map_vars(pot)

	return pot


# Description: Create a mapping from the variable numbers to [0, 1, ..., number of variables - 1]
# Inputs:	- pot: the potential to find a mapping for
# Outputs: none (since the mapping is saved in the potential given)
#=====================
def map_vars(pot):

	mapping = {}
	free_var_start = -1

	for variables in pot:

		#if type(variables) == type(""): continue

		#if type(variables) == type(0): 
		#	if variables not in mapping: h_vars += [variables]#mapping[variables] = len(mapping)

		#else:
		if type(variables) == type((0,0)):
			for v in variables: 
				if v not in mapping: mapping[v] = len(mapping)


	for v in pot:
		if type(v) == type(0) and v not in mapping:
			if free_var_start == -1: free_var_start = len(mapping)
			mapping[v] = len(mapping)

	pot['mapping'] = mapping
	if free_var_start == -1: pot['free var start'] = len(mapping)
	else: pot['free var start'] = free_var_start


def get_num_vars(pot):

	xs = []
	for k in pot:
		if type(k) == type(0):
			if k not in xs: xs += [k]
		elif type(k) == type((0, 0)):
			(a, b) = k
			if a not in xs: xs += [a]
			if b not in xs: xs += [b]

	pot['num vars'] = len(xs)

	return len(xs)


def dwave_prepare(pot):

	if 'mapping' not in pot: map_vars(pot)
	mapping = pot['mapping']

	max_var = max(pot['mapping'].values()) + 1
	const = 0
	h = zeros(max_var)
	J = {}
	adj = [(v, v) for v in range(pot['free var start'], max_var)] # make sure free variables are included
	
	if 'const' in pot: const = pot['const']
	for k in pot:

		if type(k) == type(""): continue
		if type(k) == type(0): 
			h[mapping[k]] = pot[k]
		else:
			(a, b) = k
			a = mapping[a]
			b = mapping[b]

			if a <= b:
				J[a, b] = pot[k]
				adj += [(a, b)]
			else:
				J[b, a] = pot[k]
				adj += [(b, a)]

	#free_state = {v: int((pot[v] < 0) - (pot[v] > 0)) for v in pot['free vars']}

	return const, h, J, adj#, free_state





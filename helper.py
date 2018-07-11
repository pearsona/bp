from numpy import zeros

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



# Description: Take a truth table potential and turn it into a potential dictionary as used in this file
# Inputs:	- table: a description of the truth table potential... dictionary (key = state of variables, value = potential)
# Outputs:	- the qubo potential in terms of the variables... dictionary
#=====================
def truth_table_2_qubo(table): pass




# Description: Turn a potential dictionary into a matrix equivalent
# Inputs:	- pot: a dictionary to be converted... dictionary
# Outputs:	- mat: matrix containing the potential... np matrix
#			- const: the constant of the potential... float
#=====================
def dict_2_mat(pot):

	if 'mapping' not in pot.keys(): map_vars(pot)
	mapping = pot['mapping']

	mat = zeros((pot['num vars'], pot['num vars']))
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

	for variables in pot:

		if type(variables) == type(""): continue

		if type(variables) == type(0): 
			if variables not in mapping: mapping[variables] = len(mapping)

		else:

			for v in variables: 
				if v not in mapping: mapping[v] = len(mapping)


	pot['mapping'] = mapping








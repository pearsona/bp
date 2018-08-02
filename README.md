# Belief Propogation and LDPC Decoding

## [ldpc.py](ldpc.py)
This is the code to run BP on LDPC instances, which includes loading instances (from .mat files) of arbitrary density (the quality of the mapping for more than 3 bit parity checks is not guaranteed) and creating a factor graph with a factor for each parity check.

Usage example:

1) To run bp on all instances stored in a directory labelled 'instances':
```python
from ldpc import *
runBP(folder = 'instances')
```

2) To create a factor graph from a parity check matrix H and syndrome s, then change the solver to svmc:
```python
from ldpc import *
graph = create_factor_graph(H, s)
graph.solver = svmc
```



## [bp.py](bp.py)
This is the main code to use for running loopy min-sum belief propogation.

Usage examples:

1) To solve a default problem with all the default settings
```python
from bp import *
min_sum_BP()
```

2) To solve a factor graph you've made with the following specifications
```python
from bp import *

# use spin vector monte carlo to get solutions
solv = svmc 

# initialize the message from variable 0 to factor 0 and factor 1 (assuming an ising model)
mess = {0: {0: {-1: 1.5, 1: 2.0}, 1: {-1: 1.0, 1: 0.75}}} 

# allow bp to run for 10 iterations
num_iter = 10

# print all messages and beliefs as they're updated
verb = True

min_sum_BP(g = my_factor_graph, solver = solv, init_mess = mess, max_iter = num_iter, verbose = verb)
```


## [FactorGraph.py](FactorGraph.py)
This is the class (a child of [networkx](https://networkx.github.io/)) that defines the factor graph (more details at top of file) with the ability to run min-sum BP and to switch betweeen a QUBO and Ising representation, though technically the code is written to work with any definition of a potential and any possible variable states.

Usage example:

1) Create a basic qubo factor graph and then convert it into an ising factor graph
```python
from FactorGraph import *

g = FactorGraph(states = [0, 1])
g.add_factor(0, {'const': -1.0, 0: -1.0, 1: -1.0, (0, 1): -1.0, 'num vars': 2})
g.add_factor(1, {'const': -2.0, 1: 1.0, (1, 2): -3.0, 'num vars': 2})

g.graph_qubo_2_ising()
```
2) Define the potential: -2 + x<sub>1</sub> - 3x<sub>1</sub>x<sub>2</sub>
<br>NOTE: The values these variables can take (e.g. [0, 1]/QUBO or [-1, 1]/Ising) are defined for the factor graph, not for the specific potential
```python
potential = {'const': -2.0, 1: 1.0, (1, 2): -3.0, 'num vars': 2}
```


## [solvers.py](solvers.py)
This is a collection of various solvers that all require the potential and the possible variable states, but may have other parameters. Currently, there is a brute force solver and svmc written, with plans to possibly develop sa, sqa, and dwave.

## [helper.py](solvers.py)
This is a collection of various functions that are used throughout the above code. This includes functions to:
* convert between qubo and ising potentials 
* convert between a matrix and dictionary representation of potentials
* map a list of potential variables to indices, e.g. {1: 0, 4: 1, 5: 2} would map variable 1 to index 0, etc.




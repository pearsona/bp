# bp: Belief Propogation

## bp.py
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
iter = 10

# print all messages and beliefs as they're updated
verb = True

min_sum_BP(g = my_factor_graph, solver = solv, init_mess = mess, max_iter = iter, verbose = verb)
```


## FactorGraph.py
This is the class (a child of [networkx](https://networkx.github.io/)) that defines the factor graph (more details at top of file) with the ability to run min-sum BP and to switch betweeen a QUBO and Ising representation, though technically the code is written to work with any definition of a potential and any possible variable states.

Usage example:

Create a basic qubo factor graph and then convert it into an ising factor graph
```python
from FactorGraph import *

g = FactorGraph(states = [0, 1])
g.add_factor(0, {'const': -1.0, 0: -1.0, 1: -1.0, (0, 1): -1.0, 'num vars': 2})
g.add_factor(1, {'const': -2.0, 1: 1.0, (1, 2): -3.0, 'num vars': 2})

g.graph_qubo_2_ising()
```


## solvers.py
This is a collection of various solvers that all require the potential and the possible variable states, but may have other parameters. Currently, there is a brute force solver and svmc written, with plans to possibly develop sa, sqa, and dwave.

## helper.py
This is a collection of various functions that are used throughout the above code. This includes functions to:
* convert between qubo and ising potentials 
* convert between a matrix and dictionary representation of potentials
* map a list of potential variables to indices, e.g. {1: 0, 4: 1, 5: 2} would map variable 1 to index 0, etc.



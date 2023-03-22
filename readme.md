## Cirq Implementation

"Overview.ipynb" provides an overview on the proccess of creating the QAOA circuit and finding the lowest energy level by QAOA
"Utils_cirq.py" contains all neccessary functions
"multi_agent" was used for experiments with different settings (agents, layers, hyper paremters) and logging results. It also contains the Q_optim function which allows to optimize for 4 agents.


In contrast to the Qiskit implementation by this point in time there was no framework in Cirq that would allow QUBO to solve by QAOA.
To solve the CSG problem a parameterizes circuit has to be created according to the QUBO formulation.The cost layer consists parameterized ZZ(a,b,z) (Cnot(a,b), Z(b,z), Cnot(a,b)) and Z(a,s) gates with rotation z equal to 0.25* the interaction term between a and b in the QUBO matrix and s beeing the sum of row entries of the a-th row. The mixer layer consists of rotation X gates.
Parameters are realized by symbols from the sympy library and stored in dictonaries.

Optmization is based on the expectation value $\langle\gamma,\beta |H_c| \gamma,\beta\rangle$
A SGD algorithm has been implemented that works the following:
1 for given gamma, beta: calculate base energy
2 determine seperate changes in energy for little changes in gamma and beta -> gradient
3 do a gradient step
4 when stuck in a minima or plateau, perform random step

The lowest energy and corresponding parameter set is always recorded and returned after the max. epochs are reached.

After that the cirquit (with optimal parameter set) is beeing measured and from the samples the lowest cost solution is extracted.

Expectation bases optimization relies on $H_c$ and it was problematic to use this more than 3 agents, since the operator takes unreasonably long to constuct.
A work around was a different optimization strategy:
Using the final state vector of $\langle\gamma,\beta|$ and square it, we get the probability p(z) for each state z (coalition). The cost of this state vector is the sum of zQz * p(z) for all z. 
This way a circuit that produces high energy states with higher probability would be penelized.
However, this did not work well in practice. 

import numpy as np
import cirq
from cirq.circuits import InsertStrategy
import matplotlib.pyplot as plt
import math
from sympy import *
import random

def convert_to_BILP(coalition_values):
  """
  convert_to_BILP formulates the BILP problem for a given CSG problem instance
  :param coalition_values: dictionary of game/problem instance with key as the coalitions and value as coalition values
                          Also called the Characteristic function
  :return: tuple of (c,S,b) where c is the coalition values, S is a the binary 2D array and b is a binary vector
  """
  x={}
  for i in range(len(coalition_values)):
    x[i] = symbols(f'x_{i}')
  n = list(coalition_values.keys())[-1].count(',')+1                              #get number of agents
  S=[]
  for agent in range(n):
    temp = []
    for coalition in coalition_values.keys():
      if str(agent+1) in coalition:
        temp.append(1)                                                            #'1' if agent is present in coalition
      else:
        temp.append(0)                                                            #'0' if agent is not present in coalition
    S.append(temp)
  b=[1]*n                                                                         # vector b is a unit vector in case of BILP of CSGP
  c = list(coalition_values.values())                                             # vector of all costs (coalition values)
  return (c,S,b)


def get_coalition(agents):
    """There are 2-, 3-, and 4-agent problems prepared to choose from"""
    if agents == 2:
        coalition_values = {
            '1': 30,
            '2': 40,
            '1,2': 75

        }
    elif agents == 3:
        coalition_values = {
            '1': 30,
            '2': 40,
            '3': 25,
            '1,2': 70,
            '1,3': 60,
            '2,3': 65,
            '1,2,3': 90
        }
    elif agents ==4:
        coalition_values = {
            '1': 30,
            '2': 40,
            '3': 25,
            '4': 35,
            '1,2': 75,
            '1,3': 60,
            '1,4': 70,
            '2,3': 65,
            '2,4': 60,
            '3,4': 80,
            '1,2,3': 90,
            '1,2,4': 95,
            '1,3,4': 100,
            '2,3,4': 110,
            '1,2,3,4': 160
        }
    else:
        raise Exception("Invaild number of agents")
    return coalition_values

def get_QUBO_coeffs(c,S,b,P):
  """
  get_QUBO_coeffs converts the BILP problem instace into linear and quadratic terms required for QUBO formulation
  :param c: list of coalition values
         S: a binary 2D array
         b: a binary vector
         P: Penalty coefficient for the constraints of BILP problem to convert into an unconstrained QUBO problem
  :return linear: dictionary of linear coefficient terms
          quadratic: dictionary of quadratic coefficient terms
  """
  x={}
  for i in range(len(c)):
    x[i] = symbols(f'x_{i}')
  final_eq = simplify(sum([c_value*x[i] for i,c_value in enumerate(c)])+P*sum([expand((sum([x[idx] for idx,element in enumerate(agent) if element])-1)**2) for agent in S])) #simplify numerical equation
  linear={}
  quadratic={}
  for term in final_eq.as_coeff_add()[1]:
    term = str(term)
    if '**' in term:                                                              #get the coefficient of the squared terms as linear coefficients (diagonal elements of the Q matrix in the QUBO problem)
      if term.count('*')==3:
        linear[term.split('*')[1]] = float(term.split('*')[0])
      else:
        if not term.startswith('-'):
          linear[term.split('**')[0]] = float(1)
        else:
          linear[term.split('**')[0][1:]] = float(-1)
    elif term.count('*')==1:                                                        #get the coefficient of the linear terms (diagonal elements of the Q matrix in the QUBO problem)
      linear[term.split('*')[1]] += float(term.split('*')[0])
    else:
      quadratic[(term.split('*')[1],term.split('*')[2])] = float(term.split('*')[0])  #get the coefficient of quadratic terms (upper diagonal elements of the Q matrix of the QUBO problem)
  linear = {k:-v for (k,v) in linear.items()}
  #{k: abs(v) for k, v in D.items()}
  quadratic = {k:-v for (k,v) in quadratic.items()}
  return linear,quadratic

def get_Q(coalition_values, qubo_penalty = 50 * -1):
    """
    creates a QUBO matrix for given coalition values and QUBO penalty
    :param coalition_values:
    :param qubo_penalty:
    :return: normalized QUBO matrix
    """
    c, S, b = convert_to_BILP(coalition_values)  # A function in Utils_CSG.py

    linear, quadratic = get_QUBO_coeffs(c, S, b, qubo_penalty)
    Q = np.zeros([len(linear), len(linear)])
    Qmax = 0
    # diagonal elements
    for key, value in linear.items():
        if Qmax < abs(value):
            Qmax = abs(value)

        Q[int(key.split('_')[1]), int(key.split('_')[1])] = value

    # non diagonal elements
    for key, value in quadratic.items():
        if Qmax < abs(value / 2):
            Qmax = abs(value / 2)
        Q[int(key[0].split('_')[1]), int(key[1].split('_')[1])] = value / 2
        Q[int(key[1].split('_')[1]), int(key[0].split('_')[1])] = value / 2
    Q = Q / Qmax

    return Q

def cirq_init(n_prob):
    """Initializes a circuit of n_prob Line Qubits in equal superposition
    :param n_prob: problem size, number of qubits that the circuit should have
    :return the circuit itself and the qubits used in it"""
    circuit = cirq.Circuit() #initializing a circuit object
    qubits = cirq.LineQubit.range(n_prob) # creating n_prob qubits
    circuit.append(cirq.H(q) for q in qubits) # adding the just created qubits each with a H-Gate to the circuit
    return circuit, qubits


def mixer_layer(in_circuit, qubits, beta_value: float):
    """
    Adds a mixer layer to circuit with parameter beta_value
    :param in_circuit: circuit to which the mixer layer will be added
    :param qubits: qubits used in the input circuit
    :param beta_value: rotation of gates in mixer layer
    :return: copy of the input circuit with the added mixer layer
    """

    circuit = in_circuit.copy()
    # without copy() circuit would still be the same instance of in_circuit
    # and mixer() would in-place change in_circuit

    n_qubits = len(circuit.get_independent_qubit_sets())
    circuit.append([cirq.X(q) ** beta_value for q in qubits],
                   strategy=InsertStrategy.NEW_THEN_INLINE)
    # adding an X-gate to the power of beta_value to every qubit. Using an insert strategy to put them all in the same moment
    # for better readability in the diagram later
    return circuit


def cost_layer(in_circuit: cirq.circuits.circuit.Circuit, qubits,  gamma_value: float,
               Q: np.ndarray) -> cirq.circuits.circuit.Circuit:
    """
    Adds the cost layer defined by QUBO matrix Q to circuit with the parameter gamma_value
    :param in_circuit:
    :param qubits:
    :param gamma_value:
    :param Q:
    :return:
    """

    circuit = in_circuit.copy()
    nrow = np.size(Q, 0)
    ncol = np.size(Q, 1)
    w = -Q.sum(axis=1)
    # In the Q matrix, the sum of the i-th row entries represent the coefficients for the single Z roational gate on the i-th qubit,
    # while the of-diagonal non-zero elements are the coefficients for mixed terms.
    # A linear term will be implemented as Z rotation gate on the respective qubit,
    # while a mixed term is implemented as ZZ rotational gate on the 2 respective qubits.
    # the rotations are parametrized by the gamma_value for this specific layer

    for row in range(nrow):
        # we don't need to iterrate through the whole matrix, since it's symetrical. The upper right part and diagonal is enough
        for col in range(row, ncol):

            if row == col:
                lin = Q[row, col]
                continue
                # if the element is on the diagonal we will collect the coefficient to apply the single Z gate after all
                # ZZ gates are done on this qubit

            elif Q[row, col] != 0:
                # when we have a non-diagonal element that is not zero we append a gama_value
                # parametrized ZZ rotation gate on the resprective 2 qubits
                circuit.append(cirq.ZZ(qubits[row], qubits[col]) ** (0.5 * gamma_value * Q[row, col]))
        # the single qubit z gate for linear terms at the end of all ZZ gate for this qubit
        if lin != 0:
            circuit.append(cirq.Z(qubits[row]) ** (gamma_value * w[row]))

    return circuit


def ez_filter(n, dim):
    '''
    Creates matrices like Z_n (for i={n,...,#qubits}) , where we apply the Z gate to the n-th qubit
    in a circuit of dim number of qubits
    :param n:
    :param dim:
    :return
    '''
    # start with Pauli-Z matrix
    ez_f = np.eye(2)
    ez_f[1, 1] = -1

    if n == 1:
        # with for Z_1 we need to do the tensor product of Pauli-Z with dim-1 unit matrices
        ez_f = np.kron(ez_f, np.eye(2 ** (dim - 1)))

    else:
        # for Z_n we need a tensor product of n-1 unit matrices then Pauli-Z and again unit matrices for all the other qubits
        ez_f = np.kron(np.eye(2 ** (n - 1)), ez_f)
        ez_f = np.kron(ez_f, np.eye(2 ** (dim - n)))

    return ez_f

def m_op(Q):
    '''
    A matrix corresponding to the problem hamiltonian H_c. To compute the expectation values in terms of cost for the qubo
    :param Q:
    :return
    '''

    dim = len(Q[1])

    nrow = np.size(Q, 0)
    ncol = np.size(Q, 1)

    w = Q.sum(axis=1)
    ez = np.zeros((2 ** nrow, 2 ** ncol))

    for row in range(nrow):

        for col in range(ncol):

            if row == col:
                lin = Q[row, col]

                continue

            elif Q[row, col] != 0:

                ez += 0.25 * Q[row, col] * ez_filter(row + 1, dim) @ ez_filter(col + 1, dim)

        if lin != 0:
            ez -= w[row] * ez_filter(row + 1, dim)

    return ez

def m_z(dim):
    """For measurments in the computational basis this function can create a matrix corresponding to the
    tensor produkt of number of dim Pauli-Z matrices.

    :param dim: number of qubits in the circuit
    :return
    """
    mat_Z = np.eye(2)
    mat_Z[1,1] = -1
    M = [1]
    for i in range(dim):
        M = np.kron(M, mat_Z)
    return M


def exp_optim(param_dict, circuit, lr, epochs, M,  prt = False):
    """

    :param param_dict: parameter dictionary that will be optimized
    :param circuit: circuit used in the training
    :param lr: learning rate
    :param epochs: number of maximum gradient steps
    :param M: hamiltonian for expectation <y,b|M|y,b>
    :param prt: when true prints out the optimal energy found every 25 epochs and whenever a random step was performed
    :return: optimal parameter dictionary
    """
    # Creating the meassurement operator

    # setting up some utilities
    opt_energy = float('inf')
    opt_param_dict = {}
    simulator = cirq.Simulator()  # initializing the simulator object
    params = cirq.ParamResolver(param_dict)

    random_restart = 0
    vals = []

    for step in range(epochs):

        # The final state vector and energy of the circuit with the parameter values given by param_dict
        base_state = abs(simulator.simulate(circuit, param_resolver=params).final_state_vector)
        base_energy = base_state @ M @ base_state
        vals.append(base_energy)
        # whenever the energy is lower than previously detected, the best value and it's parameter will be stored
        if base_energy < opt_energy:
            opt_energy = base_energy
            opt_param_dict = param_dict.copy()

        # Calculating the gradient numerically
        grad = np.zeros(len(param_dict))
        j = 0
        for i in param_dict:

            param_dict[i] += 0.001

            params = cirq.ParamResolver(param_dict)
            eps_state = abs(simulator.simulate(circuit, param_resolver=params).final_state_vector)
            param_dict[i] -= 0.001
            eps = base_energy - (eps_state @ M @ eps_state)
            grad[j] = eps
            j += 1

        grad = grad * lr

        j = 0
        for i in param_dict:
            param_dict[i] += grad[j]
            j += 1

        # the energy of the learned state = base state + gradient * learning rate
        params = cirq.ParamResolver(param_dict)  # new parameter after the gradient step
        vec = abs(simulator.simulate(circuit, param_resolver=params).final_state_vector)
        energy = vec @ M @ vec  # energy after the gradient step
        if energy < opt_energy:
            opt_energy = energy
            opt_param_dict = param_dict.copy()
        if energy > base_energy - 0.001:

            random_restart += 2

        if random_restart >= epochs // 20:
            random_restart = 0
            if prt:
                # print("possible local minimum -> random step")
                pass
            for par in param_dict:
                param_dict[par] += random.uniform(-2, 2)

        if not step % 50 and prt:
            print(f"Epoch {step + 1}, Energy: {opt_energy}")
    if prt:
        x = (range(len(vals)))
        plt.plot(x, vals)
        plt.show()
    return opt_param_dict.copy(), opt_energy


def decode(solution, coalition_values):
    """
    Convert the solution binary string into a coalition sructure(a list of coalitions)
    :param solution: binary string solution. Example: [1, 0, 1, 0, 0, 0, 0]
    :return
    """
    output = []
    for index, element in enumerate(solution):
      if int(element) != 0:
        output.append(set(list(coalition_values)[index].split(',')))
    return output


def to_bin_array(n, agents):
    """
    :param n: decimal number of measurement
    :param agents: number of agents used
    :return: binary string of the solution used by decode function
    """
    ar = np.zeros(2 ** agents - 1)
    for i in str(bin(n))[2:]:
        ar = np.append(ar, int(i))

    return ar[-(2 ** agents - 1):]


def find_sol(sol_counter, coalition_values):
    """
    Finds the optimal solution in a measurement histogram
    :param sol_counter:
    :param coalition_values:
    :return:
    """
    min_cost = float("inf")
    tot = 0
    ag = math.log2(len(coalition_values) + 1)
    ag = int(ag)
    Q = get_Q(coalition_values)
    for k in sol_counter:
        v = sol_counter[k]
        tot += v
        bits = np.array(to_bin_array(k, ag))
        cost_sol = bits @ Q @ bits
        if cost_sol < min_cost:
            min_cost = cost_sol
            sol = k
    prob = sol_counter[sol]/tot
    sol_coalition = decode(np.array(to_bin_array(sol, ag)), coalition_values)
    return sol_coalition, sol, prob


def state_vector_params(ga, be, circuit):
    """
    Simulates the state vector of the circuit |γ,β> with the given parameter
    Used only in visualization with p = 1
    :param ga:
    :param be:
    :param circuit:
    :return:
    """
    simulator = cirq.Simulator()  # initializing the simulator object
    params = cirq.ParamResolver({"γ_0": ga, "β_0": be})
    # the param resolver maps values from the function input to the respective variables in the circuit

    result = abs(simulator.simulate(circuit, param_resolver=params).final_state_vector)
    # simulates the final state vector of the circuit with the input values for the parameters gamma and beta
    # only take tha absolut values since the imaginary parts should only exist due to rounding errors
    tot = 0
    for elm in result:
        tot += elm ** 2
    if 1 != round(tot, 3):
        print("Warning: final state vector is normalized. Difference to 1 is more than .001")

    return result



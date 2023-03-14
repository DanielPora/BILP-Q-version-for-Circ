from Utils_CSG import *
import numpy as np
import cirq
from cirq.circuits import InsertStrategy


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

def get_Q(coalition_values):
    c, S, b = convert_to_BILP(coalition_values)  # A function in Utils_CSG.py
    qubo_penalty = 50 * -1
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
    Return: the circuit itself and the qubits used in it"""
    circuit = cirq.Circuit() #initializing a circuit object
    qubits = cirq.LineQubit.range(n_prob) # creating n_prob qubits
    circuit.append(cirq.H(q) for q in qubits) # adding the just created qubits each with a H-Gate to the circuit
    return circuit, qubits


def mixer_layer(in_circuit, qubits, beta_value: float):
    """Adds a mixer layer to circuit with parameter beta_value"""
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
    """Adds the cost layer defined by QUBO matrix Q to circuit with the parameter gamma_value"""

    circuit = in_circuit.copy()
    nrow = np.size(Q, 0)
    ncol = np.size(Q, 1)
    w = np.zeros(len(Q[0]))
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
                circuit.append(cirq.ZZ(qubits[row], qubits[col]) ** (gamma_value * Q[row, col]))
        # the single qubit z gate for linear terms at the end of all ZZ gate for this qubit
        if lin != 0:
            circuit.append(cirq.Z(qubits[row]) ** (gamma_value * w[row]))

    return circuit
# Introduction <a id='introduction'></a>

The Deutsch-Jozsa algorithm is a quantum algorithm that determines whether a given Boolean function is constant or balanced. A Boolean function takes one or more bits as input and produces a single bit as output. A constant function always produces the same output, regardless of the input, while a balanced function produces output that is equally likely to be 0 or 1 for exactly half of the possible input values, and always produces the opposite output for the other half of the inputs.

The Deutsch-Jozsa algorithm is able to determine whether a given Boolean function is constant or balanced with just one function evaluation on a quantum computer. This is much faster than classical algorithms, which require at least two function evaluations to determine whether a function is constant or balanced.

The algorithm works by preparing a quantum computer in a superposition of all possible input values, applying a function evaluation to this superposition, and then measuring the resulting state. The measurement result will be either 0 or 1, and depending on the function being evaluated, this result will either be the same for all input values (indicating a constant function) or equally likely to be 0 or 1 for half of the input values and the opposite for the other half (indicating a balanced function).

The Deutsch-Jozsa algorithm was proposed by David Deutsch and Richard Jozsa in 1992 and is one of the earliest examples of a quantum algorithm that provides an exponential speedup over classical algorithms.

## The Deutsch-Jozsa algorithm <a id='quantum-solution'> </a>

Below is the generic circuit for the Deutsch-Jozsa algorithm with a quantum oracle that maps the state $\vert x\rangle \vert y\rangle $ to $ \vert x\rangle \vert y \oplus f(x)\rangle$, where $\oplus$ is addition modulo $2$.

![deutsch_steps.png](deutsch_steps.png)

### Our goal here is to replicate the above quantum circuit and to do this, we will follow the below steps:

</br>

[Go to the Notebook for the implementation.](https://github.com/Austinuc/Quantum-Algorithms/blob/main/Deutsch-Jozsa-algorithm/Deutsch_jozsa_algorithm.ipynb)

</br></br>

### Reference:

---

https://qiskit.org/textbook/ch-algorithms/deutsch-jozsa.html

http://einsteinrelativelyeasy.com/index.php/quantum-mechanics/154-hadamard-gate-on-multiple-qubits

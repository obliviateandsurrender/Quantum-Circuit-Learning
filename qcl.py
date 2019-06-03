import numpy as np
import itertools
import copy
import matplotlib.pyplot as plt
from scipy import linalg

import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *
from pyquil.quil import DefGate
from pyquil import get_qc
from pyquil.paulis import *

class QCL(object):

    def __init__(self, num_qubits, depth, proc_type, steps, initial_theta, operator_programs=None, learning_rate=1.0, epochs=1, batches=1, qvm=None):

        self.n_qubits = num_qubits
        self.qvm = self.make_qvm(qvm)
        self.depth = depth
        self.circ = pq.Program()
        self.input_circ = pq.Program()
        self.output_circ = pq.Program()
        self.T = steps       # Ising Hamiltonian Preparation
        self.ising_circ = self.ising_prog()
        self.grad_circ = [pq.Program(), pq.Program()]
        self.operator_programs = operator_programs
        self.proc_type = proc_type # Classification?
        if self.proc_type:
            self.loss = 1   # Cross Entropy
        else:
            self.loss = 0   # Mean Square
        self.initial_theta = initial_theta
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batches = batches
        self.results = dict()

    # Return QVM Forest Connection
    @staticmethod
    def make_qvm(qvm=None):
        if qvm:
            return qvm
        else:
            return api.QVMConnection()

    @staticmethod
    def multi_kron(*args):
        ret = np.array([[1.0]])
        for q in args:
            ret = np.kron(ret, q)
        return ret
    
    @staticmethod
    def multi_dot(*args):
        for i, q in enumerate(args):
            if i == 0:
                ret = q
            else:
                ret = np.dot(ret, q)
        return ret

    @staticmethod
    def generate_batches(X, y, batch_size):
        num_complete_batches = len(X)//batch_size
        permutation = list(np.random.permutation(len(X)))
        shuff_X = X[permutation, :]
        shuff_y = y[permutation]
        batches = []

        for k in range(0, num_complete_batches):
            batch_X = shuff_X[k*batch_size:(k+1)*batch_size, :]
            batch_y = shuff_y[k*batch_size:(k+1)*batch_size]
            batches.append((batch_X, batch_y))

        if m % batch_size != 0:
            batch_X = shuff_X[num_complete_batches*batch_size:m]
            batch_y = shuff_y[num_complete_batches*batch_size:m]
            batches.append((batch_X, batch_y))

        return batches

    # Prepare Input Circuit
    def input_prog(self, sample):
        circ = pq.Program()
        for j in range(self.n_qubits):
            circ += RY(np.arcsin(sample[0]),j)
            circ += RZ(np.arccos(sample[0]**2), j)
        return circ
    
    # MODIFY THIS USING PAULI OPERATORS
    # Prepare Ising Hamiltonian for evolution
    def ising_prog(self, trotter_steps=10):
        # Initilize coefficients
        gate_I = np.eye(2)
        gate_X = np.array([[0.0, 1.0],
                           [1.0, 0.0]])
        gate_Z = np.array([[1.0, 0.0],
                           [0.0, -1.0]])
        
        h_coeff = np.random.uniform(-1.0, 1.0, size=self.n_qubits)
        J_coeff = dict()
        for idx in itertools.combinations(range(self.n_qubits), 2):
            J_coeff[idx] = np.random.uniform(-1.0, 1.0)
        
        for steps in range(trotter_steps):
            non_inter = [linalg.expm(-(1j)*self.T/trotter_steps*self.multi_kron(*[h * gate_X if i == j else gate_I for i in range(n_qubits)])) for j, h in enumerate(h_coeff)]
            inter = [linalg.expm(-(1j)*self.T/trotter_steps*self.multi_kron(*[J * gate_Z if i == k[0] else gate_Z if i == k[1] else gate_I for i in range(n_qubits)])) for k, J in J_coeff.items()]
            ising_step = self.multi_dot(*non_inter+inter)

            if not steps:
                ising_gate = ising_step
            else:
                ising_gate = self.multi_dot(ising_step, ising_gate)

        ising_name = 'ISING_GATE'
        ising_prog = pq.Program().defgate(ising_name, ising_gate)
        ising_prog.inst(tuple([ising_name] + list(reversed(range(n_qubits)))))

        '''
        # Prepare Pauli Terms
        ising_self = PauliSum([term_with_coeff(sX(i), self.T/trotter_steps*h_coeff[i]) for i in range(self.n_qubits)])
        print(ising_self)
        ising_inter = PauliSum([term_with_coeff(sZ(idx[0])*sZ(idx[1]), self.T/trotter_steps*coeff) for idx, coeff in J_coeff.items()])
        print(ising_inter)
        print(simplify_pauli_sum(ising_self + ising_inter))
        #print(check_commutation(ising_self + ising_inter))
        ising_ham = exponentiate_commuting_pauli_sum(ising_self + ising_inter)
        # Trotterize
        #ising_ham = #trotterize(ising_self, ising_inter, trotter_steps=trotter_steps)
        return ising_ham
        '''
        return ising_prog

    # Prepare Output Circuit
    def output_prog(self, theta):
        circ = pq.Program()
        theta = theta.reshape(3, self.n_qubits, self.depth)
        for i in range(self.depth):
            circ += self.ising_circ
            for j in range(self.n_qubits):
                rj = self.n_qubits-j-1
                circ += RX(theta[0, rj, i], j)
                circ += RZ(theta[1, rj, i], j)
                circ += RX(theta[2, rj, i], j)
        return circ

    # Prepare Gradient Circuit
    def gradient_prog(self, theta, idx, sign):
        circ = pq.Program()
        theta = theta.reshape(3, self.n_qubits, self.depth)
        for i in range(self.depth):
            circ += self.ising_circ
            for j in range(self.n_qubits):
                rj = n_qubits-j-1
                if idx == (0,rj,i):
                    circ += RX(sign*np.pi/2.0, j)
                circ += RX(theta[0,rj,i], j)
                if idx == (1,rj,i):
                    circ += RZ(sign*np.pi/2.0, j)
                circ += RZ(theta[1,rj,i], j)
                if idx == (2,rj,i):
                    circ += RX(sign*np.pi/2.0, j)
                circ += RX(theta[2,rj,i], j)
        return circ

    def fit(self, X, y):
        X, y = self._make_data(X, y)
        self.results = self.gradient_descent(X, y)

    # Predcitions
    def predict(self, X):
        X = self._make_data(X)

        n_samples = len(X)
        n_operators = len(self.operator_programs)
        y_pred = np.zeros((n_samples, n_operators))
        
        for k in range(n_samples):
            prog = self.input_prog(X[k, :])
            prog += self.output_prog(self.results['theta'])
            y_pred[k, :] = self.results['coeff'] * \
                np.array(self.qvm.expectation(prog, self.operator_programs))
            if self.loss:
                y_pred[k, :] = np.exp(y_pred[k, :]) / \
                    np.sum(np.exp(y_pred[k, :]))
        return y_pred

    # Perform Gradient Descent
    def gradient_descent(self, orig_X, orig_y):
        history_theta, history_loss, history_grad = [], [], []
        coeff, theta = 1.0, self.initial_theta
        
        n_samples_orig = len(orig_X)
        n_theta = len(theta)
        n_operators = len(self.operator_programs)

        # Loop over epochs
        for e in range(self.epochs):
            batches = self.generate_batches(orig_X, orig_y, self.batches)
            n_batches = len(batches)
            for i, batch in enumerate(batches):
                X, y = batch
                n_samples = len(X)

                y_pred = np.zeros((n_samples, n_operators))

                for k in range(n_samples):
                    prog = self.input_prog(X[k, :])
                    prog += self.output_prog(theta)
                    y_pred[k, :] = coeff * \
                        np.array(self.qvm.expectation(prog, operator_programs))
                    if self.loss:
                       y_pred[k, :] = np.exp(y_pred[k, :]) / \
                           np.sum(np.exp(y_pred[k, :]))
                # Comput loss
                loss_value = self._compute_loss(y, y_pred)

                print('Epoch: {}/{} ::: Batch: {}/{} ::: Loss: {:.5f}'.format(e + 1, self.epochs, i+1, n_batches, loss_value))
                if not (e == self.epochs - 1 and i == n_batches - 1):
                    grad = np.zeros((n_samples, n_operators, n_theta))
                    for k in range(n_samples):
                        for j in range(n_theta):
                            for sign in [-1, 1]:
                                grad_sign = np.zeros(n_operators)
                                grad_progs = [self.gradient_prog(theta, j, sign) for x in   range     (n_operators)]
                                #grad_progs = [self.gradient_prog(theta, j, sign)]

                                for grad_prog in grad_progs:
                                    prog = self.input_prog(X[k,:])
                                    prog += grad_prog
                                    grad_sign += np.array(self.qvm.expectation(prog,        operator_programs))
                                grad[k, :, j] += sign / 2.0 * grad_sign
                
                    # Gradient update
                    grad_full = self._compute_grad_full(y, y_pred, grad)
                    if not self.loss:
                        grad_full_coeff = -2.0 * np.mean((y - y_pred) * y_pred)
                    # Update theta
                    theta -= self.learning_rate * grad_full
                    if not self.loss:
                        coeff -= self.learning_rate * grad_full_coeff
            
                # Append to history
                history_loss.append(loss_value)
                history_theta.append(theta)
                history_grad.append(grad)

            # Prepare results
            results = dict()
            results['theta'], results['coeff'] = theta, coeff
            results['loss'] = loss_value
            results['history_grad'] = history_grad
            results['history_loss'] = history_loss
            results['history_theta'] = history_theta

        return results

    # Data Preparations
    def _make_data(self, X, y=None):
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        if y is not None:
            if np.ndim(y) == 1:
                y = y.reshape(-1, 1)
            if self.loss and y.shape[1] == 1:
                y = np.c_[y, 1-y]
            return X, y
        else:
            return X

    # Loss Calculations
    def _compute_loss(self, y, y_pred):
        if not self.loss:
            return np.mean((y - y_pred) ** 2)
        else:
            return -1.0 * np.mean(np.sum(y * np.log(y_pred), axis=1))

    # Gradient Calculations
    def _compute_grad_full(self, y, y_pred, grad):
        if not self.loss:
            return -2.0 * np.mean((y - y_pred) * grad[:, 0, :], axis=0)
        else:
            return -1.0 * np.mean((grad[:, 0, :] - grad[:, 1, :]) * (y[:, 0, np.newaxis] - y_pred[:, 0, np.newaxis]), axis=0)

if __name__ == "__main__":
    np.random.seed(0)
    m = 20
    X = np.linspace(-0.95, 0.95, m)
    X_test = np.linspace(-1.0, 1.0, 20)
    y = X**2
    n_qubits = 3
    depth = 3
    initial_theta = np.random.uniform(0.0, 2*np.pi, size=3*n_qubits*depth)
    operator_programs = [pq.Program(Z(0))]
    len(operator_programs)
    proc_type = 0
    est = QCL(n_qubits,depth,proc_type,10,initial_theta,operator_programs=operator_programs, epochs=20, batches=20)
    est.fit(X,y)
    y_pred = est.predict(X_test)

    plt.figure()
    plt.plot(X, y, 'bs', X_test,  y_pred, 'r-')
    plt.show()
    plt.figure()
    plt.plot(np.arange(1, 21), est.results['history_loss'], 'bo-')
    plt.xticks(np.arange(1, 21))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

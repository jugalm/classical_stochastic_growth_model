'''
Jugal Marfatia
Numerical value function iteration
'''

import numpy as np


def find_nearest(array, value):
    id_x = (np.abs(array-value)).argmin()
    return id_x


def transition_matrix(vector=np.array([]), p=.5, simulations_n=100):

    result = np.zeros((len(vector), len(vector)))

    for i in range(0, len(vector)):
        for j in range(0, simulations_n):
            z_prime = p * np.log(vector[i]) + np.random.normal()
            z_id = find_nearest(vector, z_prime)
            result[i][z_id] += 1

    return np.array(result)/simulations_n


def expected_value(state_vector, prob_vector):
    return np.inner(np.array(state_vector), np.array(prob_vector))


class ValueFunction:
    def __init__(self, z_vector, grid_n, lambda1, beta, alpha, theta):
        self.k_vector = np.linspace(.1, 100, grid_n)
        self.z_vector = z_vector
        self.transition_matrix = transition_matrix(z_vector)
        self.lambda1 = lambda1
        self.beta = beta
        self.alpha = alpha
        self.grid_n = grid_n
        self.theta = theta
        self.k_prime_matrix = np.zeros((self.grid_n, len(self.z_vector)))
        self.value_matrix = np.zeros((self.grid_n, len(self.z_vector)))

    def solve_2(self, e=1, n=0):

        if e < .01:
            return n, self.value_matrix, self.k_prime_matrix
        else:
            new_value_matrix = np.zeros((self.grid_n, len(self.z_vector)))
            for i in range(0, len(self.k_vector)):
                for j in range(0, len(self.z_vector)):
                    c_vector = self.z_vector[j] * self.k_vector[i] + (1-self.theta) * self.k_vector[i] \
                               - self.k_vector
                    c_vector[c_vector < 0] = 0.1

                    value_vector = (np.power(c_vector, 1-self.lambda1)/(1-self.lambda1)) + \
                        self.beta * expected_value(self.value_matrix, self.transition_matrix[j])

                    new_value_matrix[i][j] = np.amax(np.array(value_vector))
                    self.k_prime_matrix[i][j] = self.k_vector[np.argmax(np.array(value_vector))]
            e = np.amax(np.abs(np.array(new_value_matrix) - np.array(self.value_matrix)))
            self.value_matrix = new_value_matrix
            return self.solve_2(e=e, n=n+1)


if __name__ == '__main__':

    new_object = ValueFunction(z_vector=[.9, .95, 1.05, 1.1], grid_n=20, lambda1=2, beta=.995, alpha=1, theta=.95)
    print new_object.solve_2()

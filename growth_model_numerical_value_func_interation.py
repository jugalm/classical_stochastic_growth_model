'''
Jugal Marfatia
Numerical value function iteration
'''

import numpy as np
import numpy.matlib as mt


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


class ValueFunction:
    def __init__(self, grid_n, lambda1, beta, alpha, theta, lower_bound=.95, upper_bound=1.05):
        self.k_steady_state = (alpha / ((1/beta) - (1-theta))) ** (1 / (1 - alpha))
        self.k_vector = np.linspace(lower_bound * self.k_steady_state, upper_bound * self.k_steady_state, grid_n) # create_grid(value=self.k_steady_state, n=grid_n, deviation=k_deviation)
        # self.z_vector = z_vector
        # self.transition_matrix = transition_matrix(z_vector)
        self.lambda1 = lambda1
        self.beta = beta
        self.alpha = alpha
        self.grid_n = grid_n
        self.k_prime = np.zeros((self.grid_n, 1))  # k_prime(i): next-period capital given current state k(i)

    ''' initialization of endogeneous variables '''
    def solve(self):
        v = np.zeros((self.grid_n, 1))  # v(i): value function at k(i)

        c = mt.repmat(np.power(self.k_vector, self.alpha), self.grid_n, 1).transpose() - \
            mt.repmat(self.k_vector, self.grid_n, 1)  # cons(i,j): consumption given state k(i) and decision k(j)

        # c[c < 0] = 0.0  # gives -Inf utility
        util = np.power(c, 1-self.lambda1)/(1-self.lambda1)  # util(i,j): current utility at state k(i) and decision k(j)
        return self.iterate(v=v, k_prime=self.k_prime, c=c, util=util)

    def iterate(self, e=1, it=0, v=None, k_prime=None, c=None, util=None):
        if e < .01:
            return self.k_prime
        else:
            print self.k_prime
            print e
            v_vec = mt.repmat(v, 1, self.grid_n).transpose()
            util_v_vec = util + self.beta * v_vec
            tv = np.array([util_v_vec.max(1)]).T
            self.k_prime = self.k_vector[util_v_vec.argmax(1)]
            e = np.max(np.abs(tv - v))  # criteria for tolerance
            v = tv  # update value function
            it = it + 1

            return self.iterate(e=e, it=it, v=v, k_prime=k_prime, c=c, util=util)

if __name__ == '__main__':
    # print(create_grid(value=3.96, n=100.0, deviation=.10))
    print(transition_matrix([.8, .9, 1.1, 1.2]))
    # new_object = ValueFunction(grid_n=2, lambda1=2, beta=.995, alpha=.36, theta=.1)
    # print new_object.solve()

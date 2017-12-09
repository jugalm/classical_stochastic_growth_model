'''
Jugal Marfatia, Tiana Randriamaro
Numerical value function iteration
'''
import sys
import numpy as np
import pandas as pd
from highcharts import Highchart
import plotly.plotly as py
import plotly.graph_objs as go
sys.setrecursionlimit(100000000)


def find_nearest(array, value):
    id_x = (np.abs(array-value)).argmin()
    return id_x


def transition_matrix(vector=np.array([]), p=.5, simulations_n=100):

    result = np.zeros((len(vector), len(vector)))

    for i in range(0, len(vector)):
        for j in range(0, simulations_n):
            z_prime = p * np.log(vector[i]) + np.random.randn()
            z_id = find_nearest(vector, z_prime)
            result[i][z_id] += 1

    return np.array(result)/simulations_n


def expected_value(state_vector, prob_vector):
    return np.inner(np.array(state_vector), np.array(prob_vector))


class ValueFunction:
    def __init__(self, z_vector, grid_n, lambda1, beta, alpha, theta, rho, transition_simulation):
        self.k_vector = np.linspace(0, 100, grid_n)
        self.z_vector = z_vector
        self.transition_matrix = transition_matrix(z_vector, p=rho, simulations_n=transition_simulation)
        self.lambda1 = lambda1
        self.beta = beta
        self.alpha = alpha
        self.grid_n = grid_n
        self.theta = theta
        self.k_prime_matrix = np.zeros((self.grid_n, len(self.z_vector)))
        self.value_matrix = np.zeros((self.grid_n, len(self.z_vector)))

    def solve(self, e=1, n=0):
        print(e)

        if e < .1:
            return n, self.value_matrix, self.k_prime_matrix
        else:
            new_value_matrix = np.zeros((self.grid_n, len(self.z_vector)))
            for i in range(0, len(self.k_vector)):
                for j in range(0, len(self.z_vector)):
                    c_vector = self.z_vector[j] * self.k_vector[i] + (1-self.theta) * self.k_vector[i] \
                               - self.k_vector
                    c_vector[c_vector == 0] = .1 # Since we cannot divide by 0

                    c_vector[c_vector < 0] = .01 # Since we cannot divide by 0 and consumption cannot be negative

                    value_vector = (np.power(c_vector, 1-self.lambda1)/(1-self.lambda1)) + \
                        self.beta * expected_value(self.value_matrix, self.transition_matrix[j])

                    value_vector[c_vector == .01] = -999999 # We cannot select k_prime which yields negative c

                    new_value_matrix[i][j] = np.amax(np.array(value_vector))
                    self.k_prime_matrix[i][j] = self.k_vector[np.argmax(np.array(value_vector))]
            e = np.amax(np.abs(np.array(new_value_matrix) - np.array(self.value_matrix)))
            self.value_matrix = new_value_matrix
            return self.solve(e=e, n=n+1)

    def plot_2d(self):
        h = Highchart(width=750, height=600)

        options = {
            'title': {
                'text': 'Value Function Iteration'
            },
            'xAxis': {
                'title': {
                    'text': "K - Capital Level"
                }
            },
            'yAxis': {
                'title': {
                    'text': "Value of Capital"
                }
            },
            'tooltip': {
                'crosshairs': False,
                'shared': True,
            },
            'legend': {
            }
        }

        h.set_dict_options(options)
        for x in range(0, len(self.z_vector)):
            df1 = pd.DataFrame({'k': self.k_vector[1:], 'value': self.value_matrix[1:, [x]].flatten()})

            df1 = df1[['k', 'value']]
            df1 = df1.values.tolist()
            h.add_data_set(df1, 'spline', 'z_' + str(x) + ' = ' + str(self.z_vector[x]), zIndex=1, marker={
                'fillColor': 'white', 'lineWidth': 2, 'lineColor': 'Highcharts.getOptions().colors[1]'})

        html_str = h.htmlcontent.encode('utf-8')

        html_file = open("chart.html", "w")
        html_file.write(html_str)
        html_file.close()

    def plot_3d(self, data_name="Value"):
        if data_name == "Policy":
            z_data = self.k_prime_matrix
        else:
            z_data = self.value_matrix

        data = [
            go.Surface(
                z=z_data,
                x=self.z_vector,
                y=self.k_vector,
                text='hover',
                hoverinfo='text',
            )
        ]
        layout = go.Layout(
            title='Value Function Iteration',
            autosize=False,
            width=700,
            height=700,
            scene=dict(
                zaxis=dict(
                    title=data_name),
                yaxis=dict(
                    title='K- Level of Capital'),
                xaxis=dict(
                    title='Z- Values'), ),
            margin=dict(
                l=65,
                r=50,
                b=65,
                t=90,
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename='3d-surface_' + data_name)

if __name__ == '__main__':

    new_object = ValueFunction(z_vector=np.linspace(0.1, 2, 20), grid_n=100, lambda1=2, beta=.995, alpha=1,
                               theta=.95, rho=.5, transition_simulation=1000)

    print(new_object.solve())
    new_object.plot_2d()
    new_object.plot_3d('Value')
    new_object.plot_3d('Policy')

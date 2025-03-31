import numpy as np
from numpy import sqrt
from dynamical_systems.DynamicalSystem import DynamicalSystem


class TwoTanks(DynamicalSystem):
    def __init__(self, state_size=2, input_size=1, output_size=1, non_linear_input=False):
        super().__init__(state_size, input_size, output_size, non_linear_input)
        self.input_state_linearity = True
        self.output_linearity = True
        self.exponent = 0.5

    def state_map(self, x, u):
        if self.non_linear_input:
            x1 = x[0] - 0.5 * sqrt(x[0]) + 0.4 * np.sign(u[0]) * (u[0])**2
        else:
            x1 = x[0] - 0.5 * sqrt(x[0]) + 0.4 * u[0]

        x2 = x[1] + .2 * sqrt(x[0]) - .3 * sqrt(x[1])
        x = np.zeros((self.state_size, 1))

        if x1 > 0:
            x[0] = x1
        else:
            x[0] = 0

        if x2 > 0:
            x[1] = x2
        else:
            x[1] = 0

        return x

    def output_map(self, xk):
        return np.array(xk[1])

    def system_dynamics(self, dim, flag=True):
        x_k = np.ones((self.state_size, 1))
        y_n = np.zeros((dim, 1))
        u_n = np.zeros((dim, self.input_size))
        noise = np.random.normal(1, 1.0, size=(self.input_size, dim))

        u = np.array([0.0])

        for i in range(0, dim):
            if i % 10000 == 0:
                print('.',  end='')

            u[0] = noise[0][int(i/5)]
            y_n[i] = self.output_map(x_k) * 1
            x_k = self.state_map(x_k, np.reshape(u, (self.input_size, 1))) * 1
            u_n[i] = u

        return y_n, u_n

    def loop(self, x_k, duk):
        y_n = np.array([])

        for i in range(0, len(duk)):
            u = np.reshape(np.array(duk[i]),(1, 1))
            u = u * self.stdU + self.meanU
            temp = self.output_map(x_k)
            temp = (temp - self.meanY) / self.stdY
            y_n = np.append(y_n, temp)
            x_k = self.state_map(x_k, u)

        return y_n*1, x_k*1

    def prepare_dataset(self, training_size, validation_size):
        y_n, u_n = self.system_dynamics(training_size, True)
        y_vn, u_vn = self.system_dynamics(validation_size, True)

        self.mean_y = np.mean(y_n)
        self.mean_u = np.mean(u_n)
        self.std_y = np.std(y_n)
        self.std_u = np.std(u_n)

        y_n = (y_n - self.mean_y) / self.std_y + np.random.normal(0, 0.02, (training_size, 1))
        y_vn = (y_vn - self.mean_y) / self.std_y + np.random.normal(0, 0.02, (validation_size, 1))
        u_n = (u_n - self.mean_u) / self.std_u+np.random.normal(0, 0.02, (training_size, 1))
        u_vn = (u_vn - self.mean_u) / self.std_u + np.random.normal(0, 0.02, (validation_size, 1))

        return np.reshape(u_n, (training_size, 1)), np.reshape(y_n, (training_size, 1)),\
                    np.reshape(u_vn, (validation_size, 1)), np.reshape(y_vn, (validation_size, 1))
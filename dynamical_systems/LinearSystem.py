import numpy as np
from dynamical_systems.DynamicalSystem import DynamicalSystem


class LinearSystem(DynamicalSystem):
    def __init__(self, state_size=3, input_size=1, output_size=1, non_linear_input=False, sigma=0.02):
        super().__init__(state_size, input_size, output_size, non_linear_input)
        self.A = np.array([[-0.5, 0.5, 0], [1/3, 1/3, 1/3], [0, 0.5, 0.5]])
        self.B = np.array([[1, 0.5, -0.5]]).T
        self.C = np.array([[1, 0, 0]])
        self.sigma = sigma
        self.mean_y = None
        self.mean_u = None
        self.std_y = None
        self.std_u = None

    def state_map(self, xk, uk):
        xk = np.dot(self.A, xk) + np.dot(self.B, uk)
        return xk

    def output_map(self, xk):
        y = np.dot(self.C, xk)
        return y

    def system_dynamics(self, dim):
        x_k = np.ones((self.state_size, 1))
        y_n = np.zeros((dim, 1))
        u_n = np.zeros((dim, self.input_size))

        noise = np.random.normal(0, 1, size=(self.input_size, dim))
        u = np.array([0.0])

        for i in range(0, dim):
            if i % 10000 == 0:
                print('.',  end='')

            u[0] = noise[0][int(i/7)]
            u_k = u[0]

            y_n[i] = self.output_map(x_k) * 1
            x_k = self.state_map(x_k, np.reshape(u_k, (self.input_size, 1))) * 1
            u_n[i] = u[0]

        return y_n, u_n

    def loop(self, x_k, duk):
        y_n = np.array([])
        x_k = np.reshape(x_k, (self.state_size, 1))

        for i in range(0,len(duk)):
            u = np.array(duk[i]) + np.random.normal(0, self.sigma, (1, 1)) * .0
            # u = u * self.std_u + self.mean_u

            temp = self.output_map(x_k)

            # temp = (temp - self.mean_y) / self.std_y
            y_n = np.append(y_n, temp) + np.random.normal(0, self.sigma, (1, 1)) * .0

            x_k = self.state_map(x_k, u) * 1

        return y_n, x_k

    def prepare_dataset(self, training_size, validation_size):
        y_n, u_n = self.system_dynamics(training_size)
        y_vn, u_vn = self.system_dynamics(validation_size)

        self.mean_y = np.mean(y_n)
        self.mean_u = np.mean(u_n)
        self.std_y = np.std(y_n)
        self.std_u = np.std(u_n)

        y_n = (y_n - self.mean_y) / self.std_y + np.random.normal(0, self.sigma, (training_size, 1))
        y_vn = (y_vn - self.mean_y) / self.std_y + np.random.normal(0, self.sigma, (validation_size, 1))
        u_n = (u_n - self.mean_u) / self.std_u + np.random.normal(0, self.sigma, (training_size, 1))
        u_vn = (u_vn - self.mean_u) / self.std_u + np.random.normal(0, self.sigma, (validation_size, 1))

        return np.reshape(u_n, (training_size, 1)),np.reshape(y_n, (training_size, 1)),\
                    np.reshape(u_vn, (validation_size, 1)), np.reshape(y_vn, (validation_size, 1))

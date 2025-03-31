import numpy as np
from dynamical_systems.DynamicalSystem import DynamicalSystem

class GearSystem(DynamicalSystem):
    def __init__(self, a_vals=(0.7, 0.85, 1.0), b=1.0, d=0.1, theta=(20.0, 40.0),
                 phi=(15.0, 35.0), initial_speed=0.0, initial_gear=1, sigma=0.02):
        super().__init__(state_size=2, input_size=1, output_size=1)
        self.a_vals = a_vals
        self.b = b
        self.d = d 
        self.theta = theta 
        self.phi = phi
        self.sigma = sigma
        self.x = np.array([[initial_speed], [initial_gear]])

        self.mean_y = None
        self.std_y = None
        self.mean_u = None
        self.std_u = None

    def state_map(self, xk, uk):
        speed = xk[0, 0]
        gear = xk[1, 0]
        u = uk[0, 0]

        a = self.a_vals[int(gear - 1)]

        speed_next = a * speed + self.b * u - self.d 
        gear_next = gear 

        if gear == 1 and speed_next > self.theta[0]:
            gear_next = 2
        elif gear == 2 and speed_next > self.theta[1]:
            gear_next = 3
        elif gear == 3 and speed_next < self.phi[1]:
            gear_next = 2
        elif gear == 2 and speed_next < self.phi[0]:
            gear_next = 1

        return np.array([[speed_next.item()], [gear_next]])

    def output_map(self, xk):
        return np.array([[xk[0,0]]])

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
        """
        Propagate the system with a given sequence of control inputs duk.
        """
        y_n = np.array([])
        x_k = np.reshape(x_k, (self.state_size, 1))
        
        for i in range(len(duk)):
            # Add optional noise to the control input
            u = np.array([[duk[i]]]) + np.random.normal(0, self.sigma, (1, 1))
            y = self.output_map(x_k) + np.random.normal(0, self.sigma)
            y_n = np.append(y_n, y)
            x_k = self.state_map(x_k, u)
            
        return y_n, x_k

    def prepare_dataset(self, training_size, validation_size):
        """
        Generate training and validation datasets using system dynamics.
        """
        y_n, u_n = self.system_dynamics(training_size)
        y_vn, u_vn = self.system_dynamics(validation_size)
        
        self.mean_y = np.mean(y_n)
        self.mean_u = np.mean(u_n)
        self.std_y = np.std(y_n)
        self.std_u = np.std(u_n)
        
        y_n = (y_n - self.mean_y) / self.std_y + np.random.normal(0, self.sigma, y_n.shape)
        y_vn = (y_vn - self.mean_y) / self.std_y + np.random.normal(0, self.sigma, y_vn.shape)
        u_n = (u_n - self.mean_u) / self.std_u + np.random.normal(0, self.sigma, u_n.shape)
        u_vn = (u_vn - self.mean_u) / self.std_u + np.random.normal(0, self.sigma, u_vn.shape)
        
        return u_n, y_n, u_vn, y_vn

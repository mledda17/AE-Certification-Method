class DynamicalSystem:
    def __init__(self, state_size, input_size, output_size, non_linear_input=False):
        self.non_linear_input = non_linear_input
        self.state_size = state_size
        self.input_size = input_size
        self.output_size = output_size

    def state_map(self, x, u):
        pass

    def output_map(self, xk):
        pass

    def system_dynamics(self, dim):
        pass

    def loop(self, x_k, duk):
        pass

    def prepare_dataset(self, training_size, validation_size):
        pass
import numpy as np
import keras
import time
import matplotlib.pyplot as plt
import tensorflow.python.keras.backend

from utilities.L21Regularizer import l21
from tensorflow.python.keras.backend import abs
from tensorflow.python.keras import layers
from keras.src.losses import mean_absolute_error
from keras.src.layers import Input, Dense
from keras.src.models import Model
from keras.src.regularizers import L2
from keras.src.constraints import *
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau


class Autoencoder:
    batch_size = 24
    state_size = -1
    non_linearity = ''
    n_neurons = -1
    validation_split = -1
    n_a = -1
    n_b = -1
    free_loading = 10

    def __init__(self, non_linearity='relu', n_neurons=30, n_layer=3, fit_horizon=5, group_lasso=True,
                 state_reduction=False, validation_split=0.05, state_size=-1, stride_len=10, output_window_len=2,
                 affine_struct=True, regularizer_weight=0.0005):
        self.non_linearity = non_linearity
        self.output_window_len = output_window_len
        self.state_size = state_size
        self.n_neurons = n_neurons
        self.state_reduction = state_reduction
        self.validation_split = validation_split
        self.stride_len = stride_len
        self.n_layer = n_layer
        self.model = None
        self.affine_struct = affine_struct
        self.max_range = fit_horizon
        self.regularizer_weight = regularizer_weight
        self.kernel_regularizer = L2(self.regularizer_weight)
        self.group_lasso = group_lasso
        self.shuffled_indexes = None
        self.constraint_on_input_hidden_layer = None
        self.encoder_network = None
        self.decoder_network = None
        self.bridge_network = None
        self.u = None
        self.n_u = None
        self.y = None
        self.n_y = None
        self.u_val = None
        self.y_val = None

        if group_lasso and regularizer_weight > 0.0:
            if state_reduction:
                self.input_layer_regularizer = l21(self.regularizer_weight, a=0, b=state_size)
            else:
                self.input_layer_regularizer = l21(self.regularizer_weight, a=stride_len, b=stride_len)
            self.kernel_regularizer = L2(0.0001)
            print("=> if GroupLasso is used,  l2 regularizer is set to 0.0001 in all bridge, encoder, and decoder")
        else:
            self.input_layer_regularizer = self.kernel_regularizer

    @staticmethod
    def mean_pred(y_true, y_pred):
        return mean_absolute_error(y_pred * 0, y_pred)

    def set_dataset(self, u, y, u_val, y_val):
        if u is not None:
            self.u = u.copy()
            self.n_u = u.shape[1]

        if y is not None:
            self.y = y.copy()
            self.n_y = y.shape[1]

        if u_val is not None:
            self.u_val = u_val.copy()

        if y_val is not None:
            self.y_val = y_val.copy()

    def encoder(self, future=0):
        inputs_u = Input(shape=(self.stride_len * self.n_u,))
        inputs_y = Input(shape=(self.stride_len * self.n_y,))
        input_concat = keras.layers.concatenate([inputs_y, inputs_u], name='concatIk')
        i_kr = self.kernel_regularizer

        if self.group_lasso and (not self.state_reduction):
            i_kr = self.input_layer_regularizer

        x = Dense(kernel_regularizer=i_kr,
                  kernel_constraint=self.constraint_on_input_hidden_layer,
                  units=self.n_neurons, activation=self.non_linearity,
                  name='enc0' + str(future))(input_concat)

        for i in range(0, self.n_layer - 1):
            x = Dense(use_bias=True,
                      kernel_regularizer=self.kernel_regularizer,
                      units=self.n_neurons,
                      activation=self.non_linearity,
                      name='enc' + str(i + 1) + str(future))(x)

        # x is the output of the Encoder Network
        x = Dense(kernel_regularizer=self.kernel_regularizer,
                  units=self.state_size, activation="linear",
                  name='encf' + str(future))(x)

        ann = Model(inputs=[inputs_y, inputs_u], outputs=[x])

        return ann

    def decoder(self, future=0):
        inputs_state = Input(shape=(self.state_size,))
        i_kr = self.kernel_regularizer

        if self.group_lasso and self.state_reduction:
            i_kr = self.input_layer_regularizer

        x = Dense(kernel_regularizer=i_kr,
                  kernel_constraint=self.constraint_on_input_hidden_layer,
                  units=self.n_neurons, activation=self.non_linearity,
                  name='dec0' + str(future))(inputs_state)

        for i in range(0, self.n_layer - 1):
            x = Dense(use_bias=True,
                      kernel_regularizer=self.kernel_regularizer,
                      units=self.n_neurons, activation=self.non_linearity,
                      name='dec' + str(i + 1) + str(future))(x)

        if self.affine_struct:
            x = Dense(units=self.output_window_len * self.state_size,
                      activation="linear", name='decf' + str(future))(x)
            x = keras.layers.Reshape((self.output_window_len, self.state_size))(x)
            out = keras.layers.dot([x, inputs_state], axes=-1)
        else:
            out = Dense(kernel_regularizer=self.kernel_regularizer,
                        units=self.output_window_len * self.n_y,
                        activation="linear", name='decf' + str(future))(x)
            x = out

        ann = Model(inputs=[inputs_state], outputs=[out, x])
        return ann

    def bridge(self, future=0):
        inputs_state = Input(shape=(self.state_size, ), name='inputs_state')
        inputs_novel_u = Input(shape=(self.n_u, ), name='inputs_novel_u')

        input_concat = keras.layers.concatenate([inputs_state, inputs_novel_u])
        i_kr = self.kernel_regularizer

        if self.group_lasso and self.state_reduction:
            i_kr = self.input_layer_regularizer
            print('Using group lasso on state')

        x = Dense(kernel_regularizer=i_kr,
                  kernel_constraint=self.constraint_on_input_hidden_layer,
                  units=self.n_neurons, activation=self.non_linearity,
                  name='bridge0' + str(future))(input_concat)

        for i in range(0, self.n_layer - 1):
            x = Dense(
                use_bias=True,
                kernel_regularizer=self.kernel_regularizer,
                units=self.n_neurons, activation=self.non_linearity,
                name='bridge' + str(i + 1) + str(future))(x)

        bias = Dense(
            kernel_regularizer=self.kernel_regularizer,
            units=self.state_size, activation="linear",
            name='bridgeBias' + str(future))(x)

        if self.affine_struct:
            ab_unshape = Dense(units=self.state_size * (self.state_size + self.n_u), activation="linear",
                               name='bridgef' + str(future))(x)
            ab = keras.layers.Reshape((self.state_size, self.state_size + self.n_u))(ab_unshape)
            out = keras.layers.dot([ab, input_concat], axes=-1)
            out = keras.layers.add([bias, out])
            ann = Model(inputs=[inputs_novel_u, inputs_state], outputs=[out, ab, bias])
        else:
            out = bias
            ann = Model(inputs=[inputs_novel_u, inputs_state], outputs=[out, x, bias])

        return ann

    def ann_model(self):
        inputs_y = Input(shape=((self.stride_len + self.max_range) * self.n_y,), name="input_y")
        inputs_u = Input(shape=((self.stride_len + self.max_range) * self.n_u,), name="input_u")

        stride_len = self.stride_len
        bridge_network = self.bridge()
        encoder_network = self.encoder()
        decoder_network = self.decoder()
        prediction_error_collection = []
        forward_error_collection = []
        forwarded_predicted_error_collection = []
        predicted_ok_collection = []  # O_k
        state_k_collection = []  # x_k
        max_range = self.max_range
        forwarded_state = None

        for k in range(0, max_range):
            information_vector_yk = keras.layers.Lambda(lambda x: x[:, k: stride_len + k])(inputs_y)
            information_vector_uk = keras.layers.Lambda(lambda x: x[:, k: stride_len + k])(inputs_u)
            i_target_k = keras.layers.Lambda(lambda x: x[:, stride_len + k - self.output_window_len + 1: stride_len + k + 1])(
                inputs_y)
            novel_uk = keras.layers.Lambda(lambda x: x[:, stride_len + k: stride_len + k + 1])(inputs_u)

            state_k = encoder_network([information_vector_yk, information_vector_uk])
            predicted_ok = decoder_network(state_k)[0]
            predicted_ok_collection += [predicted_ok]
            state_k_collection += [state_k]
            prediction_error_k = keras.layers.subtract([predicted_ok, i_target_k], name='oneStepDecoderError' + str(k))
            prediction_error_k = keras.layers.Lambda(lambda x: abs(x), output_shape=(1, 1))(prediction_error_k)
            prediction_error_collection += [prediction_error_k]
            if not (forwarded_state is None):
                forwarded_state_n = [bridge_network([novel_uk, state_k])[0]]

                for this_f in forwarded_state:
                    forward_error_k = [layers.subtract([state_k_elem, this_f]) for state_k_elem in state_k]
                    forward_error_k = keras.layers.Lambda(lambda x: abs(x), output_shape=(1, 1))(forward_error_k)
                    forward_error_collection += [forward_error_k]
                    forwarded_predicted_output_k = decoder_network(this_f)[0]
                    forwarded_predicted_error_k = keras.layers.subtract([forwarded_predicted_output_k, i_target_k])
                    forwarded_predicted_error_collection += [forwarded_predicted_error_k]
                    forwarded_state_n += [bridge_network([novel_uk, this_f])[0]]

                forwarded_state = forwarded_state_n
            else:
                forwarded_state = [bridge_network([novel_uk, state_k])[0]]

        one_step_ahead_pred_error = keras.layers.concatenate(prediction_error_collection, name='oneStepDecoderError')

        if len(forwarded_predicted_error_collection) > 1:
            forwarded_predicted_error = keras.layers.concatenate(forwarded_predicted_error_collection,
                                                                 name='multiStep_decodeError')
        else:
            forwarded_predicted_error = keras.layers.Lambda(lambda x: abs(x), name='multiStep_decodeError')(
                forwarded_predicted_error_collection[0])

        if len(forward_error_collection) > 1:
            forward_error = keras.layers.concatenate(forward_error_collection, name='forwardError')
        else:
            forward_error = keras.layers.Lambda(lambda x: abs(x), name='forwardError')(forward_error_collection[0])

        ann = Model(inputs=[inputs_y, inputs_u],
                    outputs=[one_step_ahead_pred_error,
                             forwarded_predicted_error, forward_error])
        return ann, encoder_network, decoder_network, bridge_network

    def compute_gradients(self, train_state_vector=None, train_input_vector=None, index=0):
        if (train_input_vector is None) or (train_state_vector is None):
            train_input_vector, train_output_vector = self.prepare_dataset(self.u, self.y)

        self.sess = tensorflow.python.keras.backend.get_session()
        gr = self.sess.run(self.gradientState,
                           feed_dict={self.model.input[0]: train_state_vector, self.model.input[1]: train_input_vector})
        return gr

    def prepare_dataset(self, u=None, y=None):
        if u is None:
            u = self.u
        if y is None:
            y = self.y

        pad = self.max_range - 2
        stride_len = self.stride_len + pad
        len_ds = u.shape[0]
        input_vector = np.zeros((len_ds - 2, self.n_u * (stride_len + 2)))
        output_vector = np.zeros((len_ds - 2, self.n_y * (stride_len + 2)))
        offset = self.stride_len + 1 + pad

        for i in range(offset, len_ds):
            regressor_state_inputs = np.ravel(u[i - stride_len - 1: i + 1])
            regressor_state_outputs = np.ravel(y[i - stride_len - 1: i + 1])

            input_vector[i - offset] = regressor_state_inputs.copy()
            output_vector[i - offset] = regressor_state_outputs.copy()

        return input_vector[:i - offset + 1].copy(), output_vector[:i - offset + 1].copy()

    def fit_model(self, shuffled: bool = True):
        tmp = self.train_model(shuffled, None, kfpe=0, kae_prediction=10, k_forward=.3)  # gamma, alpha, beta
        tmp = self.train_model(shuffled, tmp, 1, 0, 10)  # gamma, alpha, beta

    def train_model(self, shuffled: bool = True, tmp=None, kfpe=1, kae_prediction=1, k_forward=1):
        input_vector, output_vector = self.prepare_dataset()
        optimizer = keras.optimizers.Adam(learning_rate=0.002, amsgrad=True, clipvalue=0.5)

        if not (tmp is None):
            model = self.model
            encoder = self.encoder_network
            decoder = self.decoder_network
            bridge = self.bridge_network
            model.set_weights(tmp)
        else:
            model, encoder, decoder, bridge = self.ann_model()

        if self.shuffled_indexes is None:
            self.shuffled_indexes = np.random.permutation(list(range(0, np.shape(output_vector)[0])))

        shuffled_indexes = self.shuffled_indexes

        model.compile(optimizer=optimizer,
                      loss_weights={'multiStep_decodeError': kfpe,
                                    'oneStepDecoderError': kae_prediction,
                                    'forwardError': k_forward},
                      loss={"multiStep_decodeError": Autoencoder.mean_pred,
                            "oneStepDecoderError": Autoencoder.mean_pred,
                            'forwardError': Autoencoder.mean_pred}, )

        model.fit(
            {'input_y': output_vector[shuffled_indexes, :], 'input_u': input_vector[shuffled_indexes, :]},
            {'multiStep_decodeError': output_vector[:, 0:5] * 0, 'oneStepDecoderError': output_vector[:, 0:5] * 0,
             'forwardError': output_vector[:, 0:5] * 0},
            epochs=150,
            verbose=1,
            validation_split=self.validation_split,
            shuffle=shuffled,
            batch_size=self.batch_size,
            callbacks=[ReduceLROnPlateau(factor=0.3, min_delta=0.001, patience=3),
                       EarlyStopping(patience=8, min_delta=0.001, monitor='val_loss')]
        )

        self.model = model
        self.encoder_network = encoder
        self.decoder_network = decoder
        self.bridge_network = bridge
        tmp = model.get_weights()
        return tmp

    def evaluate_network(self, u_val, y_val):
        train_state_vector, train_input_vector, train_output_vector = self.prepare_dataset(u_val, y_val)
        t = time.time()
        fitted_y = self.model.predict([train_state_vector, train_input_vector])
        elapsed = time.time() - t

        return fitted_y, train_output_vector, elapsed

    def validate_model(self, plot: bool = True):
        fitted_y, train_output_vector, elapsed = self.evaluate_network(self.u_val, self.y_val)
        fitted_y = fitted_y[0]
        if plot:
            plt.figure(figsize=(7, 7))
            plt.plot(fitted_y)
            plt.plot(train_output_vector)
            plt.show()
        return fitted_y, train_output_vector, elapsed

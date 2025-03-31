import numpy as np
from keras._tf_keras.keras import backend as K
from keras._tf_keras.keras.regularizers import Regularizer

class l21(Regularizer):
    """
    Regularizer for L21 regularization.
    Arguments:
        C: Float; L21 regularization factor.
    """

    def __init__(self, C=0.0, a=0, b=0, bias=0.000):
        self.a = a
        self.b = b
        C = K.cast_to_floatx(C)
        self.C = (bias + C) * np.square(np.concatenate([0 +
                                                        a -
                                                        np.array(range(0,a)),
                                                        0 +
                                                        b-np.array(range(0,b))
                                                        ]))
        print("****Squared weigthing enabled****")
        print(self.C)


    def __call__(self, x):
        print(x)
        w = K.sum(K.abs(x), 1)
        print(str(w))
        w = w[0 : self.a + self.b]
        print(w * self.C)
        return K.sum(w * self.C)


    def get_config(self):
        return {'C': float(self.l1)}
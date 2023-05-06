"""
Neural Network combining LSTM and CNN architecture
"""

from __future__ import annotations
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Flatten, Dropout, Layer, Concatenate
from keras.models import Model

from ..forecast.naive import Forecast
from ..forecast.cnn import CNN_func
from ..forecast.lstm import LSTM_func


class CLSTM(Forecast):
    def __init__(self, h: int) -> None:
        super().__init__(h)

    def predict(self) -> np.array:
        return super().predict()

    def fit(self) -> Forecast:
        ...


class CLSTMModel:
    """
    Connects LSTM and CNN parts of model.
    """

    def __init__(self, 
                 nodes: list = [32, 16], 
                 activation="selu", 
                 **kwargs) -> None:
        self.model = CLSTM_func(
            nodes=nodes,
            activation=activation
        )
        
        

    def fit(self):
        ...
        
    def predict(self):
        ...


def CLSTM_func(nodes: list[int], activation: str):
    inp1, out1 = LSTM_func()
    inp2, out2 = CNN_func()
    concat_1 = tf.keras.layers.concatenate([out1, out2])
    dense_1 = Dense(nodes[0], activation=activation)(concat_1)
    dense_2 = Dense(nodes[1], activation=activation)(dense_1)
    dense_3 = Dense(1)(dense_2)

    model = Model(inputs=[inp1, inp2], outputs=dense_3)

    return model

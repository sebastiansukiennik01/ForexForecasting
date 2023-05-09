"""
Neural Network combining LSTM and CNN architecture
"""

from __future__ import annotations
import tensorflow as tf
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import backend as K
from keras.layers import Input, Dense, Flatten, Dropout, Layer, Concatenate
from keras.optimizers import Adam
from keras.losses import MeanAbsolutePercentageError, MeanAbsoluteError
from keras.models import Model

from ..forecast.naive import Forecast
from ..forecast.cnn import CNN_func, datagen, testgen
from ..forecast.lstm import LSTM_func, CheckpoinCallback


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
                 activation: str = "selu",
                 target_col: str = "target_value",
                 seq_len: int = 10,
                 **kwargs) -> None:
        self.model = CLSTM_func(
            nodes=nodes,
            activation=activation,
            seq_len=seq_len,
            **kwargs
        )
        self.target_col = target_col
        self.seq_len = seq_len

    def compile(self,
                optimizer=Adam(learning_rate=0.01),
                loss=MeanAbsolutePercentageError(),
                metrics=[MeanAbsoluteError()],
                **kwargs):
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs)
    
    def fit(self,
            data,
            epochs=5,
            steps_per_epoch=100,
            batch_size=128,
            callbacks=None,
            verbose=1,
            **kwargs):
        """
        Overrides built in fit method with custom deafault arguments
        """
        if not callbacks:
            callbacks = self._default_callbacks()
        self.model.fit(
            _CLSTM_datagen(df=data,
                           seq_len=self.seq_len,
                           batch_size=batch_size,
                           targetcol=[self.target_col],
                           kind="train",
                           **kwargs),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Returns prediction data
        """
        pred = self.model.predict(
            _CLSTM_testgen(data,
                    seq_len=self.seq_len,
                    targetcol=self.target_col))
        
        return pred.flatten()
    
    @staticmethod
    def load(path: str) -> Model:
        """
        Loads specified model
        args:
            path : path to saved model, can be specified as 'best'
        returns: tensorflow Mdoel
        """
        if path == "best":
            models_saved = {m: float(m.split('_')[1].replace('.h5', '')) 
                            for m in os.listdir('models/clstm/')}
            best_model = sorted(models_saved.items(), key=lambda x: x[1])[0]
            return tf.keras.models.load_model(f'models/clstm/{best_model[0]}')
            
        
        
        
    def _default_callbacks(self) -> list[Callback]:
        """
        Returns list of deafault callbacks
        """
        callbacks = [
            CheckpoinCallback(metric='mean_absolute_error')
        ]
        return callbacks


def CLSTM_func(nodes: list[int], activation: str, seq_len: int, **kwargs):
    """
    Creates and returns CLSMT model created with functional API
    args:
        nodes : number of nodes in consecutive Dense layers
        activation : activation funciton in Dense layers
        seq_len : sequence length used in CNN module
    returns : tensorflow functional CNN-LSTM model
    """
    lstm_kwargs = _filter_kwargs(kwargs=kwargs, starts_with='lstm_')
    cnn_kwargs = _filter_kwargs(kwargs=kwargs, starts_with='cnn_')
    
    print(lstm_kwargs)
    print(cnn_kwargs)
    
    inp1, out1 = LSTM_func(functional=True, **lstm_kwargs)
    inp2, out2 = CNN_func(**cnn_kwargs)
    
    concat_1 = tf.keras.layers.concatenate([out1, out2])
    dense_1 = Dense(nodes[0], activation=activation)(concat_1)
    dense_2 = Dense(nodes[1], activation=activation)(dense_1)
    dense_3 = Dense(1)(dense_2)

    model = Model(inputs=[inp1, inp2], outputs=dense_3)

    return model

def _filter_kwargs(kwargs: dict, starts_with: str) -> dict:
    """
    Filters out and returns only kwargs starting with specified prefix.
    args:
        kwargs : dictionary like keyword arguments
        starts_wtih : prefix string
    returns : dictionary with keys only starting with provided prefix
    """
    return {k.replace(starts_with, ''): v 
            for k, v in kwargs.items() if str(k).startswith(starts_with)}


def _CLSTM_datagen(df: pd.DataFrame, seq_len: int, batch_size, targetcol: list, kind, **kwargs):
    """
    Takes df, seq_len and returns 2 inputs and one output label
    args:
        df : dataframe with data
        seq_len : length of sequence
        batch_size : integar size of batch
        targetcol : list of target columns
        kind : train/valid for depending on type of dataset needed
    returns : inputs, output tuple
    """
    # cnn input has to be last batch_size-elements of seq_len length
    # lstm input can be the same?
    while True:
        clstm_x, clstm_y = next(datagen(df=df, seq_len=seq_len, batch_size=batch_size, targetcol=targetcol, kind=kind))
    
        yield [clstm_x[..., 0], clstm_x], clstm_y
        

def _CLSTM_testgen(data: pd.DataFrame, seq_len: int, targetcol: list):
    """ Return array of all test samples """
    clstm_x, _ = testgen(data=data, seq_len=seq_len, targetcol=targetcol)
    
    return clstm_x[..., 0], clstm_x
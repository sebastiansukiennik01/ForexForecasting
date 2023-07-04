"""
Neural Network combining LSTM and CNN architecture
"""

from __future__ import annotations
import tensorflow as tf
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import MeanAbsolutePercentageError, MeanAbsoluteError 
from keras.metrics import RootMeanSquaredError
from keras.callbacks import Callback
from keras.models import Model

from ..forecast.naive import Forecast
from ..forecast.cnn import CNN_func, datagen, testgen
from ..forecast.lstm import LSTM_func, CheckpoinCallback, EarlyStopping


class CLSTM(Forecast):
    def __init__(self, **kwargs) -> None:
        self.model = kwargs.pop('model', None)

    def predict(self, test_X: np.ndarray, *args, **kwargs) -> np.array:
        """
        Generates prediction for provided test set.
        """
        # if only label column provided, raise ValueError
        if self.model.model.input_shape[0][2] != test_X.shape[1]:
            raise ValueError
        # if seq_len required is greater than number of observations
        # raise TypeError so the next provided 'test_X' is actual 'train_X' extended with 'test_X'
        if self.model.seq_len > test_X.shape[0]:
            raise TypeError
        
        self.forecast = self.model.predict(test_X)
        
        return self.forecast.flatten()

    def fit(self, train_X: np.ndarray, train_y: np.ndarray, **kwargs) -> Forecast:
        """
        Declare model, compile, and fit.
        """
        # declare model
        if not self.model:
            self.model = CLSTMModel(**kwargs)
        self.model.compile()
        
        train = pd.concat([train_X, train_y], axis=1)
        self.model.fit(
            data=train,
            **kwargs
        )
        
        return self
    
    def load(self, path: str = "best") -> CLSTM:
        """
        Loads model from specified path. If best, than CLSMT model with lowest loss is loaded.
        """
        best_mod = CLSTMModel.load(path)
        self.model = best_mod
        
        return self
    
    def get_residuals(self, train_X: np.ndarray, train_y: np.ndarray, **kwargs):
        """ Calculates insample residuals """
        
        '''
        # FIT EVERY 5 VALUES INSAMPLE
        self.h = self.h if hasattr(self, 'h') else 5
        # first_idx = self.model.seq_len if hasattr(self.model, 'seq_len') and self.model.seq_len else 10
        first_idx = 128+10
        pred_y = list(train_y.iloc[:first_idx].values.flatten())
        
        [
            pred_y.extend(self._fit_predict(train_X, train_y, i))
            for i in range(first_idx, train_y.shape[0], self.h)
        ]'''
        
        first_idx = self.model.seq_len
        pred_y = np.concatenate([train_y.values.flatten()[:first_idx], 
                                 self.predict(test_X=train_X)])
        
        # calc resid and insample prediction
        self.pred_insample = pred_y[:len(train_y)]
        self.residuals = np.subtract(train_y.values.flatten(), self.pred_insample)

    def _fit_predict(self, train_X: pd.DataFrame, train_y: pd.DataFrame, i: int) -> np.ndarray:
        """ Fit model than return prediction """
        temp_train_X = train_X.iloc[:i]
        temp_train_y = train_y.iloc[:i]
        temp_test_X = train_X.iloc[i:i+self.h]
        print(temp_train_X.shape)
        
        self.fit(train_X=temp_train_X, train_y=temp_train_y)
        
        if self.model.seq_len > temp_test_X.shape[0]:
            n = self.model.seq_len + self.h
            temp_test_X = pd.concat([train_X, temp_test_X], axis=0).iloc[-n:]
            
        return self.predict(test_X=temp_test_X)
        


class CLSTMModel:
    """
    Connects LSTM and CNN parts of model.
    """

    def __init__(self, 
                 nodes: list = [128, 64], 
                 activation: str = "selu",
                 target_col: str = "target_value",
                 **kwargs) -> None:
        
        self.seq_len = kwargs.get('seq_len', 120)
        self.target_col = target_col
        self.model = CLSTM_func(
            nodes=nodes,
            activation=activation,
            # seq_len=self.seq_len,
            **kwargs
        )

    def compile(self,
                optimizer=Adam(learning_rate=0.0001, clipnorm=1.0, clipvalue=1),
                loss=MeanAbsolutePercentageError(),
                metrics=[RootMeanSquaredError()],
                **kwargs):
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs)
    
    def fit(self,
            data,
            epochs=10,
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
                        #    seq_len=self.seq_len,
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
        
        clstm_best_model = CLSTMModel()
        models_saved = os.listdir('models/clstm/')
        if '.DS_Store' in models_saved:
            models_saved.remove('.DS_Store')
        if path == "best":
            models_saved = {m: float(m.split('_')[1].replace('.h5', '')) 
                            for m in models_saved}
            best_model = sorted(models_saved.items(), key=lambda x: x[1])[0]
            clstm_best_model.model = tf.keras.models.load_model(f'models/clstm/{best_model[0]}')
            print(f"Loading model: {best_model[0]}..")
        else:
            clstm_best_model.model = tf.keras.models.load_model(f'models/clstm/{path}')
            print(f"Loading model: {path}..")
            
        return clstm_best_model
        
    def _default_callbacks(self) -> list[Callback]:
        """
        Returns list of deafault callbacks
        """
        callbacks = [
            CheckpoinCallback(metric='loss'),
            EarlyStopping(min_delta=0, patience=5, mode="min")
        ]
        return callbacks


def CLSTM_func(nodes: list[int] = [32, 64], activation: str = "selu", **kwargs):
    """
    Creates and returns CLSMT model created with functional API
    args:
        nodes : number of nodes in consecutive Dense layers
        activation : activation funciton in Dense layers
        seq_len : sequence length used in CNN module
    returns : tensorflow functional CNN-LSTM model
    """
    seq_len = kwargs.pop('seq_len', 120)
    lstm_kwargs = _filter_kwargs(kwargs=kwargs, starts_with='lstm_')
    cnn_kwargs = _filter_kwargs(kwargs=kwargs, starts_with='cnn_')
    lstm_kwargs['seq_len'] = seq_len
    cnn_kwargs['seq_len'] = seq_len
    
    print(f"lstm_kwargs: {lstm_kwargs}")
    print(f"cnn_kwargs: {cnn_kwargs}")
    print(nodes, activation)
    
    inp1, out1 = LSTM_func(functional=True, **lstm_kwargs)
    inp2, out2 = CNN_func(functional=True, **cnn_kwargs)
    
    concat_1 = tf.keras.layers.concatenate([out1, out2])
    dense_1 = Dense(nodes[0], activation=activation)(concat_1)
    # batch_1 = tf.keras.layers.BatchNormalization(axis=-1)(dense_1) # TODO for future testing
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


def _CLSTM_datagen(df: pd.DataFrame, batch_size, targetcol: list, kind, **kwargs):
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
    seq_len = kwargs.get('seq_len', 120)
    # cnn input has to be last batch_size-elements of seq_len length
    # lstm input can be the same?
    while True:
        clstm_x, clstm_y = next(datagen(df=df, seq_len=seq_len, batch_size=batch_size, targetcol=targetcol, kind=kind))
    
        yield [clstm_x[..., 0], clstm_x], clstm_y
        

def _CLSTM_testgen(data: pd.DataFrame, seq_len: int, targetcol: list):
    """ Return array of all test samples """
    clstm_x = testgen(data=data, seq_len=seq_len, targetcol=targetcol)
    
    return clstm_x[..., 0], clstm_x
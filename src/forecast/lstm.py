"""
Neural Network combining LSTM architecture
"""

from __future__ import annotations

import tensorflow as tf
import numpy as np
import pandas as pd
import datetime as dt
import os
from keras import backend as K
from keras.layers import Dense, Lambda, Bidirectional, LSTM, Input, Layer, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MeanAbsolutePercentageError
from keras.metrics import accuracy, MeanSquaredError, MeanAbsoluteError
from keras.callbacks import ModelCheckpoint, Callback

from ..forecast.naive import Forecast
from ..forecast.cnn import datagen, testgen


class LSTM_:
    def __init__(
        self,
        nodes: int = [64, 32, 32, 32],
        activation: str = "selu",
        target_col: str = "target_value"
    ):
        self.model = LSTM_func(
            nodes=nodes,
            activation=activation
        )
        self.nodes = nodes
        self.activation = activation
        self.target_col = target_col
        
    def compile(self,
                optimizer=Adam(learning_rate=0.0001, decay=1e-3),
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
        # self.model.fit(
        #     x=data.drop(columns=[self.target_col]),
        #     y=data[self.target_col],
        #     epochs=epochs,
        #     steps_per_epoch=steps_per_epoch,
        #     batch_size=batch_size,
        #     callbacks=callbacks,
        #     verbose=verbose
        # )
        self.model.fit(
            _lstm_datagen(df=data, 
                    seq_len=10, 
                    batch_size=batch_size, 
                    targetcol=[self.target_col],
                    kind="train"),
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
            testgen(data, 
                    seq_len=10, 
                    targetcol='target_value')
        )
        return pred.flatten()
        

    def _default_callbacks(self) -> list[Callback]:
        """
        Returns list of deafault callbacks
        """
        callbacks = [
            CheckpoinCallback(metric='loss')
        ]
        return callbacks
    
    
def _lstm_datagen(df: pd.DataFrame, seq_len: int, batch_size, targetcol: list, kind, **kwargs):
    """
    As a generator to produce samples for Keras model

    args:
        df : datframe with data for model
        seq_len : length of sequence
        batch_size : integar size of batch
        targetcol : list of target columns
        kind : train/valid for depending on type of dataset needed
    """
    batch = []
    i = seq_len
    while True:
        # Pick one dataframe from the pool

        input_cols = [c for c in df.columns if c not in targetcol]
        index = df.index
        split = len(index)
        
        if kind == "train":
            index = index[:split]  # range for the training set
        elif kind == "valid":
            index = index[split:]  # range for the validation set
        
        
        # Pick one position, then clip a sequence length
        while True:
            # NEW 
            frame = df.iloc[i - seq_len : i]
            t = df.index[i - 1]
            i += 1
            batch.append([frame[input_cols].values, df.loc[t, targetcol]])
            
            break
        # if we get enough for a batch, dispatch
        
        if len(batch) == batch_size:
            # check if next iteration can be done, if not reset to beginning
            if i + batch_size >= df.shape[0]:
                print("\n\n### Loop from beggining!\n")
                i = seq_len
                
            X, y = zip(*batch)
            X, y = np.array(X), np.array(y)
            yield X, y
            batch = []


def LSTM_func(
    functional: bool = False,
    **kwargs
):
    """
    Returns LSTM sequential (on default) model.
    args:
        nodes : number of node in each layer
        activation : activation function for each layer
        functional : weather to return functionl model, tuple of input and outut layers
    returns : either sequential or inout output of functional model
    """
    nodes = kwargs.get('nodes', [64, 32, 32, 32])
    activation = kwargs.get('activation', 'tanh')
    
    model = tf.keras.models.Sequential([
        Lambda(lambda x: x),
        Bidirectional(LSTM(nodes[0], return_sequences=True)),
        Bidirectional(LSTM(nodes[1], return_sequences=True)),
        Bidirectional(LSTM(nodes[2])),
        Dense(nodes[3]),
        Dense(1, activation=activation),
    ])
    if functional:
        return _convert_model_to_functional(seq_model=model, **kwargs)

    return model


def _convert_model_to_functional(seq_model: Model, **kwargs) -> tuple[Layer]:
    """
    Converts sequential model to functional one. Returns its input and output layers
    args:
        model : tensorflow sequential model
    returns: input, output layers of model
    """
    seq_len = kwargs.get('seq_len', 10)
    
    # input_layer = Input(batch_shape=seq_model.layers[0].input_shape)
    input_layer = Input(batch_shape=(128, seq_len, 49))

    prev_layer = input_layer
    for layer in seq_model.layers:
        layer._inbound_nodes = []
        prev_layer = layer(prev_layer)

    return input_layer, prev_layer


class CheckpoinCallback(tf.keras.callbacks.Callback):
    """
    Custom Callback chceckpoint
    """
    
    def __init__(self, metric: str):
        super().__init__()
        self.metric = metric
    
    def on_epoch_end(self, epoch, logs):
        """
        On every epoch end checks if current model is better than the best save
        inside modesl/lstm/ directory. If it is, the current model is saved.
        """
        mae = round(logs.get(self.metric), 4)
        
        saved_models = os.listdir('models/clstm/')    
        try:
            saved_models.remove('.DS_Store')        
        except ValueError:
            pass
        
        models_results = [float(s.split('_')[1].replace('.h5', '')) for s in saved_models]
        if mae < min(models_results):
            print(f"\nNew lowest {self.metric}: {mae}")
            checkpoint_path = f"models/clstm/{self.metric}_{mae}.h5"
            self.model.save(checkpoint_path)
            
class EarlyStopping(tf.keras.callbacks.EarlyStopping):
    """
    Custom early stoping callback
    """
    def __init__(self, monitor: str = "loss", min_delta: int = 0.1, patience: int = 2, mode: str = "auto", verbose: int = 1, restore_best_weights: bool = True):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            mode=mode,
            verbose=verbose,
            restore_best_weights=restore_best_weights
        )
        
    def on_epoch_end(self, epoch, logs=None):
        """
        On every epoch end checks if current model made significant increase in tracked
        metrics, if not stopp training process early.
        """
        return super().on_epoch_end(epoch, logs)
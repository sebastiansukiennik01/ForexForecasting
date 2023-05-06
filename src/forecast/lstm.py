"""
Neural Network combining LSTM architecture
"""

from __future__ import annotations

import tensorflow as tf
import numpy as np
import datetime as dt
import os
from keras import backend as K
from keras.layers import Dense, Lambda, Bidirectional, LSTM, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MeanAbsolutePercentageError
from keras.metrics import accuracy, MeanSquaredError, MeanAbsoluteError
from keras.callbacks import ModelCheckpoint, Callback

from ..forecast.naive import Forecast
from ..forecast.cnn import datagen


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
                optimizer=Adam(),
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
            x=data.drop(columns=[self.target_col]),
            y=data[self.target_col],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
    def predict(self, data):
        """
        Returns prediction data
        """
        try:
            data = data.drop(columns=[self.target_col])
        except KeyError:
            pass
        pred = self.model.predict(
            data
        )
        return pred.flatten()
        

    def _default_callbacks(self) -> list[Callback]:
        """
        Returns list of deafault callbacks
        """
        callbacks = [
            CheckpoinCallback(metric='mean_absolute_error')
        ]
        return callbacks

def LSTM_func(
    nodes: str, 
    activation: str
):
    model = tf.keras.models.Sequential([
        Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
        Bidirectional(LSTM(nodes[0], return_sequences=True)),
        Bidirectional(LSTM(nodes[1], return_sequences=True)),
        Bidirectional(LSTM(nodes[2])),
        Dense(nodes[3]),
        Dense(1, activation=activation),
    ])

    return model

class CheckpoinCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, metric: str):
        super().__init__()
        self.metric = metric
    
    def on_epoch_end(self, epoch, logs):
        mae = logs.get(self.metric)
        
        saved_models = os.listdir('models/lstm/')
        saved_models.remove('.DS_Store')        
        models_results = [float(s.split('_')[1].replace('.h5', '')) for s in saved_models]
        if mae < min(models_results):
          print(f"\nNew lowest {self.metric}: {mae}")
          checkpoint_path = f"models/lstm/mae_{mae}.h5"
          self.model.save(checkpoint_path)
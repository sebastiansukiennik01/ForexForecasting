"""
Module for CNN part of final neural network model
"""

# from __future__ import annotations

import pandas as pd
import numpy as np
import random
import datetime as dt
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Flatten, Dropout, Layer, Conv1D, MaxPool1D, BatchNormalization
from keras.models import Model

import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    Dense,
    Conv1D,
    MaxPool1D,
)
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import accuracy
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback

from sklearn.preprocessing import StandardScaler


TRAIN_TEST_CUTOFF = "2016-04-21"
TRAIN_VALID_RATIO = 0.75


def f1macro(y_true: list, y_pred: list):
    f_pos = f1_m(y_true, y_pred)
    # negative version of the data and prediction
    f_neg = f1_m(1 - y_true, 1 - K.clip(y_pred, 0, 1))
    return (f_pos + f_neg) / 2

    
class CNN_:
    def __init__(
        self,
        filters: int = 8,
        kernel_size: int = 4,
        pool_size: int = 2,
        activation: str = "selu",
        seq_len: int = 10,
        n_features: int = 49,
        target_col: str = "target_direction"
    ):
        self.model = CNN_func(
            filters=filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            activation=activation,
            seq_len=seq_len,
            n_features=n_features
        )
        self.seq_len = seq_len
        self.n_features = n_features
        self.target_col = target_col
    
    def compile(self,
                optimizer=Adam(),
                loss=BinaryCrossentropy(),
                metrics=[accuracy, f1macro],
                **kwargs):
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs)
        
    def fit(self,
            data,
            seq_len=10,
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
        #TODO FIX DATAGEN DIVIDING DATA
        self.model.fit(
            datagen(df=data, seq_len=seq_len, batch_size=batch_size, targetcol=["target_direction"], kind="train", **kwargs),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
    
    def predict(self, data):
        """
        Returns prediction
        """
        pred = self.model.predict(
            testgen(data,
                    seq_len=self.seq_len,
                    targetcol=self.target_col)[0])
        
        return  map(int, pred.flatten()>0.5)
    
    def _default_callbacks(self) -> list[Callback]:
        """
        Returns list of deafault callbacks
        """
        checkpoint_path = f"./models/cnn/ccn-{dt.datetime.now().strftime('%m%d_%H%M%S')}-" + "{epoch}.h5"
        callbacks = [
            ModelCheckpoint(checkpoint_path,
                            monitor='f1macro', 
                            mode="max",
                            verbose=0,
                            save_best_only=True, 
                            save_weights_only=False, 
                            save_freq="epoch")
        ]
        return callbacks
    
    
def CNN_func(n_features: int = 49, **kwargs):
    """
    Creates and returns CNN model created with functional API
    args:
        filters : number of filter in each kernel
        kernel_size : size of square kernels
        pool_size : size of pooling layers
        activation : activation function
        seq_len : sequence length used in each step
    returns : tensorflow functional CNN model 
    """
    
    filters = kwargs.pop('filters', 8)
    kernel_size = kwargs.pop('kernel_size', 4)
    pool_size = kwargs.pop('pool_size', 2)
    activation = kwargs.pop('activation', 'selu')
    seq_len = kwargs.pop('seq_len', 10)
    
    inp = tf.keras.layers.Input(shape=(seq_len, n_features, 1))
    conv1D_0 = Conv2D(filters, kernel_size, activation=activation, padding="same")(inp)
    # batch_1 = BatchNormalization(axis=-1)(conv1D_0) # TODO for future testing
    max_pool_0 = MaxPool2D((pool_size, pool_size))(conv1D_0)
    conv1D_1 = Conv2D(filters, kernel_size, activation=activation, padding="same")(max_pool_0)
    max_pool_1 = MaxPool2D((pool_size, pool_size))(conv1D_1)
    conv1D_2 = Conv2D(filters, kernel_size, activation=activation, padding="same")(max_pool_1)
    max_pool_2 = MaxPool2D((pool_size, pool_size))(conv1D_2)
    
    flatten = Flatten()(max_pool_2)
    dense_0 = Dense(32, activation=activation)(flatten)
    dense_1 = Dense(1, activation='sigmoid')(dense_0)
    
    # model = Model(inputs=inp, outputs=dense_1)
    
    return inp, dense_1


def datagen(df: pd.DataFrame, seq_len: int, batch_size, targetcol: list, kind, **kwargs):
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
            """t = random.choice(index)  # pick one time step
            n = (df.index == t).argmax()  # find its position in the dataframe
            if n - seq_len + 1 < 0:
                continue  # can't get enough data for one sequence length
            frame = df.iloc[n - seq_len + 1 : n + 1]
            batch.append([frame[input_cols].values, df.loc[t, targetcol]])
            """
            # NEW 
            frame = df.iloc[i - seq_len : i]
            t = df.index[i - 1]
            i += 1
            batch.append([frame[input_cols].values, df.loc[t, targetcol]])
            
            break
        # if we get enough for a batch, dispatch
        if len(batch) == batch_size:
            X, y = zip(*batch)
            X, y = np.expand_dims(np.array(X), 3), np.array(y)
            yield X, y
            batch = []


def testgen(data: pd.DataFrame, seq_len: int, targetcol: list):
    "Return array of all test samples"
    batch = []

    input_cols = [c for c in data.columns if c not in targetcol]
    # find the start of test sample
    # t = data.index[data.index >= TRAIN_TEST_CUTOFF][0]
    # n = (data.index == t).argmax()
    for i in range(seq_len + 1, len(data) + 1):
        frame = data.iloc[i - seq_len : i]
        batch.append([frame[input_cols].values, frame[targetcol].values[-1]])
    X, y = zip(*batch)
    return np.expand_dims(np.array(X), 3), np.array(y)


def scale_inputs(data: pd.DataFrame, targetcol: list) -> pd.DataFrame:
    """
    Normalizes inputs
    """
    labels = [c for c in data.columns if c not in targetcol]
    scaler = StandardScaler().fit(data.loc[:, labels])
    data.loc[:, labels] = scaler.transform(data.loc[:, labels])

    return data


def recall_m(y_true: list, y_pred: list):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_loss(y_true: list, y_pred: list) -> list:
    tf.reduce_mean()


def precision_m(y_true: list, y_pred: list):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true: list, y_pred: list):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def f1macro(y_true: list, y_pred: list):
    f_pos = f1_m(y_true, y_pred)
    # negative version of the data and prediction
    f_neg = f1_m(1 - y_true, 1 - K.clip(y_pred, 0, 1))
    return (f_pos + f_neg) / 2

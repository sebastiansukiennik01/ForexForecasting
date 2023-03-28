"""
Module for CNN part of final neural network model
"""

import pandas as pd
import numpy as np
import random

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import StandardScaler



TRAIN_TEST_CUTOFF = '2016-04-21'
TRAIN_VALID_RATIO = 0.75
 
def datagen(df: pd.DataFrame, seq_len: int, batch_size, targetcol: list, kind):
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
    while True:
        # Pick one dataframe from the pool
        
        input_cols = [c for c in df.columns if c not in targetcol]
        index = df.index[df.index < TRAIN_TEST_CUTOFF]
        split = int(len(index) * TRAIN_VALID_RATIO)
        if kind == 'train':
            index = index[:split]   # range for the training set
        elif kind == 'valid':
            index = index[split:]   # range for the validation set
        # Pick one position, then clip a sequence length
        while True:
            t = random.choice(index)      # pick one time step
            n = (df.index == t).argmax()  # find its position in the dataframe
            if n-seq_len+1 < 0:
                continue # can't get enough data for one sequence length
            frame = df.iloc[n-seq_len+1:n+1]
            batch.append([frame[input_cols].values, df.loc[t, targetcol]])
            break
        # if we get enough for a batch, dispatch
        if len(batch) == batch_size:
            X, y = zip(*batch)
            X, y = np.expand_dims(np.array(X), 3), np.array(y)
            yield X, y
            batch = []
            
            
def cnnpred_2d(seq_len=60, n_features=82, n_filters=(8,8,8), droprate=0.1):
    "2D-CNNpred model according to the paper"
    model = Sequential([
        Input(shape=(seq_len, n_features, 1)),
        Conv2D(n_filters[0], kernel_size=(1, n_features), activation="relu"),
        Conv2D(n_filters[1], kernel_size=(3,1), activation="relu"),
        MaxPool2D(pool_size=(2,1)),
        Conv2D(n_filters[2], kernel_size=(3,1), activation="relu"),
        MaxPool2D(pool_size=(2,1)),
        Flatten(),
        Dropout(droprate),
        Dense(1, activation="sigmoid")
    ])
    return model


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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
 
def f1macro(y_true: list, y_pred: list):
    f_pos = f1_m(y_true, y_pred)
    # negative version of the data and prediction
    f_neg = f1_m(1-y_true, 1-K.clip(y_pred,0,1))
    return (f_pos + f_neg)/2


def testgen(data: pd.DataFrame, seq_len: int, targetcol: list):
    "Return array of all test samples"
    batch = []
    
    input_cols = [c for c in data.columns if c not in targetcol]
    # find the start of test sample
    t = data.index[data.index >= TRAIN_TEST_CUTOFF][0]
    n = (data.index == t).argmax()
    for i in range(n+1, len(data)+1):
        frame = data.iloc[i-seq_len:i]
        batch.append([frame[input_cols].values, frame[targetcol].values[-1]])
    X, y = zip(*batch)
    return np.expand_dims(np.array(X),3), np.array(y)
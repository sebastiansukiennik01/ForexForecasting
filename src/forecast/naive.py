"""
Naive methods forecasting
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
import copy
from abc import ABC, abstractmethod


class Forecast(ABC):
    """
    Abstract Naive class is base class for all naive forecasting classes.
    """

    _forecast: np.array
    _y: np.array
    _values: np.array
    _resid: np.ndarray

    def __init__(self, h: int) -> None:
        """
        Base initializer for Naive classes.
        args:
            values : numpy array of time series values
            h : forecast horizon
        """
        self.h = h

    @property
    def forecast(self) -> np.array:
        return self._forecast

    @forecast.setter
    def forecast(self, forecast: np.array) -> None:
        self._forecast = forecast

    @property
    def values(self) -> np.array:
        return self._values

    @values.setter
    def values(self, values: np.array) -> None:
        if isinstance(values, np.ndarray):
            self._values = values
        elif isinstance(values, (pd.DataFrame, pd.Series)):
            self._values = values.values
        else:
            self._values = np.array(values)

    @property
    def y(self) -> np.array:
        return self._y

    @y.setter
    def y(self, y: np.array) -> None:
        self._y = y
        
    @property
    def forecast(self) -> np.array:
        return self._resid

    @forecast.setter
    def forecast(self, resid: np.array) -> None:
        self._resid = resid

    @abstractmethod
    def predict(self) -> np.array:
        ...

    def fit(self, train_X: np.ndarray, train_y: np.ndarray, **kwargs) -> Forecast:
        """For compatibility"""
        ...
        # first_idx = self.k if self.k else 5
        # pred_y = np.array(train_y[:first_idx])
        
        # for i in range(first_idx, train_y.shape[0], self.h):
        #     pred_y = np.append(pred_y, self.predict(train_y[:i]))
            
        # # calc resid
        # self.pred_insample = pred_y[:len(train_y)]
        # self.residuals = np.subtract(train_y.values.flatten(), self.pred_insample)

    def get_residuals(self, train_y: np.ndarray, **kwargs):
        """ Calculates insample residuals """
        print("\n CHANGED K FROM 5 TO 200!!! ")
        first_idx = self.k if hasattr(self, 'k') and self.k else 5
        pred_y = np.array(train_y[:first_idx])
        
        for i in range(first_idx, train_y.shape[0], self.h):
            print(f"{i}/{train_y.shape[0]}")
            pred_y = np.append(pred_y, self.predict(train_y[:i]))
            
        # calc resid and insample prediction
        self.pred_insample = pred_y[:len(train_y)]
        self.residuals = np.subtract(train_y.values.flatten(), self.pred_insample)

@dataclass
class NaiveAVG(Forecast):
    _forecast: np.array
    _y: np.array

    def __init__(self, h: int = 1, T: int = 1) -> None:
        """
        Initializer for Naive Average class.
        args:
            values : numpy array of time series values
            h : forecast horizon
            T : number of periods from which to calculate average
        """
        super().__init__(h)
        self._T = T

    @property
    def T(self) -> int:
        return self._T

    @T.setter
    def T(self, t: int) -> None:
        assert (
            len(self.values) >= t
        ), f"Tries to calculate average from {t} periods \
            but time series has only {len(self.values)} elements"
        self._T = t

    def predict(self, values: np.array, **kwargs) -> np.array:
        """
        Predicts future values using naive average foreacsting.

        returns : np.array with time series values and forecast combined
        """
        self.values = values

        mean_value = np.mean(self.values[-self.T :])
        self.forecast = np.full(self.h, mean_value)
        self.y = np.append(self.values, self.forecast)

        return self.forecast


@dataclass
class NaiveLast(Forecast):
    _forecast: np.array
    _y: np.array

    def __init__(self, h: int = 1) -> None:
        """
        Initializer for Naive Average class.
        args:
            values : numpy array of time series values
            h : forecast horizon
        """
        super().__init__(h)
        
   

    def predict(self, values: np.array, **kwargs) -> np.array:
        """
        Predicts future values using naive average foreacsting.

        returns : np.array with time series forecast
        """
        self.values = values

        self.forecast = np.full(self.h, self.values[-1])
        self.y = np.append(self.values, self.forecast)

        return self.forecast


@dataclass
class NaiveSeasonal(Forecast):
    _forecast: np.array
    _y: np.array
    _k: np.array

    def __init__(self, h: int = 1, k: int = 4) -> None:
        """
        Initializer for Naive Average class.
        args:
            values : numpy array of time series values
            h : forecast horizon
        """
        super().__init__(h)
        self.k = k

    @property
    def k(self) -> int:
        return self._k

    @k.setter
    def k(self, k) -> None:
        # assert (
        #     len(self.values) > k
        # ), f"Provided seasonal lag {k} cannot be greater than \
        #     lenght of provided values {len(self.values)}"
        self._k = k

    def predict(self, values: np.array, **kwargs) -> np.array:
        """
        Predicts future values using naive average foreacsting.

        returns : np.array with time series values and forecast combined
        """
        self.values = values
        # self.h = h 
        # self.k = k

        frc = copy.copy(self.values)

        for _ in range(self.h):
            frc = np.append(frc, frc[-self.k])

        self.forecast = frc[-self.h :]
        self.y = np.append(self.values, self.forecast)

        return self.forecast


@dataclass
class NaiveDrift(Forecast):
    _forecast: np.array
    _y: np.array

    def __init__(self, h: int = 1) -> None:
        """
        Initializer for Naive Average class.
        args:
            values : numpy array of time series values
            h : forecast horizon
        """
        super().__init__(h)

    def predict(self, values: np.array, **kwargs) -> np.array:
        """
        Predicts future values using naive average foreacsting.

        returns : np.array with time series values and forecast combined
        """
        self.values = values

        h_periods = np.linspace(1, self.h, self.h)
        h_trend = (self.values[-1] - self.values[0]) / len(self.values)
        self.forecast = np.add(np.multiply(h_trend, h_periods), self.values[-1])

        self.y = np.append(self.values, self.forecast)

        return self.forecast

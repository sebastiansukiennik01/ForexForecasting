"""
Contains time series cross-validation, unified for all forecasting methods.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

from typing import Union

from src.preprocessing import DataSet
from src.utils import rmse
from src.forecast.naive import NaiveLast, NaiveAVG, NaiveDrift, NaiveSeasonal


Model = Union[NaiveLast, NaiveAVG, NaiveDrift, NaiveSeasonal]


class TSCV:
    """
    Takes dataset and forecasting class which will be applied on the data.
    Perfmors time series cross validation, and returns calculated error statistics
    (on default MAPE and RMSPE).
    All forecasting classes have to have implemented predict() method, which returns
    forecasted values.
    """

    def __init__(
        self,
        dataset: DataSet,
        n_splits: int = 5,
        max_train_size: int = None,
        test_size: int = None,
        gap: int = 0,
        forecast_errors: list[callable] = [mean_absolute_percentage_error, rmse],
    ) -> None:
        """
        Takes dataset which is the same for all prediction methods.
        args:
            dataset : DataSet object with data for training and testing
            n_splits : number of splits, at least 2
            max_train_size : maximum size for single training set
            test_size : maximum size od test set
            gap : number of observations between end of train set and start of test set.
        """
        self.dataset = dataset
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
        self.forecast_errors = forecast_errors
        self.errors = {fe.__name__: [] for fe in forecast_errors}
        self.predicted = []
        self.__post_init__()
        
    def __post_init__(self) -> None:
        self.dataset.clean_data()

    def run(self, model: Model, **kwargs) -> tuple[float]:
        """
        Runs time series crossValidation using provided model
        args:
            model : model which will be used to fit and forecast values, has to have predict()
                method implemented
            fit : boolean indicating whether model has to be fitted before predicting values
        returns : calculated errors
        """
        self._clean_data()

        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            gap=self.gap,
        )

        for train_idx, test_idx in tscv.split(self.dataset.data):
            train_X, train_y, test_X, test_y = self.split_data(
                train_idx=train_idx, test_idx=test_idx, **kwargs
            )
            
            # fit model and predict
            model.fit(train_X, train_y) # if needed
            predicted_y = model.predict(test_y, **kwargs)
            self.predicted.extend(predicted_y)

            # calculate and append errors
            [
                self.errors[fe.__name__].append(fe(y_true=test_y.values, y_pred=predicted_y))
                for fe in self.forecast_errors
            ]
                

        # return average errors for each error statistic
        return {fe: np.mean(values) for fe, values in self.errors.items()}

    def split_data(self, train_idx: list, test_idx: list, **kwargs) -> tuple[pd.DataFrame]:
        """
        Splits data using TimeSeriesSlits and transforms it.
        """
        norm_type = kwargs.pop('norm_type', 'standarize')

        label = self.dataset.label # clean missing weekends data before choosing indexes

        # change train, test datasets to new ones
        self.dataset.train = self.dataset.data.iloc[train_idx].copy()
        self.dataset.test = self.dataset.data.iloc[test_idx].copy()

        self.dataset.clean_data().normalize(how=norm_type)

        train_X, train_y = (
            self.dataset.train.drop(columns=label),
            self.dataset.train.loc[:, label],
        )
        test_X, test_y = (
            self.dataset.test.drop(columns=label),
            self.dataset.test.loc[:, label],
        )

        return train_X, train_y, test_X, test_y
    
    def _clean_data(self) -> None:
        self.errors = {fe.__name__: [] for fe in self.forecast_errors}
        self.predicted = []
        

    def __repr__(self) -> str:
        return f"TSCV(gap={self.gap}, n_splits={self.n_splits}, max_train_size={self.max_train_size}, test_size={self.test_size}, forecast_errors={self.forecast_errors})"

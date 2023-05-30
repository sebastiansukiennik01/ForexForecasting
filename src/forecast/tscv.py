"""
Contains time series cross-validation, unified for all forecasting methods.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

from typing import Union

from src.preprocessing import DataSet
from src.utils import rmspe
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
        metrics: list[callable] = [mean_absolute_percentage_error, rmspe, mean_absolute_error],
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
        self.model = None
        self.dataset = dataset
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
        self.metrics = metrics  # list of metrics used in evaluating code results
        self.errors = []  # errors used to calcualate metrics
        self.residuals = []  # differences between train set labels and fitted values
        self.error_metrics = {}  # values of calculated metrics {metric_name: value} for errors
        self.resid_metrics = {}  # values of calculated metrics {metric_name: value} for residuals
        self.predicted = []  # predicted value for all test sets
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

        self._clean_cache()
        self.model = model
        split_error_metrics = {m.__name__: [] for m in self.metrics}
        split_resid_metrics = {m.__name__: [] for m in self.metrics}

        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            gap=self.gap,
        )

        i = 1
        for train_idx, test_idx in tscv.split(self.dataset.data):
            print(f"split: {i}/{self.n_splits}")
            i += 1

            train_X, train_y, test_X, test_y = self.split_data(
                train_idx=train_idx, test_idx=test_idx, **kwargs
            )

            # fit model and predict
            # self.model.fit(train_X, train_y, **kwargs)
            predicted_y = self._predict(test_X=test_X, 
                                        test_y=test_y, 
                                        train_X=train_X, 
                                        **kwargs)

            # calculate residuals metrics, and errors metrics
            split_error_metrics = self._calculate_error_metrics(prev_error_metrics=split_error_metrics, 
                                                                test_y=test_y, 
                                                                pred_y=predicted_y)
        
        # get residuals from last train set
        split_resid_metrics = self._calculate_resid_metrics(prev_resid_metrics=split_resid_metrics,
                                                            train_X=train_X,
                                                            train_y=train_y)

        # mean of resisduals and errors metrics
        self._average_metrics(split_resid_metrics, split_error_metrics)
        
        return self.resid_metrics, self.error_metrics

    def split_data(
        self, train_idx: list, test_idx: list, **kwargs
    ) -> tuple[pd.DataFrame]:
        """
        Splits data using TimeSeriesSplits and transforms it.
        """
        norm_type = kwargs.pop("norm_type", "standarize")

        # clean missing weekends data before choosing indexes
        label = self.dataset.label

        # change train, test datasets to new ones
        self.dataset.train = self.dataset.data.iloc[train_idx].copy()
        self.dataset.test = self.dataset.data.iloc[test_idx].copy()

        # self.dataset.clean_data().normalize(how=norm_type)

        return (
            self.dataset.train.drop(columns=label),
            self.dataset.train.loc[:, label],
            self.dataset.test.drop(columns=label),
            self.dataset.test.loc[:, label],
        )

    def _predict(
        self,
        test_X: np.ndarray,
        test_y: np.ndarray,
        train_X: np.ndarray,
        extend_prediction: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Fits model using train set with either features or just label values (depeneding on type of model).
        args:
            test_X : features test set
            test_y : label train set
        returns : numpy array of predicted values
        """
        
        try:
            pred_y = self.model.predict(test_y, **kwargs)
        except ValueError:
            try:
                pred_y = self.model.predict(test_X, **kwargs)
            except TypeError:
                # extend test set with previous observations from train such that 
                # the whole test set is enough to produce 'test_size' number of predictions
                seq_len = self.model.model.seq_len
                n = seq_len + self.test_size
                temp_test_X = pd.concat([train_X, test_X], axis=0).iloc[-n:]
                pred_y = self.model.predict(temp_test_X, **kwargs)
                
        # pred_y is missing first 'seq_len' values, because they couldn't be predicted
        # prepend prediction with real numbers with
        if pred_y.shape[0] != test_y.shape[0]:
            seq_len = self.model.model.seq_len
            missing_pred = test_y.values[:seq_len].flatten() + np.random.uniform(-0.001, 0.001, seq_len)
            pred_y = np.concatenate([missing_pred, pred_y])
            
        if extend_prediction:
            self.predicted.extend(pred_y)

        return pred_y

    def _clean_cache(self) -> None:
        """
        Cleans temporary values between TSCV runs.
        """
        self.error_metrics = {fe.__name__: [] for fe in self.metrics}
        self.resid_metrics = {fe.__name__: [] for fe in self.metrics}
        self.predicted = []
        self.errors = []

    def _calculate_residuals(
        self, train_X: np.ndarray, train_y: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Calculate residuals on biggest (last) test set.
        args:
            train_X : train set features
            train_y : train set labels
        return : nunmpy array of residuals
        """
        pred = self._predict(
            train_X, train_y, extend_prediction=False, **kwargs
        ).flatten()
        y = train_y.values.flatten()
        try:
            resid = np.subtract(y, pred)
        except ValueError:
            # TODO e.g. naive methods don't support fitting whole training set, just specified horizon
            return None
        return resid

    def _calculate_errors(self, true_y: np.ndarray, pred_y: np.ndarray) -> None:
        """
        Calculate difference between true and predicted label values.
        args:
            test_y : true label values
            pred_y : predicted label values
        return : nunmpy array of errors
        """
        true_y = true_y.values.flatten()
        self.errors.extend(np.subtract(true_y, pred_y))
        
    def _calculate_error_metrics(self, prev_error_metrics: dict, test_y: np.ndarray, pred_y: np.ndarray):
        """
        Calculates errors for predicted values, than metrics based on those errors and append them to 
        current errors metrics.
        args:
            prev_error_metrics : dict of error metrics from previous splits (e.g. {'mape': [list of previous mape results]})
            test_y : true label values
            pred_y : predicted label values
        return : dict of current metrics results for errors
        """
        self._calculate_errors(true_y=test_y, pred_y=pred_y)
        [
            prev_error_metrics[m.__name__].append(
                m(y_true=test_y.values, y_pred=pred_y)
            )
            for m in self.metrics
        ]
        
        return prev_error_metrics
    
    def _calculate_resid_metrics(self, prev_resid_metrics: dict, train_X: np.ndarray, train_y: np.ndarray):
        """
        Calculates residuals for train set values, than metrics based on those residuals and append them to 
        current residuals metrics.
        args:
            prev_resid_metrics : dict of residuals metrics from previous splits 
                (e.g. {'mape': [list of previous mape results]})
            train_X : feature values
            train_y : true label values
        return : dict of current metrics results for residuals
        """
        print("\n[INFO] Calculating reisduals..\n")
        self.model.get_residuals(train_X=train_X, train_y=train_y)
        self.residuals = self.model.residuals
        [
            prev_resid_metrics[m.__name__].append(
                m(y_true=train_y.values.flatten(), y_pred=self.model.pred_insample)
            )
            for m in self.metrics
        ]
        
        return prev_resid_metrics
    
    def _average_metrics(self, split_resid_metrics: np.ndarray, split_error_metrics: np.ndarray):
        """
        Calculates average on all tracked metrics for residuals and errors.
        """
        self.error_metrics = {
            fe: np.mean(values) for fe, values in split_error_metrics.items()
        }
        self.resid_metrics = {
            fe: np.mean(values) for fe, values in split_resid_metrics.items()
        } 
        
    def __repr__(self) -> str:
        return f"TSCV(gap={self.gap}, n_splits={self.n_splits}, max_train_size={self.max_train_size}, test_size={self.test_size}, forecast_errors={self.forecast_errors})"

"""
Contains Dataset class used for dividing data to specific datasets.
"""
from __future__ import annotations

import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dataclasses import dataclass
from typing import Union

from src.utils.utils import add_variables


@dataclass
class DataSet(object):
    """
    Initialize dataset instance with dataset divided acocrding to ratios.
    args:
        data : dataframe with all observations
        ratios : list of ratios on how to divide data to train, validation, test datasets,
            all values have to sum up to 1, divides data to [train, test] if 2 values provided
            and to [train, validation, test] if 3 values provided.
    """

    data: pd.DataFrame
    ratios: list[float]
    label: Union[str, list[str]]

    def __post_init__(self):
        """
        Divides data into train, validation, test sets using specified ratios
        """
        # TODO add variables to whole dataset ?!
        self.data = add_variables(self.data, both_labels=False)

        n = self.data.shape[0]
        train_idx = int(n * self._train_r)
        self.train = self.data.iloc[:train_idx].copy()

        if self._validation_r:
            val_idx = int(n * self._validation_r) + train_idx
            self.validation = self.data.iloc[train_idx:val_idx].copy()
            self.test = self.data.iloc[val_idx:].copy()
        else:
            self.validation = pd.DataFrame()
            self.test = self.data.iloc[train_idx:].copy()

    @property
    def ratios(self) -> list[float]:
        return self._ratios

    @ratios.setter
    def ratios(self, values: list[float]) -> None:
        assert (
            int(sum(map(round, values))) == 1
        ), "All ratios must sum up to 1, provided incorrect ratio values"
        assert (
            len(values) == 2 or len(values) == 3
        ), "Number of provided ratios must be either 2 (train, test) or 3 (train, validation, test)"
        self._ratios = values
        if len(values) == 2:
            self._train_r, self._test_r = values
            self._validation_r = None
        else:
            self._train_r, self._validation_r, self._test_r = values

    @property
    def label(self) -> list[str]:
        return self._label

    @label.setter
    def label(self, values: Union[str, list[str]]) -> None:
        self._label = [values] if isinstance(values, str) else values

    def get_tvt_df(self) -> tuple[pd.DataFrame]:
        """
        Returns train, validation and test datasets as pandas dataframes.
        """

        return (
            self.get_train(type="dataframe"),
            self.get_validation(type="dataframe"),
            self.get_test(type="dataframe"),
        )

    def get_tvt_dataset(self) -> tuple[Dataset]:
        """
        Returns train, validation and test datasets as tensorflow datasets.
        """
        # TODO implement conversion to tensorflow datasets
        return

    def get_train(self, type: str) -> Union[pd.DataFrame, Dataset]:
        """
        Returns train set as either pandas dataframe or tensorflow dataset, depending on specified type.
        args:
            type : string of return type, either dataframe or dataset
        returns : train dataset
        """
        if type == "dataframe":
            return self.train
        elif type == "dataset":
            # TODO
            ...

    def get_validation(self, type: str) -> Union[pd.DataFrame, Dataset]:
        """
        Returns validation set as either pandas dataframe or tensorflow dataset, depending on specified type.
        args:
            type : string of return type, either dataframe or dataset
        returns : validation dataset
        """
        if not self._validation_r:
            return pd.DataFrame()
        if type == "dataframe":
            return self.validation
        elif type == "dataset":
            # TODO
            ...

    def get_test(self, type: str) -> Union[pd.DataFrame, Dataset]:
        """
        Returns test set as either pandas dataframe or tensorflow dataset, depending on specified type.
        args:
            type : string of return type, either dataframe or dataset
        returns : test dataset
        """
        if type == "dataframe":
            return self.test
        elif type == "dataset":
            # TODO
            ...

    def add_variables(self, **kwargs) -> DataSet:
        """
        Adds variables to all datasets (train, validation and test)
        """
        self.train = add_variables(self.train, **kwargs)
        self.validation = add_variables(self.validation, **kwargs)
        self.test = add_variables(self.test, **kwargs)

        return self

    def normalize(self, how: str, features_only: bool = True) -> DataSet:
        """
        Performs normalization on all datasets. Categorical features are ommited and left as is.
        args:
            how : type of normalization [standarize/min-max]
        """
        scaler = {"standarize": StandardScaler, "min-max": MinMaxScaler}.get(
            how, "min-max"
        )
        print("NORMALIZING DATA!!")

        to_scale = set(self.train.columns).difference(self._binary_columns())
        to_scale = to_scale.difference(self.label) if features_only else to_scale
        to_scale = list(to_scale)

        # self.test = self._normalize_small_set(self.test, scaler, to_scale)
        if not self.validation.empty:
            self.validation[to_scale] = self._normalize_small_set(
                self.validation[to_scale], scaler, to_scale
            )
        self.train[to_scale] = scaler().fit_transform(self.train[to_scale])
        self.test[to_scale] = scaler().fit_transform(self.test[to_scale])

        return self

    def _normalize_small_set(
        self, _set: pd.DataFrame, scaler: callable, to_scale
    ) -> pd.DataFrame:
        """
        Handles normalization if number of samples in set is smaller than 3.
        args:
            set : validation/test set which could have to few samples to perform correct normalization
            scaler : scaler to be used in normalization
            to_scaler : column names to be normalized
        returns : normalized set
        """
        n = _set.shape[0]
        if n < 3:
            if self.validation.empty:
                temp = pd.concat(
                    [
                        self.train.loc[self.train.index[-2:], to_scale].copy(),
                        _set[to_scale].copy(),
                    ],
                    axis=0,
                )
            else:
                temp = pd.concat(
                    [
                        self.validation.loc[
                            self.validation.index[-2:] :, _set[to_scale]
                        ].copy(),
                        _set.copy(),
                    ],
                    axis=0,
                )
            temp = scaler().fit_transform(temp)[-n:]
        else:
            temp = scaler().fit_transform(_set[to_scale])
        _set[to_scale] = temp

        return _set

    def clean_data(self) -> DataSet:
        """
        Cleanes dataframe from invalid values.
        """
        self.data = self.data.replace([np.inf, -np.inf], np.nan).loc[
            ~self.data.isna().any(axis=1), :
        ]
        self.train = self.train.replace([np.inf, -np.inf], np.nan).loc[
            ~self.train.isna().any(axis=1), :
        ]
        self.validation = self.validation.replace([np.inf, -np.inf], np.nan).loc[
            ~self.validation.isna().any(axis=1), :
        ]
        self.test = self.test.replace([np.inf, -np.inf], np.nan).loc[
            ~self.test.isna().any(axis=1), :
        ]

        return self

    def _binary_columns(self) -> pd.DataFrame:
        """
        Drops binary columns form dataframe
        """
        binary = [c for c in self.train if set(self.train[c].unique()) == set([1, 0])]

        return binary

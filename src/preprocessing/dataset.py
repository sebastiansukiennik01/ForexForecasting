"""
Contains Dataset class used for dividing data to specific datasets.
"""
import pandas as pd
import tensorflow as tf

from tensorflow.data import Dataset

from dataclasses import dataclass
from typing import Union


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
        train_x, train_y = self.get_train(type="dataframe")
        validation_x, validation_y = self.get_validation(type="dataframe")
        test_x, test_y = self.get_test(type="dataframe")
        
        return train_x, train_y, validation_x, validation_y, test_x, test_y

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
            return self.train.drop(columns=self.label), self.train[self.label]
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
            return pd.DataFrame(), pd.DataFrame()
        if type == "dataframe":
            return self.validation.drop(columns=self.label), self.validation[self.label]
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
            return self.test.drop(columns=self.label), self.test[self.label]
        elif type == "dataset":
            # TODO
            ...

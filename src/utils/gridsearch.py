"""
Contains GridSearch class looking for best combination in model parameters
"""
from typing import Union
from itertools import product
import os
import datetime as dt
from multiprocessing import Pool

from src.forecast.naive import Forecast
from src.forecast import TSCV

import pandas as pd

class GridSearch:
    def __init__(
        self, model_class: Forecast, tscv: TSCV, params: dict, by: Union[list, str], p_count: int
    ) -> None:
        """
        Configure model, timeseries cross validation and parameters to be checked.
        args:
            model : model to be checked, must be compatible with TSCV class
            tscv : configured time series cross validation object
            params : dictionary of paramaters to be checked {param_name: [val_a, val_b, ...], ...}
        """
        self.model_class = model_class
        self.tscv = tscv
        self.params = params
        self._by = by
        self.p_count = p_count

    @property
    def by(self) -> list[str]:
        return self._by

    @by.setter
    def by(self, values: Union[list, str]) -> None:
        if isinstance(values, str):
            values = [values]
        available_metrics = [m.__name__ for m in self.tscv.metrics]
        assert (
            v in available_metrics for v in values
        ), "Not all metrics can be accessed. Please check\
            if all metrics can be accesed from TSCV object"
        self._by = values

    def run(
        self,
    ) -> None:
        """
        Performs grid searching in order to find best combinations of provided parameters
        Params provided when initialized are combined each with each, e.g. {'a': [1, 2], 'b': [8, 16]}
        will results in testing four combinations [[1, 8], [1, 16], [2, 8], [2, 16]].
        Sorted results are saved to file in ./gridSearchResults/ directory.
        """

        combinations = self._get_params_combinations()
        # with Pool(processes=self.p_count)  as pool:
        #     pool.map(self._run, combinations)
        
        for comb in combinations:
            self.model = self.model_class()
            results = self.tscv.run(self.model, **comb)
            self._save_results(results=results,
                               combination=comb)
            
    def _run(self, comb: dict) -> None:
        results = self.tscv.run(self.model, **comb)
        self._save_results(results=results,
                           combination=comb)
    
    def _save_results(self, results: dict, combination: dict) -> None:
        """
        Appends to or creates a file with sorted results.
        """
        f_path = self._get_file_path()
        res = {f"{k}_sample": v  for k, v in results[0].items()} | results[1] | combination
        res = {k: str(v) if isinstance(v, list) else v 
               for k, v in res.items()}
        curr_results = pd.DataFrame(res, index=[0])
        
        if os.path.exists(f_path):
            pd.concat([pd.read_csv(f_path, index_col=[0]), curr_results], 
                      axis=0).\
                          sort_values(by=self.by).\
                          reset_index(drop=True).\
                          to_csv(f_path)
        else:
            curr_results.to_csv(f_path)
            

    def _get_params_combinations(self) -> list[dict]:
        """
        Returns all possible parameter combinations.
        """
        comb_generator = product(*self.params.values())
        return [{k: v for k, v in zip(self.params.keys(), c)} for c in comb_generator]

    def _get_file_path(self) -> os.PathLike:
        """
        Generates unique file path for model-tscv combination
        """
        model_name = self.model.__class__.__name__
        return os.path.join('gridSearchResults', f'{model_name}_{id(self.tscv)}.csv')
        
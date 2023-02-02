from typing import Callable, List, Union

import pandas as pd
from sklearn.base import BaseEstimator, MultiOutputMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate

from base import OxariMetaModel
from base.metrics import smape
from base.oxari_types import ArrayLike


class BaselineCrossValidator(BaseEstimator, MultiOutputMixin):

    def __init__(self, estimator:OxariMetaModel, scoring:List[Union[str, Callable]]=[smape, r2_score], cv=10, n_jobs=-1, verbose=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.scores = {}

    def fit(self, X, y, *kwargs) -> "BaselineCrossValidator":
        for name, pipeline in self.estimator.pipelines.items():
            self.scores[name] = cross_validate(pipeline, X=X, y=y[name], scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs, verbose=self.verbose)
        return self

    def transform(self, X, y, **kwargs) -> ArrayLike:
        result = pd.DataFrame()
        for name, sc in self.scores.items():
            tmp = pd.DataFrame(self.scores).assign(scope=name)
            result = pd.concat([result, tmp])
        return result
    
class BootstrapCrossValidator(BaseEstimator, MultiOutputMixin):
    # TODO: Only needs to shuffle and run multiple times.

    def __init__(self, estimator:OxariMetaModel, scoring:List[Union[str, Callable]]=[smape, r2_score], cv=10, n_jobs=-1, verbose=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.scores = {}

    def fit(self, X, y, *kwargs) -> "BootstrapCrossValidator":
        # for name, pipeline in self.estimator.pipelines.items():
        #     self.scores[name] = cross_validate(pipeline, X=X, y=y[name], scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs, verbose=self.verbose)
        return self

    def transform(self, X, y, **kwargs) -> ArrayLike:
        result = pd.DataFrame()
        # for name, sc in self.scores.items():
        #     tmp = pd.DataFrame(self.scores).assign(scope=name)
        #     result = pd.concat([result, tmp])
        return result
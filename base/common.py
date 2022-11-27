from __future__ import annotations
import abc
import csv
import numpy as np
import optuna
import pandas as pd
import sklearn
from numbers import Number
from pmdarima.metrics import smape
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.impute import SimpleImputer, _base
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_absolute_percentage_error as mape,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    silhouette_score,
)
from sklearn.utils.estimator_checks import check_estimator
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from .metrics import dunn_index, mape
from .oxari_types import ArrayLike


# from typing import Union
# import sklearn
# import numpy as np
# import pandas as pd
# from sklearn.utils.estimator_checks import check_estimator

class OxariLogger:
    """
    This is the Oxari Logger class, which handles the output of any official print statement.
    The logger writes it's outputs to STDOUT or to a FILE if a LOG_FILE environment variable was set.   
    
    Task: 
    - Logger shall use a standardized prefix which provides information about the module and pipeline step
    - Logger should use an env var to determine whether to output the logging into a file or stdout
    - Avoid patterns like here https://docs.python.org/3/howto/logging-cookbook.html#patterns-to-avoid
    - In case of production env, the logger should upload the log file of the full pipeline run to digital ocean spaces
    
    """
    def __init__():
        # https://docs.python.org/3/howto/logging-cookbook.html
        pass


class OxariEvaluator(abc.ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def evaluate(self, y_true, y_pred, **kwargs):
        """
        Evaluates multiple metrics and returns a dict with all computed scores.
        """
        #  TODO: Solve the addition of meta data to the evaluate output by using property attribute instead. Implicitly add name when getting evaluation results
        return {"evaluator": self.name, **kwargs}

    @property
    def name(self) -> str:
        return self.__class__.__name__



class DefaultRegressorEvaluator(OxariEvaluator):
    def evaluate(self, y_true, y_pred, **kwargs):

        # TODO: add docstring here

        # compute metrics of interest

        error_metrics = {
            "sMAPE": smape(y_true, y_pred) / 100,
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred, squared=False),
            # "RMSLE": mean_squared_log_error(y_true, y_pred, squared=False),
            "MAPE": mape(y_true, y_pred)
        }

        return super().evaluate(y_true, y_pred, **error_metrics)


class DefaultClusterEvaluator(OxariEvaluator):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def evaluate(self, X, labels, **kwargs):
        """

        Computes 3 flavors of accuracy: Vanilla, AdjacentLenient, AdjacentStrict

        Each accuracy computation is scope and buckets specific

        Appends and saves the results to model/metrics/error_metrics_class.csv

        """
        error_metrics = {
            "sillhouette_coefficient": silhouette_score(X, labels),
            "dunns_index": dunn_index(X, labels),
        }
        return super().evaluate(X, labels, **error_metrics)

class DefaultClassificationEvaluator(OxariEvaluator):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def evaluate(self, y_true, y_pred, **kwargs):
        """

        Computes 3 flavors of accuracy: Vanilla, AdjacentLenient, AdjacentStrict

        Each accuracy computation is scope and buckets specific

        Appends and saves the results to model/metrics/error_metrics_class.csv

        """
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        acc = balanced_accuracy_score(y_true,y_pred)
        error_metrics = {
            "balanced_accuracy": acc,
            "balanced_precision": precision,
            "balanced_recall": recall,
            "balanced_f1": f1,
        }
        return super().evaluate(y_true, y_pred, **error_metrics)

class OxariOptimizer(abc.ABC):
    def __init__(self, num_trials=2, num_startup_trials=1, sampler=None, **kwargs) -> None:
        super().__init__()
        self.num_trials = num_trials
        self.num_startup_trials = num_startup_trials
        self.sampler = sampler or optuna.samplers.TPESampler(n_startup_trials=self.num_startup_trials, warn_independent_sampling=False)

    @abc.abstractmethod
    def optimize(self, X_train, y_train, X_val, y_val, **kwargs) -> Tuple[dict, Any]:
        """
        Evaluates multiple metrics and returns a dict with all computed scores.
        """
        return {}, pd.DataFrame()

    @abc.abstractmethod
    def score_trial(self, trial: optuna.Trial, X_train, y_train, X_val, y_val, **kwargs) -> Number:
        """
        Evaluates multiple metrics and returns a dict with all computed scores.
        """
        return 0

class DefaultOptimizer(OxariOptimizer):
    """
    This optimzer does absolutely nothing. 
    """
    
    def optimize(self, X_train, y_train, X_val, y_val, **kwargs) -> Tuple[dict, Any]:
        return super().optimize(X_train, y_train, X_val, y_val, **kwargs)

    def score_trial(self, trial: optuna.Trial, X_train, y_train, X_val, y_val, **kwargs) -> Number:
        return super().score_trial(trial, X_train, y_train, X_val, y_val, **kwargs)


class OxariMixin(abc.ABC):
    def __init__(self, object_filename=None, **kwargs) -> None:
        self.object_filename = object_filename or self.__class__.__name__
        self.start_time = None
        self.end_time = None

    # @abc.abstractmethod
    # def run(self, **kwargs) -> "OxariMixin":
    #     """
    #     Every component needs to call initialize and finish inside the run function.
    #     """
    #     return self

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        return self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred, **kwargs)

    def set_logger(self, logger: OxariLogger) -> "OxariMixin":
        self._logger = logger
        return self

    def set_evaluator(self, evaluator: OxariEvaluator) -> "OxariMixin":
        self._evaluator = evaluator
        return self

    def set_optimizer(self, optimizer: OxariOptimizer) -> "OxariMixin":
        self._optimizer = optimizer
        return self


class OxariTransformer(OxariMixin, sklearn.base.TransformerMixin, sklearn.base.BaseEstimator, abc.ABC):
    """Just for intellisense convenience. Not really necessary but allows autocompletion"""
    @abc.abstractmethod
    def fit(self, X, y=None, **kwargs) -> "OxariTransformer":
        return self

    @abc.abstractmethod
    def transform(self, X, **kwargs) -> ArrayLike:
        pass


class OxariClassifier(OxariMixin, sklearn.base.ClassifierMixin, sklearn.base.BaseEstimator, abc.ABC):
    """Just for intellisense convenience. Not really necessary but allows autocompletion"""
    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> "OxariClassifier":
        return self

    @abc.abstractmethod
    def predict(self, X, **kwargs) -> ArrayLike:
        pass


class OxariRegressor(OxariMixin, sklearn.base.RegressorMixin, sklearn.base.BaseEstimator, abc.ABC):
    """Just for intellisense convenience. Not really necessary but allows autocompletion"""
    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> "OxariRegressor":
        return self

    @abc.abstractmethod
    def predict(self, X, **kwargs) -> ArrayLike:
        pass


class OxariImputer(OxariMixin, _base._BaseImputer, abc.ABC):
    """
    Handles imputation of missing values for values that are zero. Fit and Transform have to be implemented accordingly.
    """
    def __init__(self, missing_values=np.nan, verbose: int = 0, copy: bool = False, add_indicator: bool = False, **kwargs):
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)
        self.verbose = verbose
        self.copy = copy

    @abc.abstractmethod
    def fit(self, X, y=None, **kwargs) -> "OxariImputer":
        # Takes X and y and trains regressor.
        # Include If X.shape[0] == y.shape[0]: raise ValueError(f�X and y do not have the same size (f{X.shape[0]} != f{X.shape[0]})�).
        # Set self.n_features_in_ = X.shape[1]
        # Avoid setting X and y as attributes. Only increases the model size.
        # When fit is called, any previous call to fit should be ignored.
        # Attributes that have been estimated from the data must always have a name ending with trailing underscore. (e.g.: self.coef_)
        # Reference: https://scikit-learn.org/stable/developers/develop.html#fitting
        return self

    @abc.abstractmethod
    def transform(self, X, **kwargs) -> ArrayLike:
        pass

class OxariPreprocessor(OxariTransformer, abc.ABC):
    def __init__(self, imputer: OxariImputer = None, **kwargs):
        # Only data independant hyperparams.
        # Hyperparams only as keyword arguments
        # Does not contain any logic except setting hyperparams immediately as class attributes
        # Reference:  https://scikit-learn.org/stable/developers/develop.html#instantiation
        self.imputer = imputer

    @abc.abstractmethod
    def fit(self, X, y=None, **kwargs) -> "OxariPreprocessor":
        # Takes X and y and trains regressor.
        # Include If X.shape[0] == y.shape[0]: raise ValueError(f�X and y do not have the same size (f{X.shape[0]} != f{X.shape[0]})�).
        # Set self.n_features_in_ = X.shape[1]
        # Avoid setting X and y as attributes. Only increases the model size.
        # When fit is called, any previous call to fit should be ignored.
        # Attributes that have been estimated from the data must always have a name ending with trailing underscore. (e.g.: self.coef_)
        # Reference: https://scikit-learn.org/stable/developers/develop.html#fitting
        return self

    @abc.abstractmethod
    def transform(self, X, **kwargs) -> ArrayLike:
        pass

    def set_imputer(self, imputer: OxariImputer) -> "OxariPreprocessor":
        self.imputer = imputer
        return self

    # def set_feature_selector(self, feature_selector: OxariFeatureSelector) -> "OxariPreprocessor":
    #     self.feature_selector = feature_selector
    #     return self


class OxariScopeEstimator(OxariRegressor, abc.ABC):
    def __init__(self, **kwargs):
        # Only data independant hyperparams.
        # Hyperparams only as keyword arguments
        # Does not contain any logic except setting hyperparams immediately as class attributes
        # Reference: https://scikit-learn.org/stable/developers/develop.html#instantiation
        evaluator = kwargs.pop('evaluator', DefaultRegressorEvaluator())
        self.set_evaluator(evaluator)
        optimizer = kwargs.pop('optimizer', DefaultOptimizer())
        self.set_optimizer(optimizer)
        self._name = self.__class__.__name__

    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        # Takes X and y and trains regressor.
        # Include If X.shape[0] == y.shape[0]: raise ValueError(f�X and y do not have the same size (f{X.shape[0]} != f{X.shape[0]})�).
        # Set self.n_features_in_ = X.shape[1]
        # Avoid setting X and y as attributes. Only increases the model size.
        # When fit is called, any previous call to fit should be ignored.
        # Attributes that have been estimated from the data must always have a name ending with trailing underscore. (e.g.: self.coef_)
        # Reference: https://scikit-learn.org/stable/developers/develop.html#fitting
        return self

    @abc.abstractmethod
    def predict(self, X) -> ArrayLike:
        # Takes X and computes predictions
        # Returns prediction results
        pass

    @abc.abstractmethod
    def check_conformance(self, ) -> bool:
        # Returns a boolean
        # Uses sklearn utils function check_estimator(self)
        # If this test passes, then deployment shall be allowed
        # check_estimator Makes sure that we can use model evaluation and selection tools such as model_selection.GridSearchCV and pipeline.Pipeline.
        # Reference
        pass

    @property
    def name(self):
        return self._name


class OxariPostprocessor(OxariMixin, abc.ABC):
    def __init__(self, **kwargs):
        # Only data independant hyperparams.
        # Hyperparams only as keyword arguments
        # Does not contain any logic except setting hyperparams immediately as class attributes
        # Reference: https://scikit-learn.org/stable/developers/develop.html#instantiation
        pass

    @abc.abstractmethod
    def run(self, X, y=None, **kwargs) -> "OxariPostprocessor":
        # Takes X and y and trains regressor.
        # Include If X.shape[0] == y.shape[0]: raise ValueError(f�X and y do not have the same size (f{X.shape[0]} != f{X.shape[0]})�).
        # Set self.n_features_in_ = X.shape[1]
        # Avoid setting X and y as attributes. Only increases the model size.
        # When fit is called, any previous call to fit should be ignored.
        # Attributes that have been estimated from the data must always have a name ending with trailing underscore. (e.g.: self.coef_)
        # Reference: https://scikit-learn.org/stable/developers/develop.html#fitting
        return self


class DefaultPostprocessor(OxariPostprocessor):
    def run(self, X, y=None, **kwargs) -> "OxariPostprocessor":
        return X



class OxariFeatureReducer(OxariTransformer, abc.ABC):
    """
    Handles removal of unimportant features. Fit and Transform have to be implemented accordingly.
    """
    def __init__(self, missing_values=np.nan, verbose: int = 0, copy: bool = False, add_indicator: bool = False, **kwargs):
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)
        self.verbose = verbose
        self.copy = copy

    @abc.abstractmethod
    def fit(self, X, y=None, **kwargs) -> "OxariFeatureReducer":
        """
        Trains regressor
        Include If X.shape[0] == y.shape[0]: raise ValueError(f“X and y do not have the same size (f{X.shape[0]} != f{X.shape[0]})”).
        Set self.n_features_in_ = X.shape[1]
        Avoid setting X and y as attributes. Only increases the model size.
        When fit is called, any previous call to fit should be ignored.
        Attributes that have been estimated from the data must always have a name ending with trailing underscore. (e.g.: self.coef_)
        
        Reference: https://scikit-learn.org/stable/developers/develop.html#fitting

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.

        Returns:
            OxariFeatureReducer: _description_
        """
        return self

    @abc.abstractmethod
    def transform(self, X, **kwargs) -> ArrayLike:
        pass


    def visualize(self, X, **kwargs):
        figsize = kwargs.pop('figsize',(20,20))
        fig = plt.figure(figsize=figsize)
        reduced_X = self.transform(X, **kwargs)
        x,y,z = reduced_X[:, :3]
        ax = fig.axes(projection='3d')
        ax.scatter3D(x,y,z)
        plt.show()

        
        

# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
class OxariPipeline(OxariRegressor, MetaEstimatorMixin, abc.ABC):
    def __init__(
        self,
        # dataset: OxariDataLoader = None,
        preprocessor: OxariPreprocessor = None,
        feature_selector: OxariFeatureReducer = None,
        scope_estimator: OxariScopeEstimator = None,
    ):
        # self.dataset = dataset
        self.preprocessor = preprocessor
        self.feature_selector = feature_selector
        self.estimator = scope_estimator
        self._evaluation_results = {}
        self._start_time = None
        self._end_time = None
        # self.resources_postprocessor = database_deployer

    @abc.abstractmethod
    def run_pipeline(self, **kwargs):
        # load dataset and hold in class
        #  dataset
        pass

    def predict(self, X, **kwargs) -> ArrayLike:
        X = self.preprocessor.transform(X, **kwargs)
        X = self.feature_selector.transform(X, **kwargs)
        return self.estimator.predict(X.drop(columns = ["scope_1", "scope_2", "scope_3"], axis=1), **kwargs)
    
    def fit(self, X, y, **kwargs) -> "OxariPipeline":
        self.estimator = self.estimator.fit(X, y, **kwargs)
        return self

    @property
    def evaluation_results(self):
        return {"model": self.estimator.name, **self._evaluation_results}


class OxariMetaModel(OxariRegressor, MultiOutputMixin, abc.ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.created = None # TODO: Make sure the meta-model also has an attribute which records the full creation time (data, hour). Normalize timezone to UTC.
        self.pipelines: Dict[str, OxariPipeline] = {}

    def add_pipeline(self, scope: int, pipeline: OxariPipeline) -> "OxariMetaModel":
        if not isinstance(scope, int):
            raise Exception(f"'scope' is not an int: {scope}")
        if not ((scope > 0) and (scope < 4)):
            raise Exception(f"'scope' is not between either 1, 2 or 3: {scope}")
        self.pipelines[f"scope_{scope}"] = pipeline
        return self

    def get_pipeline(self, scope: int) -> OxariPipeline:
        return self.pipelines[f"scope_{scope}"]

    def fit(self, X, y=None, **kwargs):
        pass

    def predict(self, X, **kwargs) -> ArrayLike:
        scope = kwargs.pop("scope", "all")
        X = X.drop(columns = ["isin", "year"])
        if scope == "all":
            return self._predict_all(X, **kwargs)
        return self.get_pipeline(scope).predict(X, **kwargs)

    def _predict_all(self, X, **kwargs) -> ArrayLike:
        result = pd.DataFrame()
        for scope_str, estimator in self.pipelines.items():
            y_pred = estimator.predict(X, **kwargs)
            result[scope_str] = y_pred
        return result

    def collect_eval_results(self) -> List[dict]:
        results = []

        for scope, pipeline in self.pipelines.items():
            results.append(pipeline.evaluation_results)

        return results
    
class OxariLinearAnnualReduction(OxariRegressor, OxariTransformer, OxariMixin, abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, X, y=None) -> "OxariLinearAnnualReduction":
        return self

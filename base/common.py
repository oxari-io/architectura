from __future__ import annotations
import abc
import csv
import os
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
import copy
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

from typing import Union
import sklearn
import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
import logging
from sklearn.model_selection import train_test_split

os.environ["LOGLEVEL"] = "DEBUG"
LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
WRITE_TO =  "cout" # "./logger.log"

class OxariLoggerMixin:
    """
    This is the Oxari Logger class, which handles the output of any official print statement.
    The logger writes it's outputs to STDOUT or to a FILE if a LOG_FILE environment variable was set.   
    
    Task: 
    - Logger shall use a standardized prefix which provides information about the module and pipeline step
    - Logger should use an env var to determine whether to output the logging into a file or stdout
    - Avoid patterns like here https://docs.python.org/3/howto/logging-cookbook.html#patterns-to-avoid
    - In case of production env, the logger should upload the log file of the full pipeline run to digital ocean spaces
    
    """
    logger: logging.Logger

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        if WRITE_TO == "cout":
            logging.basicConfig()
            logging.root.setLevel(LOGLEVEL)
        else:
            logging.basicConfig(filename=WRITE_TO, level=LOGLEVEL) # level=logging.DEBUG
        
        self.logger_name = self.__class__.__name__
    

class ReducedDataMixin:
    def get_sample_indices(self, X: ArrayLike) -> ArrayLike:
        max_size = len(X) // 10
        indices = np.random.randint(0, len(X), max_size)
        return indices


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
        # y_pred[np.isinf(y_pred)] = 10e12
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
        acc = balanced_accuracy_score(y_true, y_pred)
        error_metrics = {
            "balanced_accuracy": acc,
            "balanced_precision": precision,
            "balanced_recall": recall,
            "balanced_f1": f1,
        }
        return super().evaluate(y_true, y_pred, **error_metrics)


# TODO: Integrate optuna visualisation as method
class OxariOptimizer(abc.ABC):
    def __init__(self, n_trials=2, n_startup_trials=1, sampler=None, **kwargs) -> None:
        super().__init__()
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.sampler = sampler or optuna.samplers.TPESampler(n_startup_trials=self.n_startup_trials)


    @abc.abstractmethod
    def optimize(self, X_train, y_train, X_val, y_val, **kwargs) -> Tuple[dict, pd.DataFrame]:
        """
        Explore the hyperparameter tning space with optuna.
        Creates csv and pickle files with the saved hyperparameters for classification

        Parameters:
        X_train (numpy array): training data (features)
        y_train (numpy array): training data (targets)
        X_val (numpy array): validation data (features)
        y_val (numpy array): validation data (targets)
        num_startup_trials (int): 
        n_trials (int): 

        Return:
        study.best_params (data structure): contains the best found parameters within the given space
        """

        # create optuna study
        # num_startup_trials is the number of random iterations at the beginiing
        study = optuna.create_study(
            study_name=f"{self.__class__.__name__}_process_hp_tuning",
            direction="minimize",
            sampler=self.sampler,
        )

        # running optimization
        # trials is the full number of iterations

        study.optimize(lambda trial: self.score_trial(trial, X_train, y_train, X_val, y_val), n_trials=self.n_trials, show_progress_bar=False)

        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

        return study.best_params, df

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
    def __init__(self, name=None, **kwargs) -> None:
        super().__init__()
        self._name = name or self.__class__.__name__
        self.start_time = None
        self.end_time = None
        self.params = kwargs.pop("params", {})

    # @abc.abstractmethod
    # def run(self, **kwargs) -> "OxariMixin":
    #     """
    #     Every component needs to call initialize and finish inside the run function.
    #     """
    #     return self

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs) -> OxariMixin:
        return self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs) -> OxariMixin:
        return self._evaluator.evaluate(y_true, y_pred, **kwargs)

    def set_evaluator(self, evaluator: OxariEvaluator) -> OxariMixin:
        self._evaluator = evaluator
        return self

    def set_optimizer(self, optimizer: OxariOptimizer) -> OxariMixin:
        self._optimizer = optimizer
        return self

    def set_params(self, **params):
        self.params = params
        return self

    def get_params(self, deep=True):
        return {"params": self.params}

    # TODO: Needs get_params and get_config
    def get_config(self, deep=True):
        return {"name": self._name, **self.params}

    @property
    def name(self):
        return self._name

    def clone(self) -> OxariMixin:
        # TODO: Might introduce problems with bidirectional associations between objects. Needs better conceptual plan.
        return copy.deepcopy(self, {})


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
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> "OxariRegressor":
        return self

    @abc.abstractmethod
    def predict(self, X:ArrayLike, **kwargs) -> ArrayLike:
        pass

    def _set_meta(self, X:ArrayLike, **kwargs) -> ArrayLike:
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)

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


class OxariPreprocessor(OxariTransformer, OxariLoggerMixin, abc.ABC):
    def __init__(self, imputer: OxariImputer = None, **kwargs):
        super().__init__(**kwargs)
        # Only data independant hyperparams.
        # Hyperparams only as keyword arguments
        # Does not contain any logic except setting hyperparams immediately as class attributes
        # Reference:  https://scikit-learn.org/stable/developers/develop.html#instantiation
        self.imputer = imputer
        self.logger.debug("Preprocessor initialized!")

    @abc.abstractmethod
    def fit(self, X, y=None, **kwargs) -> "OxariPreprocessor":
        # Takes X and y and trains regressor.
        # Include If X.shape[0] == y.shape[0]: raise ValueError(f�X and y do not have the same size (f{X.shape[0]} != f{X.shape[0]})�).
        # Set self.n_features_in_ = X.shape[1]
        # Avoid setting X and y as attributes. Only increases the model size.
        # When fit is called, any previous call to fit should be ignored.
        # Attributes that have been estimated from the data must always have a name ending with trailing underscore. (e.g.: self.coef_)
        # Reference: https://scikit-learn.org/stable/developers/develop.html#fitting
        self.logger.debug("Preprocessor is fitted!")
        return self

    @abc.abstractmethod
    def transform(self, X, **kwargs) -> ArrayLike:
        self.logger.debug("Preprocessor is transformed!")
        pass

    def set_imputer(self, imputer: OxariImputer) -> "OxariPreprocessor":
        self.imputer = imputer
        return self

    # def debug(self, message):
    #     return super().debug(message)

    # def set_feature_selector(self, feature_selector: OxariFeatureSelector) -> "OxariPreprocessor":
    #     self.feature_selector = feature_selector
    #     return self


class OxariScopeEstimator(OxariRegressor, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Only data independant hyperparams.
        # Hyperparams only as keyword arguments
        # Does not contain any logic except setting hyperparams immediately as class attributes
        # Reference: https://scikit-learn.org/stable/developers/develop.html#instantiation
        evaluator = kwargs.pop('evaluator', DefaultRegressorEvaluator())
        self.set_evaluator(evaluator)
        optimizer = kwargs.pop('optimizer', DefaultOptimizer())
        self.set_optimizer(optimizer)
        self.n_trials = kwargs.get("n_trials", 5)
        self.n_startup_trials = kwargs.get("n_startup_trials", 1)        
        # This is a model specific preprocessor
        self._sub_preprocessor: OxariTransformer = None

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

    # def set_preprocessor(self, preprocessor:BaseEstimator):
    #     self._sub_preprocessor = preprocessor

    @staticmethod
    def _make_model_specific_preprocessor(X, y, **kwargs) -> OxariTransformer:
        return None

    # @abc.abstractmethod
    def check_conformance(self, ) -> bool:
        # TODO: Implement
        # Returns a boolean
        # Uses sklearn utils function check_estimator(self)
        # If this test passes, then deployment shall be allowed
        # check_estimator Makes sure that we can use model evaluation and selection tools such as model_selection.GridSearchCV and pipeline.Pipeline.
        # Reference
        pass


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


class OxariScopeTransformer(OxariTransformer):
    def fit(self, X, y=None, **kwargs) -> OxariScopeTransformer:
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(y, **fit_params)

    @abc.abstractmethod
    def transform(self, y, **kwargs) -> ArrayLike:
        return y.copy()

    @abc.abstractmethod
    def reverse_transform(self, y, **kwargs) -> ArrayLike:
        return y.copy()


class DummyScaler(OxariScopeTransformer):
    def transform(self, y, **kwargs) -> ArrayLike:
        return super().transform(y, **kwargs)

    def reverse_transform(self, y, **kwargs) -> ArrayLike:
        return super().reverse_transform(y, **kwargs)


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
        figsize = kwargs.pop('figsize', (15, 15))
        fig = plt.subplots(figsize=figsize)
        reduced_X = X[self.reduced_feature_columns].values
        x, y, z = reduced_X[:, :3].T
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, z)
        plt.show()

    def merge(self, old_data: pd.DataFrame, reduced_feature_data: pd.DataFrame, feature_columns: List[str]):
        reduced_feature_data.columns = self.reduced_feature_columns
        return old_data.merge(reduced_feature_data, left_index=True, right_index=True).drop(feature_columns, axis=1)


# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
class OxariPipeline(OxariRegressor, MetaEstimatorMixin, abc.ABC):
    def __init__(
        self,
        # dataset: OxariDataLoader = None,
        preprocessor: OxariPreprocessor = None,
        feature_selector: OxariFeatureReducer = None,
        scope_estimator: OxariScopeEstimator = None,
        ci_estimator: OxariConfidenceEstimator = None,
        scope_transformer: OxariScopeTransformer = None,
        **kwargs,
    ):
        # self.dataset = dataset
        super().__init__(**kwargs)
        self.preprocessor = preprocessor
        self.feature_selector = feature_selector
        self.estimator = scope_estimator
        self.ci_estimator = ci_estimator.set_pipeline(self) if ci_estimator else DummyConfidenceEstimator()
        self.scope_transformer = scope_transformer
        self._evaluation_results = {}
        self._start_time = None
        self._end_time = None
        self.features = None
        self._evaluator = DefaultRegressorEvaluator()
        # self.resources_postprocessor = database_deployer

    def _preprocess(self, X, **kwargs) -> ArrayLike:
        X_new = self.preprocessor.transform(X, **kwargs)
        X_new = self.feature_selector.transform(X_new, **kwargs)
        return X_new

    def _transform_scope(self, y, **kwargs) -> ArrayLike:
        y_new = self.scope_transformer.transform(y, **kwargs)
        return y_new
    
    def _reverse_scope(self, y, **kwargs) -> ArrayLike:
        y_new = self.scope_transformer.reverse_transform(y, **kwargs)
        return y_new

    def predict(self, X, **kwargs) -> ArrayLike:
        return_std = kwargs.pop('return_ci', False)
        # return_raw = kwargs.pop('return_raw', False) # 
        if return_std:
            preds = self.ci_estimator.predict(X, **kwargs)
            return self.scope_transformer.reverse_transform(preds)
        X_new = self._preprocess(X, **kwargs).drop(columns=["isin", "year", "scope_1", "scope_2", "scope_3"], axis=1, errors='ignore')
        preds = self.estimator.predict(X_new, **kwargs)
        return self.scope_transformer.reverse_transform(preds)

    def fit(self, X, y, **kwargs) -> OxariPipeline:
        self._set_meta(X)
        is_na = np.isnan(y)
        X = self._preprocess(X, **kwargs)
        y = self._transform_scope(y, **kwargs)
        self.estimator = self.estimator.set_params(**self.params).fit(X[~is_na], y[~is_na], **kwargs)
        return self


    
    def fit_confidence(self, X,y,**kwargs) -> OxariPipeline:
        is_na = np.isnan(y)
        # X = self._preprocess(X, **kwargs)
        # y = self._transform_scope(y, **kwargs)
        X,y =X[~is_na], y[~is_na] 
        self.ci_estimator = self.ci_estimator.fit(X,y, **kwargs)
        return self


    def optimise(self, X, y, **kwargs) -> OxariPipeline:
        df_processed: pd.DataFrame = self.preprocessor.fit_transform(X,y)
        df_reduced: pd.DataFrame = self.feature_selector.fit_transform(df_processed, features=df_processed.columns)
        y_transformed = self.scope_transformer.fit_transform(X,y)
        X, y = df_reduced, y_transformed
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)        
        self.params, self.info = self.estimator.optimize(X_train, y_train, X_test, y_test)
        return self

    def evaluate(self, X_train, y_train, X_test, y_test) -> OxariPipeline:
        X_test = self._preprocess(X_test)
        X_train = self._preprocess(X_train)
        y_pred_test = self.estimator.predict(X_test)
        y_pred_train = self.estimator.predict(X_train)
        y_pred_test_reversed = self._reverse_scope(y_pred_test)        
        y_test_transformed = self._transform_scope(y_test)
        y_train_transformed = self._transform_scope(y_train)
        self._evaluation_results = {}
        self._evaluation_results["raw"] = self._evaluator.evaluate(y_test, y_pred_test_reversed, X_test=X_test)
        self._evaluation_results["test"] = self.estimator.evaluate(y_test_transformed, y_pred_test, X_test=X_test)
        self._evaluation_results["train"] = self.estimator.evaluate(y_train_transformed, y_pred_train, X_test=X_train)
        return self

    def clone(self) -> OxariPipeline:
        # TODO: Might introduce problems with bidirectional associations between objects. Needs better conceptual plan.
        return copy.deepcopy(self, {})

    @property
    def evaluation_results(self):
        return {"pipeline": self.name, **self._evaluation_results}


class OxariConfidenceEstimator(OxariScopeEstimator, MultiOutputMixin):
    def __init__(self, pipeline: OxariPipeline = None, alpha=0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.pipeline = pipeline

    def set_pipeline(self, pipeline: OxariPipeline):
        self.pipeline = pipeline
        return self


class DummyConfidenceEstimator(OxariConfidenceEstimator):
    """
    For Probablistic models that already have a native way to predict the standard deviation.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs) -> "OxariRegressor":
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs) -> ArrayLike:
        df = pd.DataFrame()
        mean_ = self.pipeline.predict(X)
        df['lower'] = mean_
        df['pred'] = mean_
        df['upper'] = mean_
        return df


class OxariMetaModel(OxariRegressor, MultiOutputMixin, abc.ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.created = None  # TODO: Make sure the meta-model also has an attribute which records the full creation time (data, hour). Normalize timezone to UTC.
        self.pipelines: Dict[str, OxariPipeline] = {}

    def add_pipeline(self, scope: int, pipeline: OxariPipeline, **kwargs) -> "OxariMetaModel":
        self.check_scope(scope)
        self.pipelines[f"scope_{scope}"] = pipeline
        return self

    def check_scope(self, scope):
        if not isinstance(scope, int):
            raise Exception(f"'scope' is not an int: {scope}")
        if not ((scope > 0) and (scope < 4)):
            raise Exception(f"'scope' is not between either 1, 2 or 3: {scope}")

    def get_pipeline(self, scope: int) -> OxariPipeline:
        return self.pipelines[f"scope_{scope}"]

    def fit(self, X, y=None, **kwargs):
        self._set_meta(X)
        scope = kwargs.pop("scope", "all")
        if scope == "all":
            return self._fit_all(X, **kwargs)
        return self.get_pipeline(scope).fit(X, y[scope], **kwargs)

    def _fit_all(self, X, y=None, **kwargs) -> ArrayLike:
        for scope_str, estimator in self.pipelines.items():
            estimator.fit(X, y[scope_str], **kwargs)
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        scope = kwargs.pop("scope", "all")
        X = X.drop(columns=["isin", "year", "scope_1", "scope_2", "scope_3"], errors='ignore')
        if scope == "all":
            return self._predict_all(X, **kwargs)
        return self.get_pipeline(scope).predict(X, **kwargs)

    def _predict_all(self, X, **kwargs) -> ArrayLike:
        result = pd.DataFrame()
        return_ci = kwargs.pop('return_ci', False)
        if return_ci:
            for scope_str, estimator in self.pipelines.items():
                y_pred = estimator.predict(X, return_ci=return_ci, **kwargs)
                y_pred.columns = [f"{scope_str}_{col}" for col in y_pred.columns]
                result = pd.concat([result, y_pred], axis=1)
            return result               
        
        for scope_str, estimator in self.pipelines.items():
            y_pred = estimator.predict(X, **kwargs)
            result[scope_str] = y_pred
        return result

    def collect_eval_results(self) -> List[dict]:
        results = []

        for scope, pipeline in self.pipelines.items():
            results.append(pipeline.evaluation_results)

        return results


# class DefaultMetaModel(OxariMetaModel):
#     """
#     A subclass of the Meta model. However this model will also provide confidence intervalls.
#     """
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)

#     def add_pipeline(self, scope: int, pipeline: OxariPipeline, ci_strategy:OxariConfidenceEstimator, **kwargs) -> "OxariMetaModel":
#         self.check_scope(scope)
#         pipeline = ci_strategy.set_pipeline(pipeline)
#         self.pipelines[f"scope_{scope}"] = pipeline
#         return self


class OxariLinearAnnualReduction(OxariRegressor, OxariTransformer, OxariMixin, abc.ABC):
    def __init__(self):
        self.params = {}
        pass

    @abc.abstractmethod
    def fit(self, X, y=None) -> "OxariLinearAnnualReduction":
        return self

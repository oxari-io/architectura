from __future__ import annotations

import abc
import copy
import logging
import os
import time
from numbers import Number
from typing import Any, Dict, List, Tuple
import sys
import platform
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
from pmdarima.metrics import smape
from sklearn.base import MetaEstimatorMixin, MultiOutputMixin, BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.impute import _base
from sklearn.metrics import (balanced_accuracy_score, mean_absolute_error, mean_squared_error, precision_recall_fscore_support, r2_score, silhouette_score, median_absolute_error, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from typing_extensions import Self
from sklearn.preprocessing import minmax_scale
from .metrics import dunn_index, mape
from .oxari_types import ArrayLike
import colorlog
import time as tm
import datetime as dt
import cloudpickle as pkl

os.environ["LOGLEVEL"] = "DEBUG"
LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
WRITE_TO = "./logger.log"  # "cout"
logging.root.setLevel(LOGLEVEL)


# FEEDBACK:
# - Logger had no formatting
# - The logging was not really tested
# - Some of the things logged are of little use
# - WRITE_TO should be handled via environment variable
# - Many places that had a logging out put did not work
class OxariLoggerMixin(abc.ABC):
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
    _format: str = '[%(levelname)1.1s %(asctime)s] %(name)s - %(levelname)s - %(message)s'
    _format_colored: str = '%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(name)s - %(levelname)s - %(message)s'
    _colors = {
        'DEBUG': 'cyan',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }

    def __init__(self, **kwargs) -> None:
        # super().__init__(**kwargs)
        formatter = colorlog.ColoredFormatter(OxariLoggerMixin._format_colored, log_colors=OxariLoggerMixin._colors)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger_name = self.__class__.__name__
        if len(self.logger.handlers) > 0:
            return None
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        if not WRITE_TO == "cout":
            formatter = logging.Formatter(OxariLoggerMixin._format)
            fhandler = logging.FileHandler(WRITE_TO)
            fhandler.setFormatter(formatter)
            self.logger.addHandler(fhandler)
        return None


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
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        # compute metrics of interest
        # y_pred[np.isinf(y_pred)] = 10e12
        offsets = np.array(np.maximum(y_pred, y_true) / np.minimum(y_pred, y_true))
        percentile_deviation = np.quantile(offsets, [.5, .75, .90, .95])
        # NOTE MAPE: Important interpretation https://medium.com/@davide.sarra/how-to-interpret-smape-just-like-mape-bf799ba03bdc
        # NOTE R2: Why it's useless https://data.library.virginia.edu/is-r-squared-useless/
        # NOTE RMSE: Tends to grow with sample size, which is undesirable https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d
        error_metrics = {
            "sMAPE": smape(y_true, y_pred) / 100,
            "R2": r2_score(y_true, y_pred),
            "MAE": median_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred, squared=False),
            # "RMSLE": mean_squared_log_error(y_true, y_pred, squared=False),
            "MAPE": mape(y_true, y_pred),
            "offset_percentile": {
                "50%": percentile_deviation[0],
                "75%": percentile_deviation[1],
                "90%": percentile_deviation[2],
                "95%": percentile_deviation[3]
            },
        }
        print(f"Here's the sMAPE value of the regressor evaluator: {smape(y_true, y_pred) / 100}")
        # self.logger.info(f'sMAPE value of model evaluation: {smape(y_true, y_pred) / 100}')

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

        # v_label = np.concatenate([y for _, y in validation_dataset], axis=0)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)

        error_metrics = {
            "balanced_accuracy": acc,
            "balanced_precision": precision,
            "balanced_recall": recall,
            "balanced_f1": f1,
            # "conf_matrix": conf_matrix,
            # "classification_report": class_report
        }

        # print(f"Here's the classification report of the BucketClassifier evaluator: {class_report}")

        return super().evaluate(y_true, y_pred, **error_metrics)


# TODO: Integrate optuna visualisation as method
class OxariOptimizer(OxariLoggerMixin, abc.ABC):

    def __init__(self, n_trials=2, n_startup_trials=1, sampler=None, **kwargs) -> None:
        super().__init__()
        self.n_trials = n_trials
        self.sampler = sampler or optuna.samplers.TPESampler()
        self.sampler._n_startup_trials = n_startup_trials

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


class OxariMixin(OxariLoggerMixin, abc.ABC):

    def __init__(self, name=None, **kwargs) -> None:
        super().__init__(**kwargs)
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

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs) -> Self:
        return self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs) -> Self:
        return self._evaluator.evaluate(y_true, y_pred, **kwargs)

    def set_evaluator(self, evaluator: OxariEvaluator) -> Self:
        self._evaluator = evaluator
        return self

    def set_optimizer(self, optimizer: OxariOptimizer) -> Self:
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
    def fit(self, X:ArrayLike, y:ArrayLike=None, **kwargs) -> Self:
        return self

    @abc.abstractmethod
    def transform(self, X:ArrayLike, **kwargs) -> ArrayLike:
        pass


class OxariClassifier(OxariMixin, sklearn.base.ClassifierMixin, sklearn.base.BaseEstimator, abc.ABC):
    """Just for intellisense convenience. Not really necessary but allows autocompletion"""

    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> Self:
        return self

    @abc.abstractmethod
    def predict(self, X, **kwargs) -> ArrayLike:
        pass


class OxariRegressor(OxariMixin, sklearn.base.RegressorMixin, sklearn.base.BaseEstimator, abc.ABC):
    """Just for intellisense convenience. Not really necessary but allows autocompletion"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> Self:
        return self

    @abc.abstractmethod
    def predict(self, X: ArrayLike, **kwargs) -> ArrayLike:
        pass

    def _set_meta(self, X: ArrayLike, **kwargs) -> ArrayLike:
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)


class OxariImputer(OxariMixin, _base._BaseImputer, abc.ABC):
    """
    Handles imputation of missing values for values that are zero. Fit and Transform have to be implemented accordingly.
    """

    class DefaultImputerEvaluator(OxariEvaluator):

        def evaluate(self, y_true, y_pred, **kwargs):
            error_metrics = {
                "sMAPE": smape(y_true, y_pred) / 100,
                # "R2": r2_score(y_true, y_pred),
                # "MAE": mean_absolute_error(y_true, y_pred),
                "MAE": median_absolute_error(y_true, y_pred),
            }
            return super().evaluate(y_true, y_pred, **error_metrics)

    def __init__(self, missing_values=np.nan, verbose: int = 0, copy: bool = False, add_indicator: bool = False, **kwargs):
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)
        self.verbose = verbose
        self.copy = copy
        evaluator = kwargs.pop('evaluator', OxariImputer.DefaultImputerEvaluator())
        self.set_evaluator(evaluator)

    @abc.abstractmethod
    def fit(self, X:ArrayLike, y=None, **kwargs) -> Self:
        # Takes X and y and trains regressor.
        # Include If X.shape[0] == y.shape[0]: raise ValueError(f�X and y do not have the same size (f{X.shape[0]} != f{X.shape[0]})�).
        # Set self.n_features_in_ = X.shape[1]
        # Avoid setting X and y as attributes. Only increases the model size.
        # When fit is called, any previous call to fit should be ignored.
        # Attributes that have been estimated from the data must always have a name ending with trailing underscore. (e.g.: self.coef_)
        # Reference: https://scikit-learn.org/stable/developers/develop.html#fitting
        return self

    @abc.abstractmethod
    def transform(self, X:ArrayLike, **kwargs) -> ArrayLike:
        pass

    def evaluate(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        # TODO: also include supervised methods R² into eval
        p = kwargs.pop('p', 0.3)

        X_true = X.dropna(how='any')
        X_true_features = X_true.filter(regex="^ft_num", axis=1)
        ft_cols = X_true_features.columns

        rows, cols = X_true_features.shape
        mask = ~(np.random.rand(rows, cols) < p)

        X_eval = X_true.copy()

        X_eval[ft_cols] = np.where(mask, X_true[ft_cols], np.nan)
        X_pred = self.transform(X_eval, **kwargs)

        y_true = X_true[ft_cols].values[np.where(~mask)]
        y_pred = X_pred[ft_cols].values[np.where(~mask)] + np.finfo(float).eps
        self._evaluation_results = {}
        self._evaluation_results["overall"] = self._evaluator.evaluate(y_true, y_pred)
        return self

    @property
    def evaluation_results(self):
        return {"imputer": self.name, **self._evaluation_results}
        # self.logger.info(f'sMAPE value of model evaluation: {smape(y_true, y_pred) / 100}')


class OxariPreprocessor(OxariTransformer, abc.ABC):

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
    def transform(self, X, y=None, **kwargs) -> ArrayLike:
        pass

    def set_imputer(self, imputer: OxariImputer) -> "OxariPreprocessor":
        self.imputer = imputer
        return self

    # def set_feature_selector(self, feature_selector: OxariFeatureSelector) -> "OxariPreprocessor":
    #     self.feature_selector = feature_selector
    #     return self


class OxariScopeEstimator(OxariRegressor, abc.ABC):

    def __init__(self, n_trials=1, n_startup_trials=1, **kwargs):
        super().__init__(**kwargs)
        # Only data independant hyperparams.
        # Hyperparams only as keyword arguments
        # Does not contain any logic except setting hyperparams immediately as class attributes
        # Reference: https://scikit-learn.org/stable/developers/develop.html#instantiation
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        evaluator = kwargs.pop('evaluator', DefaultRegressorEvaluator())
        self.set_evaluator(evaluator)
        optimizer = kwargs.pop('optimizer', DefaultOptimizer(n_trials=self.n_trials, n_startup_trials=self.n_startup_trials))
        self.set_optimizer(optimizer)
        # This is a model specific preprocessor
        self._sub_preprocessor: OxariTransformer = None

    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> Self:
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

    # def set_optuna_params(self, n_trials=2, n_startup_trials=1) -> Self:
    #     self._optimizer.n_startup_trials = n_startup_trials
    #     self._optimizer.n_trials = n_trials
    #     return self

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
        super().__init__(**kwargs)

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


class OxariFeatureTransformer(OneToOneFeatureMixin, OxariTransformer):

    def fit(self, X, y=None, **kwargs) -> Self:
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(y, **fit_params)

    @abc.abstractmethod
    def transform(self, X, y=None, **kwargs) -> ArrayLike:
        return X.copy()

    @abc.abstractmethod
    def reverse_transform(self, X, **kwargs) -> ArrayLike:
        return X.copy()


class OxariScopeTransformer(OxariTransformer):

    def fit(self, X, y=None, **kwargs) -> OxariScopeTransformer:
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(y, **fit_params)

    @abc.abstractmethod
    def transform(self, y=None, **kwargs) -> ArrayLike:
        return y.copy()

    @abc.abstractmethod
    def reverse_transform(self, y, **kwargs) -> ArrayLike:
        return y.copy()


class OxariFeatureReducer(OxariTransformer, abc.ABC):
    """
    Handles removal of unimportant features. Fit and Transform have to be implemented accordingly.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_components_ = None

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

    def visualize(self, X, y, **kwargs):
        figsize = kwargs.pop('figsize', (15, 15))
        fig = plt.subplots(figsize=figsize)
        ax = plt.axes(projection='3d')
        X_reduced = self.transform(X).values
        y = y.flatten()
        print(X_reduced.shape)
        for g in np.unique(y):
            ix = y == g
            xc, yc, zc = X_reduced[ix, :3].T
            ax.scatter3D(xc, yc, zc, label=g, s=100)
            ax.legend()
        plt.show()

    # TODO: Needs to be optimized for automatic feature detection.
    def merge(self, old_data: pd.DataFrame, new_data: pd.DataFrame, **kwargs):
        new_data.columns = [f"ft_{i}" for i in range(len(new_data.columns))]
        return old_data.filter(regex='^!ft', axis=1).merge(new_data, left_index=True, right_index=True)

    def get_config(self, deep=True):
        return {'n_components_': self.n_components_, **super().get_config(deep)}


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
        self.stats = {}
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

    def set_ci_estimator(self, estimator: OxariConfidenceEstimator) -> Self:
        self.ci_estimator = estimator.set_pipeline(self)
        return self

    def set_estimator(self, estimator: OxariScopeEstimator) -> Self:
        self.estimator = estimator
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        return_std = kwargs.pop('return_ci', False)
        # return_raw = kwargs.pop('return_raw', False) #
        X_mod = self._convert_input(X)
        X_mod = self._extend_missing_features(X_mod, self.feature_names_in_)
        if return_std:
            preds = self.ci_estimator.predict(X_mod, **kwargs)
            return preds  # Alread reversed
            # return self.scope_transformer.reverse_transform(preds)

        X_new = self._preprocess(X_mod, **kwargs).filter(regex='^ft', axis=1)
        preds = self.estimator.predict(X_new, **kwargs)
        return self.scope_transformer.reverse_transform(preds)

    def fit(self, X, y, **kwargs) -> Self:
        st = time.time()
        self._set_meta(X)
        is_na = np.isnan(y)
        X = self._preprocess(X, **kwargs)
        y = self._transform_scope(y, **kwargs)
        self.estimator = self.estimator.set_params(**self.params).fit(X[~is_na], y[~is_na], **kwargs)
        et = time.time()
        elapsed_time = et - st
        self.logger.info(f'Fit function is completed with execution time: {elapsed_time} seconds')
        self.stats["fit"] = self._gather_stats(X, elapsed_time)
        return self

    def fit_confidence(self, X, y, **kwargs) -> Self:
        st = time.time()
        is_na = np.isnan(y)
        # X = self._preprocess(X, **kwargs)
        # y = self._transform_scope(y, **kwargs)
        X, y = X[~is_na], y[~is_na]
        self.ci_estimator = self.ci_estimator.fit(X, y, **kwargs)
        et = time.time()
        elapsed_time = et - st
        self.logger.info(f'Fit_confidence function is completed with execution time: {elapsed_time} seconds')
        self.stats["fit_confidence"] = self._gather_stats(X, elapsed_time)
        return self

    # TODO: Implement this function
    def evaluate_confidence(self, X, y, **kwargs) -> Self:
        self.ci_estimator.evaluate(X, y, **kwargs)
        return self

    def optimise(self, X, y, **kwargs) -> Self:
        st = time.time()
        df_processed: pd.DataFrame = self.preprocessor.fit_transform(X, y)
        df_reduced: pd.DataFrame = self.feature_selector.fit_transform(df_processed)
        y_transformed = self.scope_transformer.fit_transform(X, y)
        X, y = df_reduced, y_transformed
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        self.params, self.info = self.estimator.optimize(X_train, y_train, X_test, y_test, **kwargs)
        et = time.time()
        elapsed_time = et - st
        self.logger.info(f'Optimize function is completed with execution time: {elapsed_time} seconds')
        self.stats["optimise"] = self._gather_stats(X, elapsed_time)
        return self

    def _gather_stats(self, X_train, elapsed_time):
        return {"time": elapsed_time, "num_datapoints": len(X_train)}

    def evaluate(self, X_train, y_train, X_test, y_test) -> OxariPipeline:
        st = time.time()
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
        self.logger.info(f'sMAPE value of model evaluation: {smape(y_test_transformed, y_pred_test) / 100}')
        et = time.time()
        elapsed_time = et - st
        self.logger.info(f'Evaluate function is completed with execution time: {elapsed_time} seconds')
        return self

    def clone(self) -> Self:
        # TODO: Might introduce problems with bidirectional associations between objects. Needs better conceptual plan.
        return copy.deepcopy(self, {})

    @property
    def evaluation_results(self):
        return {"pipeline": self.name, "stats": self.stats, **self._evaluation_results}

    def _convert_input(self, X:dict|pd.Series|pd.DataFrame|list[dict]):
        """
        Preprocess the input variable X and run the predict function of the model.

        :param X: The input variable (pandas Series, DataFrame, dictionary, or list of dictionaries)
        :return: A pandas DataFrame with the predicted values
        """
        
        # Convert the input variable to a pandas DataFrame
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif isinstance(X, dict):
            X = pd.DataFrame(X, index=[0])
        elif isinstance(X, list) and all(isinstance(item, dict) for item in X):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("The input variable X must be a pandas Series, DataFrame, dictionary, or list of dictionaries.")
        
        return X

    def _extend_missing_features(self, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """
        Extend a DataFrame with columns of features that are not yet present.
        The new columns will be filled with None.

        :param df: The input DataFrame to be extended
        :param feature_names: A list of feature names to ensure in the output DataFrame
        :return: A new DataFrame with the missing feature columns added and filled with None
        """
        
        # Find the missing feature columns
        missing_features = set(feature_names) - set(df.columns)
        if not len(missing_features):
            return df.copy()
        
        if len(missing_features):
            self.logger.warning(f"Features {list(missing_features)} were missing in the input. They are filled with 'None'. ")

            
        # Create a new DataFrame with the same index and the missing feature columns filled with None
        missing_features_df = pd.DataFrame(columns=list(missing_features), index=df.index)
        
        # Concatenate the input DataFrame and the missing features DataFrame
        extended_df = pd.concat([df, missing_features_df], axis=1)
        
        return extended_df

class Test(OxariPipeline):

    def evaluate(self, X_train, y_train, X_test, y_test) -> Self:
        self.this_is_a_test = True
        return super().evaluate(X_train, y_train, X_test, y_test)


class TestTest(OxariPipeline):

    def evaluate(self, X_train, y_train, X_test, y_test) -> Self:
        return super().evaluate(X_train, y_train, X_test, y_test)


class OxariConfidenceEstimator(OxariScopeEstimator, MultiOutputMixin):

    class DefaultConfidenceEvaluator(OxariEvaluator):

        def evaluate(self, y_true: pd.Series, y_pred: pd.DataFrame, **kwargs):
            # TODO: add docstring here

            # compute metrics of interest
            # https://www.statisticshowto.com/coverage-probability/
            y_hat, y_lower, y_upper = y_pred["pred"], y_pred["lower"].values, y_pred["upper"].values

            coverage_pred = y_hat.between(y_lower, y_upper, inclusive='both').mean()
            coverage = y_true.between(y_lower, y_upper, inclusive='both').mean()

            ranges = y_upper - y_lower
            quantiles = [.1, .25, .5, .75, .90, .95, .99]
            percentile_ranges = np.quantile(ranges, quantiles)
            q_dict = {f"{int(q*100)}%": v for q, v in zip(quantiles, percentile_ranges)}

            # Ideas for other metrics:
            # - Fraction between range and possible intervall. https://link.springer.com/article/10.1007/s10994-014-5453-0
            error_metrics = {
                "evaluator": self.name,
                "mean_coverage_model_prediction": coverage_pred,
                "mean_coverage_ground_truth": coverage,
                "percentile_range": q_dict,
            }
            return error_metrics

    def __init__(self, pipeline: OxariPipeline = None, alpha=0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.pipeline = pipeline
        self.evaluator = kwargs.pop('evaluator', OxariConfidenceEstimator.DefaultConfidenceEvaluator())

    def set_pipeline(self, pipeline: OxariPipeline) -> Self:
        self.pipeline = pipeline
        return self

    def evaluate(self, X, y, **kwargs) -> Self:
        y_ci = self.predict(X)
        # y = self.pipeline._transform_scope(y)
        self._evaluation_results = self.evaluator.evaluate(y, y_ci)
        return self

    @property
    def evaluation_results(self):
        return {"ci_estimator": self.name, **self._evaluation_results}

    def _construct_result(self, top, bottom, preds):
        df = pd.DataFrame()
        df['lower'] = bottom
        df['pred'] = preds
        df['upper'] = top
        return self.pipeline._reverse_scope(df)


class DummyConfidenceEstimator(OxariConfidenceEstimator):
    """
    For Probablistic models that already have a native way to predict the standard deviation.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs) -> "OxariRegressor":
        self.max_ = np.max(y)
        self.min_ = np.min(y)
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs) -> ArrayLike:
        df = pd.DataFrame()
        mean_ = self.pipeline.predict(X)

        df = self._construct_result(self.max_, self.min_, mean_)
        return df


class OxariMetaModel(OxariRegressor, MultiOutputMixin, abc.ABC):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pipelines: Dict[str, OxariPipeline] = {}
        self.creation_time = dt.datetime.utcnow()
        self.python_version = sys.version
        self.pickle_package = pkl.__package__
        self.pickle_version = pkl.__version__

    def add_pipeline(self, scope: int, pipeline: OxariPipeline, **kwargs) -> "OxariMetaModel":
        self.check_scope(scope)
        self.pipelines[f"scope_{scope}"] = pipeline
        return self

    def check_scope(self, scope):
        if not isinstance(scope, int):
            self.logger.error(f"Exception: 'scope' is not an int: {scope}")
            raise Exception(f"'scope' is not an int: {scope}")
        if not ((scope > 0) and (scope < 4)):
            self.logger.error(f"Exception: 'scope' is not between either 1, 2 or 3: {scope}")
            raise Exception(f"'scope' is not between either 1, 2 or 3: {scope}")

    def get_pipeline(self, scope: int) -> OxariPipeline:
        return self.pipelines[f"scope_{scope}"]

    def fit(self, X:pd.DataFrame, y=None, **kwargs):
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
        # TODO: Remove comments
        scope = kwargs.pop("scope", "all")
        # X_aligned = self._convert_input(X)
        # X_new = X_aligned.filter(regex='^ft', axis=1)
        if scope == "all":
            # X_extended = self._extend_missing_features(X_new, self.feature_names_in_)
            return self._predict_all(X, **kwargs)
        # X_extended = self._extend_missing_features(X_new, self.get_pipeline(scope).feature_names_in_)
        return self.get_pipeline(scope).predict(X, **kwargs)

    def get_features(self, scope:int=None) -> ArrayLike:
        if not scope:
            all_features = []
            for scope_str, estimator in self.pipelines.items():
                all_features.extend(estimator.feature_names_in_)
            return list(set(all_features))
        pipeline = self.get_pipeline(scope)
        return pipeline.feature_names_in_

    @property
    def feature_names_in_(self):
        return self.get_features()

    def _predict_all(self, X, **kwargs) -> ArrayLike:
        result = pd.DataFrame()
        return_ci = kwargs.pop('return_ci', False)
        if return_ci:
            for scope_str, pipeline in self.pipelines.items():
                y_pred = pipeline.predict(X, return_ci=return_ci, **kwargs)
                y_pred.columns = [f"{scope_str}_{col}" for col in y_pred.columns]
                result = pd.concat([result, y_pred], axis=1)
            return result

        for scope_str, pipeline in self.pipelines.items():
            y_pred = pipeline.predict(X, **kwargs)
            result[scope_str] = y_pred
        return result

    def collect_eval_results(self) -> List[dict]:
        results = []

        for scope, pipeline in self.pipelines.items():
            results.append({"scope": scope, **pipeline.evaluation_results})

        return results

    # def _convert_input(self, X:dict|pd.Series|pd.DataFrame|list[dict]):
    #     """
    #     Preprocess the input variable X and run the predict function of the model.

    #     :param X: The input variable (pandas Series, DataFrame, dictionary, or list of dictionaries)
    #     :return: A pandas DataFrame with the predicted values
    #     """
        
    #     # Convert the input variable to a pandas DataFrame
    #     if isinstance(X, pd.Series):
    #         X = X.to_frame().T
    #     elif isinstance(X, dict):
    #         X = pd.DataFrame(X, index=[0])
    #     elif isinstance(X, list) and all(isinstance(item, dict) for item in X):
    #         X = pd.DataFrame(X)
    #     elif not isinstance(X, pd.DataFrame):
    #         raise ValueError("The input variable X must be a pandas Series, DataFrame, dictionary, or list of dictionaries.")
        
    #     return X

    # def _extend_missing_features(self, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    #     """
    #     Extend a DataFrame with columns of features that are not yet present.
    #     The new columns will be filled with None.

    #     :param df: The input DataFrame to be extended
    #     :param feature_names: A list of feature names to ensure in the output DataFrame
    #     :return: A new DataFrame with the missing feature columns added and filled with None
    #     """
        
    #     # Find the missing feature columns
    #     missing_features = set(feature_names) - set(df.columns)
    #     if not len(missing_features):
    #         return df.copy()
        
    #     if len(missing_features):
    #         self.logger.warning(f"Features {list(missing_features)} were missing in the input. They are filled with 'None'. ")

            
    #     # Create a new DataFrame with the same index and the missing feature columns filled with None
    #     missing_features_df = pd.DataFrame(columns=list(missing_features), index=df.index)
        
    #     # Concatenate the input DataFrame and the missing features DataFrame
    #     extended_df = pd.concat([df, missing_features_df], axis=1)
        
    #     return extended_df

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = {}

    @abc.abstractmethod
    def fit(self, X, y=None) -> Self:
        return self

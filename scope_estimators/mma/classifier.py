from typing_extensions import Self
import lightgbm as lgb
import numpy as np
import optuna
from scipy import stats as sc
from base import (DefaultClassificationEvaluator, OxariClassifier, OxariOptimizer)
# from model.abstract_base_class import MLModelInterface
# from model.misc.hyperparams_tuning import tune_hps_classifier
# from model.misc.ML_toolkit import add_bucket_label,check_scope
from base.metrics import classification_metric
from sklearn.metrics import (classification_report, confusion_matrix, balanced_accuracy_score)
import pandas as pd



class BucketClassifierEvauator(DefaultClassificationEvaluator):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def evaluate(self, y_test, y_pred, **kwargs):
        """

        Computes 3 flavors of accuracy: Vanilla, AdjacentLenient, AdjacentStrict

        Each accuracy computation is scope and buckets specific

        Appends and saves the results to model/metrics/error_metrics_class.csv

        """
        n_buckets = kwargs.get('n_buckets', len(np.unique(y_test)))
        error_metrics = {
            # vanilla accuracy is inherited from DefaultClassificationEvaluator
            "adj_lenient_acc": self.lenient_adjacent_accuracy_score(y_test, y_pred),
            "adj_strict_acc": self.strict_adjacent_accuracy_score(y_test, y_pred, n_buckets),
        }
        # TODO: This is the better way to propagate the information. Not trickle-down but bottom up

        
        return {**super().evaluate(y_test, y_pred), **error_metrics}

    def lenient_adjacent_accuracy_score(self, y_true, y_pred):
        # if true == 0 and pred == 1 --> CORRECT!
        # if true == 9 and pred == 8 --> CORRECT!
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        return np.sum(np.abs(y_pred - y_true) <= 1) / len(y_pred)

    def strict_adjacent_accuracy_score(self, y_true, y_pred, n_buckets):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        # True = 0 and Pred = 1  ====> FALSE
        # True = 1 and Pred = 0  ====> FALSE
        # True = 9 and Pred = 8  ====> FALSE
        # True = 8 and Pred = 9  ====> FALSE
        selector_bottom = ((y_true == 0) & (y_pred == 1)) | ((y_true == 1) & (y_pred == 0))
        selector_top = ((y_true == n_buckets - 1) & (y_pred == n_buckets - 2)) | ((y_true == n_buckets - 2) & (y_pred == n_buckets - 1))
        selector_inbetween = ~(selector_bottom | selector_top)

        correct_bottom = y_true[selector_bottom] == y_pred[selector_bottom]
        correct_top = y_true[selector_top] == y_pred[selector_top]
        correct_adjacency = np.abs(y_true[selector_inbetween] - y_pred[selector_inbetween]) <= 1
        return (correct_bottom.sum() + correct_top.sum() + correct_adjacency.sum()) / len(y_pred)


class ClassifierOptimizer(OxariOptimizer):

    def __init__(self, n_trials=2, n_startup_trials=1, sampler=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.sampler = sampler or optuna.samplers.TPESampler(n_startup_trials=self.n_startup_trials, warn_independent_sampling=False)

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
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
            study_name=f"classifier_hp_tuning",
            direction="maximize",
            sampler=self.sampler,
        )

        # running optimization
        # trials is the full number of iterations
        study.optimize(lambda trial: self.score_trial(trial, X_train, y_train, X_val, y_val), n_trials=self.n_trials, show_progress_bar=False)

        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        # df.to_csv(OPTUNA_DIR / f"df_optuna_hps_CL_{self.scope}_buckets_{self.n_buckets}.csv", index=False)

        # save the study so that we can plot the results
        # joblib.dump(study, OPTUNA_DIR / f"optuna_hps_CL_{self.scope}_{self.n_buckets}_buckets.pkl")

        return study.best_params, df

    def score_trial(self, trial: optuna.Trial, X_train, y_train, X_val, y_val):

        # TODO: add docstring here pls
        y_train = y_train.ravel()
        param_space = {
            'max_depth': trial.suggest_int('max_depth', 3, 21, 3),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9, step=0.1),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 5, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9, step=0.1),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int("n_estimators", 100, 500, 100),
        }

        cl = lgb.LGBMClassifier(**param_space)
        cl.fit(X_train, y_train)

        y_pred = cl.predict(X_val)

        # choose weighted average
        val = classification_metric(y_true=y_val, y_pred=y_pred)
        return val


class BucketClassifier(OxariClassifier):

    def __init__(self, n_buckets=10, **kwargs):
        super().__init__(**kwargs)
        self.n_buckets = n_buckets
        self._estimator = lgb.LGBMClassifier(**kwargs)
        self.bucket_metrics_ = {"scores":{}}

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        best_params, info = self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)
        return best_params, info

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred)

    def fit(self, X, y, **kwargs) -> Self:
        self._estimator.set_params(**self.params).fit(X, y.ravel())
        self.bucket_metrics_ = classification_report(y, self._estimator.predict(X), output_dict=True)
        cls_report_df = pd.DataFrame(classification_report(y, self._estimator.predict(X), output_dict=True)).transpose()
        cnf_df = pd.DataFrame(confusion_matrix(y, self._estimator.predict(X)))
        cnf_df.index = cnf_df.index.astype(float)
        cnf_df.index = cnf_df.index.astype(str)
        classification_metrics = cls_report_df.merge(cnf_df, how='left', left_index=True, right_index=True)
        self.bucket_metrics_ = classification_metrics
        
        return self

    def predict(self, X, **kwargs):
        return self._estimator.predict(X)

    def set_params(self, **params):
        self.params = params
        return self

    def get_config(self, deep=True):
        return {**self.params}


class UnderfittedBucketClassifier(BucketClassifier):

    def __init__(self, n_buckets=10, **kwargs):
        self.n_buckets = n_buckets
        self._estimator = lgb.LGBMClassifier(**kwargs)

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        # best_params, info = self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)
        best_params = {'max_depth': 1, 'num_leaves': 2, 'colsample_bytree': 1.0, 'min_child_weight': 1, 'subsample': 1.0, 'learning_rate': 10, 'n_estimators': 1}
        info = None
        return best_params, info


class RandomGuessBucketClassifier(BucketClassifier):

    def fit(self, X, y, **kwargs) -> Self:
        self.highest_class_ = np.max(y)
        return self

    def predict(self, X, **kwargs):
        return np.random.randint(0, self.highest_class_ + 1, size=len(X))


class MajorityBucketClassifier(BucketClassifier):
    # NOTE: Because the regressor uses all data if none of the classes are selected, the regressor basically trains on all data at once.
    def fit(self, X, y, **kwargs) -> Self:
        self.mode_ = sc.mode(np.array(y).flatten()).mode[0]
        return self

    def predict(self, X, **kwargs):
        return np.ones(len(X)) * self.mode_
from typing_extensions import Self
import optuna
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

from base.metrics import classification_metric
from base.oxari_types import ArrayLike
from scope_estimators.mini_model_army import MiniModelArmyEstimator
from scope_estimators.mma.classifier import BucketClassifier, ClassifierOptimizer

class IndustryClassifier(BucketClassifier):
    def __init__(self, industry_column='ft_catm_industry_name', **kwargs):
        super().__init__(n_buckets = None, **kwargs)
        self.industry_column = industry_column
        
class IndustryClassifierOptimizer(ClassifierOptimizer):
    def __init__(self, industry_column='ft_catm_industry_name', **kwargs) -> None:
        super().__init__(**kwargs)
        self.industry_column = industry_column

    def score_trial(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val):

        # TODO: add docstring here pls
        y_train = y_train.ravel()

        # TODO: the param space should be defined as an attribute of the class {review this idea}
        param_space = {
            'max_depth': trial.suggest_int('max_depth', 3, 21, step=3),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9, step=0.1),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 5, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9, step=0.1),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int("n_estimators", 100, 500, step=100),
        }

        cl = lgb.LGBMClassifier(**param_space)
        cl.fit(X_train.drop(self.industry_column, axis=1), y_train)

        y_pred = cl.predict(X_val.drop(self.industry_column, axis=1))

        # choose weighted average
        val = classification_metric(y_true=y_val, y_pred=y_pred)
        return val

class IndustryModelArmyEstimator(MiniModelArmyEstimator):
    def __init__(self, n_trials=1, n_startup_trials=1, cls_optimizer=None, rgs_optimizer=None, **kwargs):
        super().__init__(n_trials, n_startup_trials, **kwargs)
        self.bucket_cl: IndustryClassifier = IndustryClassifier().set_optimizer(cls_optimizer or IndustryClassifierOptimizer(n_trials=self.n_trials, n_startup_trials=self.n_startup_trials)).set_evaluator(DefaultClassificationEvaluator())
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.top_industries = None
        
    def _identify_top_industries(self, X: pd.DataFrame):
        industry_counts = X[self.bucket_cl.industry_column].value_counts()
        self.top_industries = industry_counts.nlargest(20).index.tolist()
        
    def _transform_industry_column(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        X_transformed[self.bucket_cl.industry_column] = X_transformed[self.bucket_cl.industry_column].apply(lambda x: x if x in self.top_industries else 'Other')
        return X_transformed

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> Self:
        self.n_features_in_ = X.shape[1]
        
        self._identify_top_industries(X)

        X_transformed = self._transform_industry_column(X)

        y_binned =  self.label_encoder.fit_transform(X_transformed[self.bucket_cl.industry_column])
        self.bucket_cl: IndustryClassifier = self.bucket_cl.set_params(**self.params.get("cls", {})).fit(X_transformed.drop(self.bucket_cl.industry_column, axis=1), y_binned)
        # Does it make sense for the regressor to transform y as well? Or is it enough to have the groups as y_binned to restrict the number of voting regressors?
        self.bucket_rg = self.bucket_rg.set_params(**self.params.get("rgs", {})).fit(X_transformed, y, groups=y_binned)
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        X_transformed = self._transform_industry_column(X)
        y_binned_pred = self.bucket_cl.predict(X_transformed.drop(self.bucket_cl.industry_column, axis=1))
        y_pred = self.bucket_rg.predict(X_transformed, groups=y_binned_pred)
        return y_pred

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        best_params_cls = {}
        info_cls = pd.DataFrame()
        
        X_train_transformed = self._transform_industry_column(X_train)
        X_val_transformed = self._transform_industry_column(X_val)

        self.label_encoder = self.label_encoder.fit(X_train_transformed[self.bucket_cl.industry_column])
        y_train_binned = self.label_encoder.transform(X_train_transformed[self.bucket_cl.industry_column]).reshape(-1, 1)
        y_val_binned = self.label_encoder.transform(X_val_transformed[self.bucket_cl.industry_column]).reshape(-1, 1)
        
        self.n_buckets = len(self.label_encoder.classes_)

        best_params_cls, info_cls = self.bucket_cl.optimize(X_train_transformed, y_train_binned, X_val_transformed, y_val_binned, **kwargs)
        best_params_rgs, info_rgs = self.bucket_rg.optimize(X_train_transformed, y_train, X_val_transformed, y_val, grp_train=y_train_binned, grp_val=y_val_binned, **kwargs)

        # return {**best_params_cls,**best_params_rgs}, {"classifier":info_cls, "regressor":info_rgs}
        return {"cls": best_params_cls, "rgs": best_params_rgs}, {"classifier": info_cls, "regressor": info_rgs}

    def evaluate(self, y_true, y_pred, **kwargs):
        X_test: pd.DataFrame = kwargs.get("X_test")
        X_test_transformed = self._transform_industry_column(X_test)
        
        y_true_bins = self.label_encoder.transform(X_test_transformed[self.bucket_cl.industry_column])

        y_pred_cl = self.bucket_cl.predict(X_test_transformed.drop(self.bucket_cl.industry_column, axis=1))
        results_cl = self.bucket_cl.evaluate(y_true_bins, y_pred_cl)

        y_pred_rg = self.bucket_rg.predict(X_test_transformed, groups=y_true_bins)
        results_rg = self.bucket_rg.evaluate(y_true, y_pred_rg)

        results_end_to_end = self._evaluator.evaluate(y_true, y_pred)
        combined_results = {"n_buckets": self.n_buckets, "classifier": results_cl, "regressor": results_rg, **results_end_to_end}
        return combined_results

import random
from itertools import product

import pandas as pd
import sklearn

from base import DummyScaler, OxariDataManager, OxariPipeline
from base.helper import LogarithmScaler
from datasources.core import DefaultDataManager
from feature_reducers.core import DummyFeatureReducer, PCAFeatureReducer
from imputers import KMeansBucketImputer, RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import (BaselinePreprocessor, IIDPreprocessor,
                           ImprovedBaselinePreprocessor)
from scope_estimators import (BaselineEstimator, MiniModelArmyEstimator,
                              SupportVectorEstimator)


# NOTE: IIDPreprocessor seems like a much better for most models
class Runner(object):
    def __init__(self, optimize_data, fit_data, eval_data) -> None:
        self.optimize_data = optimize_data
        self.fit_data = fit_data
        self.eval_data = eval_data

    def run(self, model: OxariPipeline):
        try:
            return model.optimise(*self.optimize_data).fit(*self.fit_data).evaluate(*self.eval_data)
        except Exception as e:
            print("Something went wrong!")
            print(e)
            return model


if __name__ == "__main__":

    dataset = DefaultDataManager().run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    X = dataset.get_features(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3
    model_list = [
        BaselineEstimator(),
        SupportVectorEstimator(),
        MiniModelArmyEstimator(3),
        MiniModelArmyEstimator(4),
        MiniModelArmyEstimator(5),
        MiniModelArmyEstimator(6),
    ]
    all_imputers = [
        RevenueQuantileBucketImputer,
        KMeansBucketImputer,
    ]
    all_feature_reducers = [
        PCAFeatureReducer,
        DummyFeatureReducer,
    ]
    all_preprocessors = [
        IIDPreprocessor,
        BaselinePreprocessor,
        ImprovedBaselinePreprocessor,
    ]
    all_scope_scalers = [
        LogarithmScaler(),
        DummyScaler(),
    ]

    all_combinations = list(product(model_list, all_preprocessors, all_imputers, all_feature_reducers, all_scope_scalers, range(5)))

    all_models = [
        DefaultPipeline(
            name=f"{model.name}-{idx}-{model.n_buckets}",
            preprocessor=Preprocessor(),
            feature_reducer=FtReducer(),
            imputer=Imputer(buckets_number=random.randint(3, 7)),
            scope_estimator=model,
            scope_transformer=scope_scaler,
        ) for model, Preprocessor, Imputer, FtReducer, scope_scaler, idx in all_combinations
    ]

    # random.shuffle(all_models)

    all_evaluations = []
    all_models_trained = []
    # TODO: how many threads? all the models in oneppol? look into this!
    optimize_data = SPLIT_1.train.X, SPLIT_1.train.y
    fit_data = SPLIT_1.train.X, SPLIT_1.train.y
    eval_data = SPLIT_1.rem.X, SPLIT_1.rem.y, SPLIT_1.val.X, SPLIT_1.val.y

    runner = Runner(optimize_data, fit_data, eval_data)
    for model in all_models:
        print(f"\n====================== MODEL: {model.name}")
        model = runner.run(model)
        all_models_trained.append(model.evaluation_results)
        eval_results = pd.json_normalize(all_models_trained)
        eval_results.to_csv('local/eval_results/results_sequential_mma.csv')
        try:
            print(model.predict(SPLIT_1.test.X))
        except sklearn.exceptions.NotFittedError as e:
            continue

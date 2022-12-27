
import time
from datetime import date
from pipeline.core import DefaultPipeline, CVPipeline
from dataset_loader.csv_loader import CSVDataManager
from base import OxariDataManager, OxariSavingManager, LocalMetaModelSaver, LocalLARModelSaver, LocalDataSaver,S3MetaModelSaver, S3DataSaver, S3LARModelSaver
from preprocessors import BaselinePreprocessor, ImprovedBaselinePreprocessor, IIDPreprocessor, NormalizedIIDPreprocessor
from postprocessors import ScopeImputerPostprocessor
from base import BaselineConfidenceEstimator, JacknifeConfidenceEstimator
from imputers import BaselineImputer, KMeansBucketImputer, RevenueBucketImputer, RevenueExponentialBucketImputer, RevenueQuantileBucketImputer, RevenueParabolaBucketImputer
from feature_reducers import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer, IsomapFeatureSelector, MDSSelector
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, SupportVectorEstimator, DummyEstimator, PredictMeanEstimator, BaselineEstimator, LinearRegressionEstimator, BayesianRegressionEstimator, GLMEstimator, IndependentFeatureVotingRegressionEstimator
from base.confidence_intervall_estimator import ProbablisticConfidenceEstimator, BaselineConfidenceEstimator
from base import helper
from base import OxariMetaModel
import pandas as pd
import joblib as pkl
import io
import pathlib
from pprint import pprint
import numpy as np

class PredictionJumpEvaluation():
    def __init__(self, estimator:OxariMetaModel) -> None:
        self.estimator = estimator 
        
    def run(self, dataset:OxariDataManager):
        pass
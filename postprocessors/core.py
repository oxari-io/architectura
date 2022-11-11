from base import OxariPostprocessor, OxariScopeEstimator, OxariDataManager, OxariModel
import pandas as pd
from tqdm import tqdm
import numpy as np

class DefaultPostprocessor(OxariPostprocessor):
    
    def __init__(self, estimator:OxariModel, **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator
    
        
    def fit(self, X: OxariDataManager, y=None, **kwargs):

        
        # we are only interested in the most recent years
        data = X.loc[X["year"].isin(list(range(2016, 2022)))]
        predicted_scope_1 = self.estimator.predict(X, scope=1)
        predicted_scope_2 = self.estimator.predict(X, scope=2)
        predicted_scope_3 = self.estimator.predict(X, scope=3)

        # adding a column that indicates whether the scope has been predicted or was reported
        data['predicted_s1'] = np.where(data['scope_1'].isnull(), True, False)
        data['predicted_s2'] = np.where(data['scope_2'].isnull(), True, False)
        data['predicted_s3'] = np.where(data['scope_3'].isnull(), True, False)

        # filling missing values of scopes with model predictions
        data["scope_1"] = data["scope_1"].fillna(predicted_scope_1)
        data["scope_2"] = data["scope_2"].fillna(predicted_scope_2)
        data["scope_3"] = data["scope_3"].fillna(predicted_scope_3)

        # retrieving only the relevant columns
        data = data[["isin", "year", "scope_1", "scope_2", "scope_3", "predicted_s1", "predicted_s2", "predicted_s3"]]

        # how many unique companies?
        print("Number of unique companies in the data: " , len(data["isin"].unique()))

        # data.to_csv("model/update_elastic/elastic_data/estimated_scopes_classifier_approach.csv", index = False)  
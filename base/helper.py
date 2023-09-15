from __future__ import annotations
from enum import Enum

from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from base.common import OxariScopeTransformer, OxariFeatureTransformer
from base.oxari_types import ArrayLike
from typing_extensions import Self

def replace_ft_num(X:pd.DataFrame, X_new:ArrayLike):
    X_tmp = X.copy()
    X_tmp[X.filter(regex='^ft_num', axis=1).columns] = X_new
    return X_tmp

def mock_data():
    data = data_point()
    df = pd.Series(data).to_frame().T.sort_index(axis=1)
    return df

def mock_data_dict():
    return data_point()

class ExampleModifications(str, Enum):
    NORMAL = "normal"
    MISSING_CATS = "missing_categoricals"
    MISSING_NUMS = "missing_numericals"

def data_point(drop_rate:float=0.0):
    num_data = {
        "ft_numc_stock_return": -0.03294267654418,
        "ft_numc_total_assets": 0.0,
        "ft_numc_ppe": 1446082.0,
        "ft_numc_roa": 0.0145850556922605,
        "ft_numc_roe": 0.34,
        "ft_numc_total_liab": 1421.287,
        "ft_numc_equity": 1124.699,
        "ft_numc_revenue": 503.999999604178,
        "ft_numc_market_cap": 635.348579719647,
        "ft_numc_inventories": 13991.0,
        "ft_numc_net_income": 34.9999999725123,
        "ft_numc_cash": 231.043,
        "ft_numd_employees": 1000,
        "ft_numc_rd": 500,
        "ft_numc_prior_tg_numc_scope_1": 26523,
        "ft_numc_prior_tg_numc_scope_2": 50033,
        "ft_numc_prior_tg_numc_scope_3": 11111,
        "key_year": 2019.0,
        "key_isin": "FR0000051070",
        "tg_numc_scope_1": None,
        "tg_numc_scope_2": None,
        "tg_numc_scope_3": None,
    }

    cat_data = {
        "ft_catm_industry_name": "Industrial Conglomerates",
        "ft_catm_country_code": "USA",
        "ft_catm_sector_name": "Industrials",
    }

    tmp = {**num_data, **cat_data}
    remove = np.random.rand(len(tmp.keys())) < drop_rate
    result = {k:v for is_remove, (k, v) in zip(remove, tmp.items()) if (not is_remove) or k.startswith("key_") or k.startswith("tg_")}
    return result

class OxariFeatureTransformerWrapper(OxariFeatureTransformer):
    def __init__(self, transformer=None, name=None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.transformer = transformer

    def fit(self, X:ArrayLike, y=None, **kwargs) -> Self:
        self.feature_names_in_ = X.columns.tolist()
        return self.transformer.fit(X, y, **kwargs)
    
    def transform(self, X:ArrayLike, **kwargs) -> ArrayLike:
        X_result = X.copy()
        X_ft = X[self.feature_names_in_]
        X_new = pd.DataFrame(self.transformer.transform(X_ft, **kwargs), columns=X_ft.columns, index=X_ft.index)
        X_result[X_new.columns]=X_new[X_new.columns].values
        return X_result
    
    def reverse_transform(self, X, **kwargs) -> ArrayLike:
        return self.transformer.reverse_transform(X, **kwargs)

class BucketScopeDiscretizer(OxariScopeTransformer):

    def __init__(self, n_buckets, prefix="bucket_", **kwargs) -> None:
        super().__init__()
        self.n_buckets = n_buckets
        self.prefix = prefix
        encode = kwargs.pop("encode", "ordinal")
        strategy = kwargs.pop("strategy", "quantile")
        self.discretizer = KBinsDiscretizer(n_buckets, encode=encode, strategy=strategy, **kwargs)
        self.info = {"bucket_counts": {}}

    def __repr__(self):
        return f"@[{self.__class__.__name__}]{self.info}"

    def fit(self, X, y=None):
        results = self.discretizer.fit_transform(np.array(y)[:, None])
        lbls, cnts = np.unique(results, return_counts=True)
        self.info["bucket_counts"] = dict(zip(lbls, cnts))
        return self

    def transform(self, y, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        return self.discretizer.transform(np.array(y)[:, None], **kwargs)

    def reverse_transform(self, y, **kwargs) -> ArrayLike:
        return super().reverse_transform(y, **kwargs)


class SingleBucketScopeDiscretizer(BucketScopeDiscretizer):
    """This is only for experimental purposes"""

    def __init__(self, n_buckets, prefix="bucket_", **kwargs) -> None:
        self.n_buckets = n_buckets
        self.prefix = prefix
        self.info = {"bucket_counts": {}}

    def fit(self, X, y=None):
        self.info["bucket_counts"][0] = len(y)
        return self

    def transform(self, y, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        return np.zeros_like(np.array(y)[:, None])


# TODO: Checkout TargetTransformer and QuantileTransformer
class LogTargetScaler(OxariScopeTransformer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def transform(self, y, **kwargs) -> ArrayLike:
        return np.log1p(y.copy())

    def reverse_transform(self, y, **kwargs) -> ArrayLike:
        transfomed_y = np.expm1(y.copy())
        # This is a failsafe in case the model spits out a number waaaay too high and np.expm1 becomes infinite
        position_inf = np.isinf(transfomed_y)
        transfomed_y[position_inf] = np.finfo(float).max
        return transfomed_y


class ArcSinhTargetScaler(OxariScopeTransformer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def transform(self, y, **kwargs) -> ArrayLike:
        return np.arcsinh(y.copy())

    def reverse_transform(self, y, **kwargs) -> ArrayLike:
        return np.sinh(y.copy())


class DummyTargetScaler(OxariScopeTransformer):

    def transform(self, y, **kwargs) -> ArrayLike:
        return super().transform(y, **kwargs)

    def reverse_transform(self, y, **kwargs) -> ArrayLike:
        return super().reverse_transform(y, **kwargs)


class ArcSinhScaler(OxariFeatureTransformer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def transform(self, X, y=None, **kwargs) -> ArrayLike:
        return np.arcsinh(X.copy())

    def reverse_transform(self, X, **kwargs) -> ArrayLike:
        return np.sinh(X.copy())


class DummyFeatureScaler(OxariFeatureTransformer):

    def transform(self, X, y=None, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)

    def reverse_transform(self, X, **kwargs) -> ArrayLike:
        return super().reverse_transform(X, **kwargs)

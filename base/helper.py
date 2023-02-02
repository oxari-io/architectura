from __future__ import annotations

from typing import Union
import numpy as np
import pandas as pd
from base.common import OxariScopeTransformer
from base.oxari_types import ArrayLike
from sklearn.preprocessing import KBinsDiscretizer


def mock_data():
    num_data = {
        "stock_return": -0.03294267654418,
        "total_assets": 0.0,
        "ppe": 1446082.0,
        "year": 2019.0,
        "roa": 0.0145850556922605,
        "roe": 0.34,
        "total_liab": 1421.287,
        "equity": 1124.699,
        "revenue": 503.999999604178,
        "market_cap": 635.348579719647,
        "inventories": 13991.0,
        "net_income": 34.9999999725123,
        "cash": 231.043,
        "employees": 1000,
        "rd_expenses": 500,
        "isin": "FR0000051070",
        "scope_1": None,
        "scope_2": None,
        "scope_3": None,
    }

    cat_data = {
        "industry_name": "Industrial Conglomerates",
        # "company_name": "Aboitiz Equity Ventures Inc",
        "country_name": "Philippines",
        "sector_name": "Industrials",
    }
    df = pd.Series({**num_data, **cat_data}).to_frame().T.sort_index(axis=1)
    return df


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
        self.info["bucket_counts"][0]=len(y)
        return self

    def transform(self, y, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        return np.zeros_like(np.array(y)[:, None])


# TODO: Checkout TargetTransformer and QuantileTransformer
class LogarithmScaler(OxariScopeTransformer):
    def __init__(self, name=None, **kwargs) -> None:
        super().__init__(name, **kwargs)

    def transform(self, y, **kwargs) -> ArrayLike:
        return np.log1p(y.copy())

    def reverse_transform(self, y, **kwargs) -> ArrayLike:
        return np.expm1(y.copy())

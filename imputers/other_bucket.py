from typing import Union

import numpy as np
import pandas as pd

from base.common import OxariImputer
from imputers.numerical import NumericalStatisticsImputer

from .core import BucketImputerBase

EXAMPLE_STRING = """
self.statistics = {
                    '0...10': {
                        'ft_numc_revenue':{
                            'min': -10,
                            'max': 1000,
                            'median': 100,
                            'mean': 450,
                        }
                    },
                    '10...20': {
                        ...
                    },
                    ...,
                }
"""


class TotalAssetsBucketImputer(NumericalStatisticsImputer):
    def __init__(self, buckets_number: int = 3, **kwargs):
        super().__init__("ft_numc_total_assets", buckets_number, **kwargs)


class TotalAssetsQuantileBucketImputer(TotalAssetsBucketImputer):

    def _get_threshold(self, buckets_number, min_, max_, data):
        x = np.linspace(0, 1, buckets_number + 1)
        threshold = np.quantile(data, x)
        return threshold


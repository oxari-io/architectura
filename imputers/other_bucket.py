from typing import Union

import numpy as np
import pandas as pd

from base.common import OxariImputer
from imputers.numerical import NumericalStatisticsImputer, NumericalStatisticsQuantileBucketImputer

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


class TotalAssetsQuantileBucketImputer(NumericalStatisticsQuantileBucketImputer):

    def __init__(self, num_buckets: int = 3, **kwargs):
        super().__init__("ft_numc_total_assets", num_buckets, **kwargs)

    @property
    def name(self):
        return f"{self.__class__.__name__}:{self.bucket_number}-buckets"


class TotalLiabilitiesQuantileBucketImputer(NumericalStatisticsQuantileBucketImputer):

    def __init__(self, num_buckets: int = 3, **kwargs):
        super().__init__("ft_numc_total_liabilities", num_buckets, **kwargs)

    @property
    def name(self):
        return f"{self.__class__.__name__}:{self.bucket_number}-buckets"

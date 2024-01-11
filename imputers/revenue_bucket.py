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


class RevenueBucketImputer(NumericalStatisticsImputer):
    def __init__(self, buckets_number: int = 3, **kwargs):
        super().__init__("ft_numc_revenue", buckets_number, **kwargs)

    def _get_threshold(self, buckets_number, min_, max_, data):
        return np.linspace(min_, max_, buckets_number + 1)

class RevenueExponentialBucketImputer(RevenueBucketImputer):

    def _get_threshold(self, buckets_number, min_, max_, data):
        return np.geomspace(min_ - min_ + 1, max_ - min_ + 1, buckets_number + 1) + min_ - 1


class RevenueQuantileBucketImputer(RevenueBucketImputer):

    def _get_threshold(self, buckets_number, min_, max_, data):
        x = np.linspace(0, 1, buckets_number + 1)
        threshold = np.quantile(data, x)
        return threshold


class RevenueParabolaBucketImputer(RevenueBucketImputer):

    def _get_threshold(self, buckets_number, min_, max_, data):
        x = np.arange(buckets_number + 1)
        start, middle, stop = x[0], x[buckets_number // 2], x[-1]
        A, B, C = self.calc_parabola_vertex(start, min_, middle, 0, stop, max_)
        return A * (x**2) + B * x + C

    def calc_parabola_vertex(self, x1, y1, x2, y2, x3, y3):
        '''
        Adapted and modifed to get the unknowns for defining a parabola:
        http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
        Taken from http://chris35wills.github.io/parabola_python/
        '''

        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
        C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

        return A, B, C

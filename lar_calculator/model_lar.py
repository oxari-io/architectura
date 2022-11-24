
import numpy as np
import pandas as pd
from tqdm import tqdm
from base.pipeline import OxariLinearAnnualReduction, OxariDataManager


class OxariLARCalculator(OxariLinearAnnualReduction):
    def __init__(self, dataset: OxariDataManager):
        super().__init__(**kwargs)
        self.dataset = dataset

    def _calculate_lar_by_two_points(self, base_year, target_year, val_base_year, val_target_year):
        # just pctage of change
        # return 100 * (val_target_year - val_base_year) / val_base_year / (target_year - base_year)
        # real reduction
        return 100 * (val_base_year - val_target_year) / abs(val_base_year) / (target_year - base_year)

    def _calculate_aggragated_lar(self, year_range, values_, show_ = False):
        # make sure that we dont use nan values to calculate LAR

        values_ = np.array(values_)
        # idxs = np.where(np.isnan(values_))
        # values_ = np.delete(values_, idxs)

        year_range = np.array(year_range)
        # year_range = np.delete(year_range, idxs)

        # print("YEAR", year_range.shape)
        # print("VALUES", values_.shape)

        slope, intercept, r, p, se = stats.linregress(year_range, values_)

        base_year = year_range[0]
        target_year = year_range[-1]

        val_base_year = base_year * slope + intercept
        val_target_year = target_year * slope + intercept

        X = year_range
        y = values_
        get_val = lambda v: v * slope + intercept
        y_pred = [get_val(v) for v in year_range]
        if show_:
            print(values_)
            print(y_pred)
            plt.scatter(X, y)
            plt.plot(X, y_pred, color="black")
            plt.plot([base_year, target_year], [values_[0], values_[-1]], color="red")
            plt.show()

        return self._calculate_lar_by_two_points(base_year, target_year, val_base_year, val_target_year)


    def calculate_LARs(self, scope_data = None):

        # REVIEWME: why not including 2021?
        # years_range = [y for y in range(2016, 2022)]

        if not scope_data:
            scopes = self.dataset.get_data_by_name(OxariDataManager.IMPUTED)
        else:
            scopes = scope_data

        # making sure that for each isin we have at least 2 datapoints 
        scopes = scopes[scopes.groupby('isin').isin.transform('count') > 1]

        # scopes = scopes[["isin", "year", "scope_1" , "scope_2", "scope_3"]]

        # scopes = scopes.loc[scopes["year"].isin(years_range)]

        scope_columns=["scope_1", "scope_2"]

        scopes["scope_1+2"] = scopes[scope_columns].sum(axis = 1)

        isins = pd.unique(scopes["isin"])

        print("unique isins", len(isins))

        data = pd.DataFrame({"isin" : isins, "lar1_2" : None, "lar_3" : None})

        grouped = scopes.groupby("isin", sort = False)

        for group_name, df_group in tqdm(grouped):

            years = df_group["year"].to_list()
            values_1_2 = df_group["scope_1+2"].to_list()
            values_3 = df_group["scope_3"].to_list()

            data.loc[data["isin"] == group_name, "lar1_2"] = calculate_aggragated_lar(years, values_1_2)
            data.loc[data["isin"] == group_name, "lar_3"] = calculate_aggragated_lar(years, values_3)


        return data

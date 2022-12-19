import numpy as np
import pandas as pd
from tqdm import tqdm
from base import OxariLinearAnnualReduction, OxariDataManager
from sklearn.linear_model import LinearRegression
from base.oxari_types import ArrayLike


def _helper(name, isins, years, emissions):
    lr = LinearRegression()
    lr.fit(years.values[:, None], emissions)
    group_name = isins.values[0]
    base_year = years.values[0]
    target_year = years.values[-1]

    slope = lr.coef_[0]
    intercept = lr.intercept_

    value = pd.Series({
        "isin": group_name,
        "scope_type": name,
        "lar": _compute_lar(base_year, target_year, slope, intercept),
        "slope": slope,
        "intercept": intercept,
        "base_year": base_year,
        "target_year": target_year,
    })
    return value


def _compute_lar(base_year, target_year, slope, intercept):
    val_base_year = base_year * slope + intercept
    val_target_year = target_year * slope + intercept
    change_rate = (val_base_year - val_target_year) / abs(val_base_year)
    percentage_of_change = 100 * change_rate
    number_of_years_spanned = (target_year - base_year)
    return percentage_of_change / number_of_years_spanned


class OxariLARCalculator(OxariLinearAnnualReduction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.dataset = dataset

    def predict(self, X, **kwargs) -> ArrayLike:
        # make sure that we dont use nan values to calculate LAR
        X = X.dropna()
        years, values = X.iloc[:, 0].values, X.iloc[:, 1].values
        # values = np.array(values)
        # year_range = np.array(year_range)

        # slope, intercept, r, p, se = stats.linregress(year_range, values_)
        lr = LinearRegression().fit(years[:, None], values)
        slope, intercept = lr.coef_, lr.intercept_

        base_year = years[0]
        target_year = years[-1]

        lar = _compute_lar(base_year, target_year, slope, intercept)
        return lar

    def transform(self, X, **kwargs) -> ArrayLike:
        new_X = X.copy()
        params = self.params_1.merge(self.params_2, how="left", on="isin", suffixes=("_scope_1_2", "_scope_3"))
        new_X = new_X.merge(params, how="left", on="isin")
        
        return new_X

    def fit(self, X, y=None) -> "OxariLinearAnnualReduction":

        # REVIEWME: why not including 2021?
        # years_range = [y for y in range(2016, 2022)]

        scopes = X

        # making sure that for each isin we have at least 2 datapoints
        scopes = scopes[scopes.groupby('isin').isin.transform('count') > 1]

        # scopes = scopes[["isin", "year", "scope_1" , "scope_2", "scope_3"]]

        # scopes = scopes.loc[scopes["year"].isin(years_range)]

        scope_columns = ["scope_1", "scope_2"]
        scopes = scopes.assign(scope_1_2=scopes[scope_columns].sum(axis=1))

        isins = pd.unique(scopes["isin"])

        print("unique isins", len(isins))

        # data = pd.DataFrame({"isin": isins, "lar1_2": None, "lar_3": None})

        grouped = scopes.groupby("isin", sort=False)
        # grouped.aggregate(self._calculate_aggragated_lar)
        # np.where()

        # collector_1 = []
        # self.scope_1_2 = pd.DataFrame()

        # max_years = scopes.groupby("isin").count().max()[0]
        # padded_nd_array = np.array(list(scopes[["isin", "year", "scope_1_2"]].groupby('isin').apply(pd.DataFrame.to_numpy)
        #             .apply(lambda x: np.pad(x, ((max_years-len(x), 0), (0, 0)), 'empty'))))
        # padded_array_masked = np.ma.masked_equal(padded_nd_array, None)
        # padded_array_masked.data[:,:,0] = 1
        # x = padded_array_masked[:,:,:2]
        # y = padded_array_masked[:,:,2]

        self.params_1 = grouped.apply(lambda df_group: _helper("scope_1_2", df_group["isin"], df_group["year"], df_group["scope_1_2"])).reset_index(drop=True)
        self.params_2 = grouped.apply(lambda df_group: _helper("scope_3", df_group["isin"], df_group["year"], df_group["scope_3"])).reset_index(drop=True)

        # for group_name, df_group in tqdm(grouped):
        # self._helper(lr, group_name, df_group)
        # lr = LinearRegression().fit(df_group["year"], df_group["scope_3"])
        # collector_1.append(("scope_1_2", group_name, lr.coef_, lr.intercept_, df_group.head(1)["year"].values[0], df_group.tail(1)["year"].values[0]))
        # data.loc[data["isin"] == group_name, "lar1_2"] = self.predict(df_group[["year", "scope_1+2"]])
        # data.loc[data["isin"] == group_name, "lar_3"] = self.predict(df_group[["year", "scope_3"]])
        return self

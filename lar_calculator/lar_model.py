import pandas as pd
from sklearn.linear_model import LinearRegression

from base import OxariLinearAnnualReduction
from base.oxari_types import ArrayLike


# TODO: Move those into the class
def _helper(name, isins, years, emissions):
    lr = LinearRegression()
    lr.fit(years.values[:, None], emissions)
    group_name = isins.values[0]
    base_year = years.values[0]
    target_year = years.values[-1]

    slope = lr.coef_[0]
    intercept = lr.intercept_

    value = pd.Series({
        "key_isin": group_name,
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


class OxariUnboundLAR(OxariLinearAnnualReduction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.dataset = dataset

    def predict(self, X, **kwargs) -> ArrayLike:
        # make sure that we dont use nan values to calculate LAR
        X = X.dropna()
        years, values = X.iloc[:, 0].values, X.iloc[:, 1].values

        # slope, intercept, r, p, se = stats.linregress(year_range, values_)
        lr = LinearRegression().fit(years[:, None], values)
        slope, intercept = lr.coef_, lr.intercept_

        base_year = years[0]
        target_year = years[-1]

        lar = _compute_lar(base_year, target_year, slope, intercept)
        return lar

    def transform(self, X, **kwargs) -> ArrayLike:
        new_X = X.copy()
        params = self.params_1.merge(self.params_2, how="left", on="key_isin", suffixes=("_scope_1_2", "_scope_3"))
        new_X = new_X.merge(params, how="left", on="key_isin")
        
        return new_X

    def fit(self, X, y=None) -> "OxariLinearAnnualReduction":

        # REVIEWME: why not including 2021?
        # years_range = [y for y in range(2016, 2022)]

        scopes = X

        # making sure that for each isin we have at least 2 datapoints
        scopes = scopes[scopes.groupby('key_isin')['key_isin'].transform('count') > 1]

        scope_columns = ["tg_numc_scope_1", "tg_numc_scope_2"]
        scopes = scopes.assign(tg_numc_scope_1_2=scopes[scope_columns].sum(axis=1))

        isins = pd.unique(scopes["key_isin"])

        # print("unique isins", len(isins))
        self.logger.debug(f"unique isins: {len(isins)}")

        grouped = scopes.groupby("key_isin", sort=False)

        self.params_1 = grouped.apply(lambda df_group: _helper("tg_numc_scope_1_2", df_group["key_isin"], df_group["key_year"], df_group["tg_numc_scope_1_2"])).reset_index(drop=True)
        self.params_2 = grouped.apply(lambda df_group: _helper("tg_numc_scope_3", df_group["key_isin"], df_group["key_year"], df_group["tg_numc_scope_3"])).reset_index(drop=True)

        return self

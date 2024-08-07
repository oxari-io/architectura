from pathlib import Path
import multiprocessing
import json
import io

USABLE_CPUS = int(multiprocessing.cpu_count()*0.75)

OBJECT_DIR = Path("local/objects")
DATA_DIR = Path("model-data/input")

IMPORTANT_EVALUATION_COLUMNS = [
    "imputer",
    "preprocessor",
    "feature_selector",
    "scope_estimator",
    "scope",
    "test.evaluator",
    "test.sMAPE",
    "train.sMAPE",
    "test.R2",
    "train.R2",
    "test.MAE",
    "train.MAE",
    "test.RMSE",
    "train.RMSE",
    "test.MAPE",
    "train.MAPE",
]

FEATURE_SET_CATEGORICALS = [
    'ft_catm_country_code',
    'ft_catm_industry_name',
    'ft_catm_sector_name',
]

FEATURE_SET_PREV_SCOPE = [
    'ft_numc_prior_tg_numc_scope_1',
    'ft_numc_prior_tg_numc_scope_2',
    'ft_numc_prior_tg_numc_scope_3',
]

# ===== BELOW ARE FEATURE SETS IDENTIFIED USING SIMPLE METHODS =====
# The analysis can be found in notebooks/analyze_feature_set_initial_discovery.py

# This is the set of features used before the 12data integration
FEATURE_SET_OLD = [
    'ft_numc_cash', 'ft_numd_employees', 'ft_numc_equity', 'ft_numc_inventories', 'ft_numc_market_cap', 'ft_numc_net_income', 'ft_numc_ppe', 'ft_numc_rd_expenses',
    'ft_numc_revenue', 'ft_numc_roa', 'ft_numc_roe', 'ft_numc_stock_return', 'ft_numc_total_assets', 'ft_numc_total_liabilities'
] + FEATURE_SET_CATEGORICALS

# This set is a reduction in which features where eliminated iteratively when they had a high correlation (>=0.5) with another feature
FEATURE_SET_CORR_ELIMINATION = [
    'ft_numc_acrued_expenses', 'ft_numc_common_stock_dividends', 'ft_numc_deferred_liabilities', 'ft_numc_diluted_shares_outstanding', 'ft_numc_eps_diluted',
    'ft_numc_financial_assets', 'ft_numc_financing_activities.common_stock_issuance', 'ft_numc_financing_activities.common_stock_repurchase',
    'ft_numc_financing_activities.short_term_debt_issuance', 'ft_numc_goodwill', 'ft_numc_hedging_assets', 'ft_numc_interest_paid',
    'ft_numc_investing_activities_purchase_of_investments', 'ft_numc_investing_actvities.net_acquisitions', 'ft_numc_investing_actvities.net_intangibles',
    'ft_numc_minority_interests', 'ft_numc_operating_activities.accounts_payable', 'ft_numc_operating_activities.other_assets_liabilities',
    'ft_numc_operating_activities.other_non_cash_items', 'ft_numc_other_current_assets', 'ft_numc_other_financing_charges', 'ft_numc_other_income_expense',
    'ft_numc_other_non_current_assets', 'ft_numc_other_non_current_liabilities', 'ft_numc_other_shareholders_equity', 'ft_numc_preferred_stock_dividends',
    'ft_numc_prior_tg_numc_scope_1', 'ft_numc_prior_tg_numc_scope_2', 'ft_numc_prior_tg_numc_scope_3', 'ft_numc_restricted_cash', 'ft_numc_roa', 'ft_numc_roe',
    'ft_numc_tax_payable', 'ft_numc_tresury_stock'
] + FEATURE_SET_CATEGORICALS

# This set is a reduction in which features where eliminated WITHOUT iterative removals when they had a high correlation (>=0.5) with another feature
FEATURE_SET_STRICT_CORR_ELIMINATION = [
    'ft_numc_hedging_assets', 'ft_numc_investing_actvities.net_acquisitions', 'ft_numc_other_financing_charges', 'ft_numc_prior_tg_numc_scope_1', 'ft_numc_prior_tg_numc_scope_2',
    'ft_numc_prior_tg_numc_scope_3', 'ft_numc_roa', 'ft_numc_roe'
] + FEATURE_SET_CATEGORICALS

# This set is a reduction in which the select K-Best method was used. (sklearn implementation)
FEATURE_SET_SELECT_K_BEST = [
    'ft_numc_basic_shares_outstanding', 'ft_numc_deferred_liabilities', 'ft_numc_diluted_shares_outstanding', 'ft_numc_hedging_assets', 'ft_numc_minority_interest',
    'ft_numc_operating_activities.depreciation', 'ft_numc_prior_tg_numc_scope_1', 'ft_numc_prior_tg_numc_scope_2', 'ft_numc_prior_tg_numc_scope_3', 'ft_numc_tax_payable'
] + FEATURE_SET_CATEGORICALS

# This set is a reduction in which Variance Inflation Factor (VIF) technique was used. Removed everything with VIV > 10
FEATURE_SET_VIF_UNDER_05 = json.load(io.open('/res/vif_05.json','w')) + FEATURE_SET_PREV_SCOPE
FEATURE_SET_VIF_UNDER_10 = json.load(io.open('/res/vif_10.json','w')) + FEATURE_SET_PREV_SCOPE
FEATURE_SET_VIF_UNDER_15 = json.load(io.open('/res/vif_15.json','w')) + FEATURE_SET_PREV_SCOPE


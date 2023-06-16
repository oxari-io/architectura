from base.dataset_loader import OxariDataManager
from lar_calculator.lar_model import OxariUnboundLAR
from postprocessors.missing_year_imputers import CubicSplineMissingYearImputer, DerivativeMissingYearImputer, SimpleMissingYearImputer
from postprocessors.scope_imputers import JumpRateEvaluator, ScopeImputerPostprocessor
import pandas as pd


def impute_missing_years(data: pd.DataFrame):
    print("\n", "Missing Year Imputation")
    my_imputer = SimpleMissingYearImputer().fit(data)
    data = my_imputer.transform(data)
    return data


def impute_scopes(model, data):
    print("Impute scopes with Model")
    scope_imputer = ScopeImputerPostprocessor(estimator=model).run(X=data)
    imputed_data = scope_imputer.data
    return scope_imputer, imputed_data


def compute_lar(imputed_data):
    print("\n", "Predict LARs on Mock data")
    lar_model = OxariUnboundLAR().fit(imputed_data)
    lar_imputed_data = lar_model.transform(imputed_data)
    return lar_model, lar_imputed_data

def compute_jump_rates(imputed_data):
    print("\n", "Compute all the jump rates")
    jump_rate_evaluator = JumpRateEvaluator().fit(imputed_data)
    jump_rates = jump_rate_evaluator.transform(imputed_data)
    return jump_rate_evaluator, jump_rates
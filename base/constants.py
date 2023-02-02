from pathlib import Path

OBJECT_DIR = Path("local/objects")
DATA_DIR = Path("local/data")

IMPORTANT_EVALUATION_COLUMNS = [
    "imputer",
    "preprocessor",
    "feature_selector",
    "scope_estimator",
    "time_to_fit",
    "time_to_optimize",
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

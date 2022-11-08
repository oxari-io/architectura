from base.common import OxariEvaluator
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, r2_score, mean_squared_log_error
from pmdarima.metrics import smape
from sklearn.metrics import mean_absolute_percentage_error as mape


class ClassifierEvaluator(OxariEvaluator):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def evaluate(self, y_test, y_pred, **kwargs):
        """

        Computes 3 flavors of accuracy: Vanilla, AdjacentLenient, AdjacentStrict

        Each accuracy computation is scope and buckets specific

        Appends and saves the results to model/metrics/error_metrics_class.csv

        """

        n_buckets = kwargs.get('n_buckets', len(np.unique(y_test)))
        error_metrics = {
            "vanilla_acc": balanced_accuracy_score(y_test, y_pred),
            "adj_lenient_acc": self.lenient_adjacent_accuracy_score(y_test, y_pred),
            "adj_strict_acc": self.strict_adjacent_accuracy_score(y_test, y_pred, n_buckets),
        }
        return error_metrics

    def lenient_adjacent_accuracy_score(self, y_true, y_pred):
        # if true == 0 and pred == 1 --> CORRECT!
        # if true == 9 and pred == 8 --> CORRECT!
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        return np.sum(np.abs(y_pred - y_true) <= 1) / len(y_pred)

    def strict_adjacent_accuracy_score(self, y_true, y_pred, n_buckets):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
    
        # True = 0 and Pred = 1  ====> FALSE  
        # True = 1 and Pred = 0  ====> FALSE
        # True = 9 and Pred = 8  ====> FALSE
        # True = 8 and Pred = 9  ====> FALSE
        selector_bottom = ((y_true == 0) & (y_pred ==1)) | ((y_true == 1) & (y_pred == 0))
        selector_top = ((y_true == n_buckets-1) & (y_pred == n_buckets-2)) | ((y_true == n_buckets-2) & (y_pred == n_buckets-1))
        selector_inbetween = ~(selector_bottom | selector_top)

        correct_bottom = y_true[selector_bottom] == y_pred[selector_bottom]
        correct_top = y_true[selector_top] == y_pred[selector_top]
        correct_adjacency = np.abs(y_true[selector_inbetween] - y_pred[selector_inbetween]) <= 1
        return (correct_bottom.sum() + correct_top.sum() + correct_adjacency.sum()) / len(y_pred)

    # def strict_adjacent_accuracy_score(self, y_true, y_pred, n_buckets):
    #     def being_strict(y_true, y_pred):
    #             """
    #             # strict with top and bottom bucket
    #             # we want extreme buckets to always be correctly predicted
    #             # if true == 0 and pred == 1 --> WRONG!
    #             # if true == 9 and pred == 8 --> WRONG!

    #             5 buckets 0 to 4
    #             y_true = [0, 2, 4, 3, 1]
    #             y_pred = [1, 1, 3, 4, 0]
    #             FALSE, TRUE, FALSE, FALSE, FALSE

    #             """
    #             if y_true in [n_buckets - 1, 0]:
    #                 return np.abs(y_true - y_pred) == 0
    #             elif y_pred in [n_buckets - 1, 0] and y_true in [n_buckets - 2, 1]:
    #                 return np.abs(y_true - y_pred) == 0
    #             else:
    #                 return np.abs(y_true - y_pred) <= 1

    #     y_true = np.array(y_true)
    #     y_pred = np.array(y_pred)
    #     vfunc = np.vectorize(being_strict)
    #     return np.sum(vfunc(y_true, y_pred)) / len(y_pred)


class RegressorEvaluator(OxariEvaluator):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def evaluate(self, y_true, y_pred, **kwargs):

        # TODO: add docstring here

        # compute metrics of interest
        error_metrics = {
            "sMAPE": smape(y_true, y_pred)/100,
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred, squared=False),
            "RMSLE": mean_squared_log_error(y_true, y_pred, squared=False),
            "MAPE": mape(y_true, y_pred)
        }

        return error_metrics

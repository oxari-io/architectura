from base.common import OxariEvaluator
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, r2_score, mean_squared_log_error
from pmdarima.metrics import smape
from sklearn.metrics import mean_absolute_percentage_error as mape
import base.common as common

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
        return super().evaluate(y_test, y_pred, **error_metrics)

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



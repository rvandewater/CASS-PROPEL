from sklearn.metrics import *


all_classification_metrics_list = ['balanced_accuracy', 'recall', 'precision', 'mcc', 'f1_score', 'roc_auc', 'prc_auc',
                    'avg_precision', 'confusion_matrix']


def compute_classification_metrics(y_true, y_pred):
    # roc_auc_score and average_precision_score require probabilities instead of predictions
    results = {'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
               'recall': recall_score(y_true, y_pred),
               'precision': precision_score(y_true, y_pred, zero_division=0),
               'mcc': matthews_corrcoef(y_true, y_pred),
               'f1_score': f1_score(y_true, y_pred),
               'confusion_matrix': confusion_matrix(y_true, y_pred)
               }
    return results

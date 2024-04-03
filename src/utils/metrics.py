import pickle

import numpy as np
import shap
from matplotlib import pyplot as plt
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


def Shap_summary(X, model_label, out_dir_seed, shap_tuples):
    """
    Generate SHAP summary plots for multiple runs of the same model.
    Args:
        X: Data
        model_label: Model name
        out_dir_seed: Out dir
        shap_tuples: SHAP values
    """
    all_shap_values = np.array([t[0] for t in shap_tuples])
    test_list = np.array([t[1] for t in shap_tuples])
    appended_shap = np.concatenate(all_shap_values, axis=0)
    appended_test = np.concatenate(test_list, axis=0)
    pickle.dump(appended_shap, open(f'{out_dir_seed}/{model_label}_SHAP_values.pkl', 'wb'))
    # for i in range(1, len(all_shap_values)):
    #     test_set = np.concatenate((test_set, [i]), axis=0)
    #     shap_values = np.concatenate((shap_values, np.array(list_shap_values[i])), axis=1)
    # bringing back variable names
    # append shap values to each other
    # mean_shap_values = mean_shap_values.transpose()
    num_features = 40
    test_df = X.iloc[appended_test, :]
    pickle.dump(test_df, open(f'{out_dir_seed}/{model_label}_SHAP_test.pkl', 'wb'))
    # Plot the combined SHAP values
    shap.summary_plot(appended_shap, features=test_df, max_display=num_features, show=False)
    # cohort = [
    #     "Main" if appended_shap[i, "identifier_cohort"].data == 0 else "External"
    #     for i in range(appended_shap.shape[0])
    # ]
    # shap.plots.bar(appended_shap.cohorts(cohort).abs.mean(0))
    plt.tight_layout()
    output_format = "pdf"
    plt.savefig(f'{out_dir_seed}/{model_label}_SHAP_summary.{output_format}'.replace(' ', '_'),
                format=output_format)
    plt.close()
    shap.summary_plot(appended_shap, features=test_df, plot_type='bar', max_display=num_features, show=False)
    plt.tight_layout()
    plt.savefig(f'{out_dir_seed}/{model_label}_SHAP_summary_bars.{output_format}'.replace(' ', '_'),
                format=output_format)
    plt.close()
    # shap.plots.bar(mean_shap_values, max_display=num_features, show=False)
    # plt.tight_layout()
    # plt.savefig(f'{out_dir_seed}/{model_label}_SHAP_bars.{output_format}'.replace(' ', '_'), format=output_format)

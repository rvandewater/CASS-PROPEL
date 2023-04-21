import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, auc
from sklearn.tree import export_graphviz

from src.models import get_feature_importance, positive_class_probability
from src.utils.metrics import compute_classification_metrics
from src.utils.plot import plot_coefficients, plot_roc_pr_curve, plot_confusion_matrix, plot_calibration_curves


def test_classification_model(model, X_train, y_train, X_test, y_test, model_name, select_features, out_dir):
    # Re-fit complete training set
    model.fit(X_train, y_train)

    # Determine optimal classification threshold
    def to_labels(pos_probs, threshold):
        return (pos_probs >= threshold).astype('int')

    y_probas = positive_class_probability(model, X_test)
    thresholds = np.linspace(0, 1, 101)
    scores = [f1_score(y_test, to_labels(y_probas, t)) for t in thresholds]
    ix = np.argmax(scores)
    optimal_threshold = thresholds[ix]
    optimal_f1 = scores[ix]
    with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
        f.write(f'optimal classification threshold: {optimal_threshold} with F1-Score {optimal_f1}\n\n')
    test_metrics = compute_classification_metrics(y_test, to_labels(y_probas, optimal_threshold))

    # ==== ROC & AUPRC ====
    roc_plot, prc_plot = plot_roc_pr_curve(X_test, y_test, y_train.name, model, model_name, out_dir)
    test_metrics['roc_auc'] = roc_plot.roc_auc
    test_metrics['avg_precision'] = prc_plot.average_precision
    aucprs = auc(prc_plot.recall, prc_plot.precision)
    test_metrics['prc_auc'] = aucprs

    plt.close()

    # ===== Confusion Matrix ====
    plot_confusion_matrix(y_train.name, test_metrics['confusion_matrix'], model_name, out_dir, "test")

    # ===== Feature Importances =====
    feature_importances = get_feature_importance(model)
    if select_features:
        feature_names = X_train.columns[model.named_steps['selector'].get_support()]
        with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
            f.write(f'selected features: {feature_names}\n')
        print(f'Selected features: {feature_names}')
    else:
        feature_names = X_train.columns

    if feature_importances is not None:
        feature_importance = pd.DataFrame([feature_importances], columns=feature_names.values)
        feature_importance.to_csv(f'{out_dir}/{y_test.name.replace(" ", "_")}/{model_name}_feature_importance.csv')
        plot_coefficients(out_dir, feature_importances, feature_names, model_name, y_test.name)

    # ===== Decision Tree =====
    if model_name == 'DecisionTreeClassifier':
        dot_data = export_graphviz(model.named_steps['model'], feature_names=feature_names, filled=True, rounded=True)
        graph = graphviz.Source(dot_data)
        graph.render(filename=f'{out_dir}/{y_train.name.replace(" ", "_")}/test/{model_name}_tree', format='png')
        graph.render(filename=f'{out_dir}/{y_train.name.replace(" ", "_")}/test/{model_name}_tree', format='png')

    # ===== Calibration Curves =====
    if not (model_name == 'LinearSVC'):
        plot_calibration_curves(X_test, y_test, y_train.name, model, model_name, out_dir)

    interp_tpr = np.interp(thresholds, roc_plot.fpr, roc_plot.tpr, left=0.0)

    return test_metrics, (interp_tpr, prc_plot.precision, prc_plot.recall)

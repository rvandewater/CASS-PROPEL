import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, auc
import logging as log
from src.models import get_feature_importance, positive_class_probability
from src.utils.metrics import compute_classification_metrics
from src.utils.plot import plot_coefficients, plot_roc_pr_curve, plot_confusion_matrix, plot_calibration_curves, \
    calculate_plot_shap_values
from imblearn.pipeline import Pipeline as imblearn_pipeline
from sklearn.pipeline import Pipeline as sklearn_pipeline


def test_classification_model(model, x_train, y_train, x_test, y_test, model_name, selector, out_dir, calibration=False,
                              shap_eval=True):
    """
    Evaluate a classification model on the test set.
    Args:
        model: Model to evaluate
        x_train: Training data
        y_train: Training labels
        x_test: Test data
        y_test: Test labels
        model_name: Name of the model
        selector: Selector object
        out_dir: Output directory
        calibration: Whether to calibrate the model
        shap_eval: Whether to evaluate SHAP values (slow)

    Returns:
        test_metrics, (interp_tpr, prc_plot.precision, prc_plot.recall), shaps
    """
    if calibration:
        log.info("Calibrating model")
        # Re-fit complete training set
        model.fit(x_train, y_train)
        plot_calibration_curves(x_test, y_test, y_train.name, model, model_name + "_uncalibrated", out_dir)
        # We use prefit as we want to fit on the entire training set
        calibration = CalibratedClassifierCV(model, method='sigmoid', cv="prefit", n_jobs=1)
        model = calibration.fit(x_train, y_train)

    def to_labels(pos_probs, threshold):
        return (pos_probs >= threshold).astype('int')

    # Determine optimal classification threshold
    optimal_threshold, thresholds, y_probas = optimal_classification_threshold(x_test, model, to_labels, y_test, out_dir)

    # Compute standard test metrics
    test_metrics = compute_classification_metrics(y_test, to_labels(y_probas, optimal_threshold))

    # ==== ROC & AUPRC ====
    roc_plot, prc_plot = plot_roc_pr_curve(x_test, y_test, y_train.name, model, model_name, out_dir)
    test_metrics['roc_auc'] = roc_plot.roc_auc
    test_metrics['avg_precision'] = prc_plot.average_precision
    aucprs = auc(prc_plot.recall, prc_plot.precision)
    test_metrics['prc_auc'] = aucprs
    plt.close()

    # ===== Confusion Matrix ====
    plot_confusion_matrix(y_train.name, test_metrics['confusion_matrix'], model_name, out_dir, "test")
    log.debug(model)
    # # estimator = model.estimator.named_steps['model']
    # # selector = model.estimator.named_steps['selector']
    if isinstance(model, CalibratedClassifierCV):
        log.debug(f"Extracting estimator from calibrated wrapper")
        estimator = model.calibrated_classifiers_[0].estimator  # .named_steps['model']
    elif isinstance(model, imblearn_pipeline):
        estimator = model.named_steps['model']
    elif isinstance(model, sklearn_pipeline):
        estimator = model.named_steps['model']
    else:
        estimator = model
    # # if select_features:
    # #     selector = model.calibrated_classifiers_[0].estimator.named_steps['selector']
    # estimator = model

    # ===== Feature Importances =====
    # log.info(f"Estimator: {estimator.estimator}")
    feature_importances = get_feature_importance(estimator)
    if shap_eval:
        shaps = calculate_plot_shap_values(x_test, x_train, y_train, estimator, model_name, out_dir, False)
    else:
        shaps = []
    if selector:
        feature_names = x_train.columns[selector.get_support()]
        with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
            f.write(f'selected features: {feature_names}\n')
        log.info(f'Selected features: {feature_names}')
    else:
        feature_names = x_train.columns

    # Save model inherent feature importances
    if feature_importances is not None:
        feature_importance = pd.DataFrame([feature_importances], columns=feature_names.values)
        feature_importance.to_csv(f'{out_dir}/{model_name}_feature_importance.csv')
        plot_coefficients(out_dir, feature_importances, feature_names, model_name, y_test.name)

    # Decision Tree: plot tree
    if model_name == 'DecisionTreeClassifier':
        plt.figure(figsize=(50, 50))
        tree.plot_tree(estimator, feature_names=feature_names, filled=True, rounded=True, fontsize=10)
        plt.savefig(f'{out_dir}/test/{model_name}_tree.pdf'.replace(' ', '_'), bbox_inches='tight')
        plt.close()

    # ===== Calibration Curves =====
    if not (model_name == 'LinearSVC'):
        plot_calibration_curves(x_test, y_test, y_train.name, model, model_name, out_dir)

    interp_tpr = np.interp(thresholds, roc_plot.fpr, roc_plot.tpr, left=0.0)

    return test_metrics, (interp_tpr, prc_plot.precision, prc_plot.recall), shaps


def optimal_classification_threshold(x_test, model, to_labels, y_test, out_dir):
    y_probabilities = positive_class_probability(model, x_test)
    thresholds = np.linspace(0, 1, 101)
    scores = [f1_score(y_test, to_labels(y_probabilities, t)) for t in thresholds]
    ix = np.argmax(scores)
    optimal_threshold = thresholds[ix]
    optimal_f1 = scores[ix]
    with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
        f.write(f'optimal classification threshold: {optimal_threshold} with F1-Score {optimal_f1}\n\n')
    log.info(f'Optimal classification threshold: {optimal_threshold} with F1-Score {optimal_f1}')
    return optimal_threshold, thresholds, y_probabilities

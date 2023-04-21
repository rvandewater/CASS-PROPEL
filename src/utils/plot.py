import numpy as np
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import auc, precision_recall_curve, average_precision_score, plot_roc_curve, plot_precision_recall_curve, \
    ConfusionMatrixDisplay, PrecisionRecallDisplay
from sweetviz import compare, analyze
import os.path as pth
import warnings
#001C7F, #D62728, #017517, #8C0900, #7600A1, #B8860B, #FF7F0E
dpi = 300
colors = ['#001C7F', '#D62728', '#017517', '#8C0900', '#7600A1', '#B8860B', '#FF7F0E']


model_name_replacements = {
    'DecisionTreeClassifier': 'Decision tree',
    'LogisticRegression': 'Logistic regression',
    'LinearSVC': 'Linear SVM',
    'GradientBoostingClassifier': 'Gradient boosting machine',
    'RandomForestClassifier': 'Random forest',
    'SVC': 'SVM',
    'MLPClassifier': 'Neural network',
}


def boxplot(out_dir, data, metric_name, y_label, ymin=0, ymax=1):
    """Prints boxplot for CV Splits and additionally plots test set value

    Parameters
    ----------
    out_dir : str
        Base output directory
    data : dict
        Metric data in the form {model_name: (list of val split results, test split result)}
    metric_name : str
        The name of the metric
    y_label : str
        The name of the predicted endpoint
    ymin : int, default=0
        min of y axis (usually 0)
    ymax : int, default=0
        max of y axis (usually 1)
    """

    fig = plt.figure()
    fig.suptitle(f'{metric_name} for all models predicting {y_label}')
    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    model_names = list(data.keys())
    # Plot val boxplot
    val_data = list(map(lambda x: x[0], data.values()))
    plt.boxplot(val_data)

    # Plot test single data point
    test_data = list(map(lambda x: x[1], data.values()))
    plt.scatter(range(1, len(model_names) + 1), test_data, marker='o', color='blue')

    # Format axes etc
    ax.set_xticklabels([model_name_replacements.get(model_name, model_name) for model_name in model_names], rotation=45, ha='right')
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{y_label.replace(" ", "_")}/all_models_{metric_name}', dpi=dpi)
    plt.close()


def plot_coefficients(out_dir, coefs, feature_names, model_name, label_name, top_features=None):
    # filter out 0 values
    filter_mask = (abs(coefs) > 1e-3)
    coefs = coefs[filter_mask]
    feature_names = feature_names[filter_mask]
    if top_features:
        top_coef_indixes = np.argsort(abs(coefs))[-top_features:]
        coefs = coefs[top_coef_indixes]
        feature_names = feature_names[top_coef_indixes]
    # sort coefficients
    sort_mask = np.argsort(coefs)
    coefs = coefs[sort_mask]
    feature_names = feature_names[sort_mask]
    # plot
    plt.figure(figsize=(30, 10))
    plt.suptitle(f'Feature importance - {model_name} predicting {label_name}')
    colors = ['red' if c < 0 else 'blue' for c in coefs]
    plt.bar(np.arange(len(coefs)), coefs, color=colors)
    plt.xticks(np.arange(len(coefs)), feature_names, rotation=60, ha='right')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{label_name.replace(" ", "_")}/test/{model_name}_feature_importance', dpi=dpi)
    plt.close()


def plot_summary_roc(all_model_metrics, out_dir, label_col, dataset_partition='val', title=None, legend=False, value_in_legend=True):
    """
    Plot ROC curves for all models in one figure.
    Parameters
    ----------
    all_model_metrics:
    out_dir: str
    label_col:
    dataset_partition: str
        'val' or 'test', depending on which curves to plot (for CV or test performance)
    title: Include title in plot.
    legend: Include legend in plot.
    """
    if dataset_partition == 'val':
        ds_index = 0
    else:
        ds_index = 1

    plt.rcParams['font.family'] = "Arial"
    plt.style.use('seaborn-notebook')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_aspect('equal')
    if title:
        fig.suptitle(f"{title} {label_col}")
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', label='Chance', alpha=.8)

    x_linspace = np.linspace(0, 1, 101)
    roc_prc_data = {model_name: entry[2] for model_name, entry in all_model_metrics.items()}
    i = 0
    for model_name, val_and_test_curves_data in roc_prc_data.items():
        if model_name == 'DummyClassifier':
            continue
        if value_in_legend:
            label = '{} (AUROC = {:.2f})'.format(model_name_replacements.get(model_name, model_name),
                                                 np.mean(all_model_metrics[model_name][ds_index]['roc_auc']))
        else:
            label = model_name_replacements.get(model_name, model_name)
        ax.plot(x_linspace, val_and_test_curves_data[ds_index][0],
                label=label,
                lw=2, alpha=.8, color=colors[i])
        i += 1
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    plt.savefig(f'{out_dir}/{label_col}/all_models_{dataset_partition}_roc_curves'.replace(' ', '_'),
                bbox_inches='tight', dpi=dpi)
    plt.close()


def plot_summary_prc(all_model_metrics, out_dir, label_col, y, dataset_partition='val', title=None, legend=False, value_in_legend=True):
    """
    Plot PRC curves for all models in one figure.
    Parameters
    ----------
    all_model_metrics:
    out_dir: str
    label_col:
    y:
    dataset_partition: str
        'val' or 'test', depending on which curves to plot (for CV or test performance)
    title: Include title in plot.
    legend: Include legend in plot.
    """
    if dataset_partition == 'val':
        ds_index = 0
    else:
        ds_index = 1

    plt.rcParams['font.family'] = "Arial"
    plt.style.use('seaborn-notebook')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_aspect('equal')
    if title:
        fig.suptitle(f"{title} {label_col}")
    ax.plot([0, 1], [(len(y[y == 1]) / len(y)), (len(y[y == 1]) / len(y))], linestyle='--', lw=2, color='grey',
            label='Chance', alpha=.8)
    roc_prc_data = {model_name: entry[2] for model_name, entry in all_model_metrics.items()}
    i = 0
    for model_name, val_and_test_curves_data in roc_prc_data.items():
        if model_name == 'DummyClassifier':
            continue
        if value_in_legend:
            label = '{} (AUPRC = {:.2f})'.format(model_name_replacements.get(model_name, model_name),
                                                 np.mean(all_model_metrics[model_name][ds_index]['prc_auc']))
        else:
            label = model_name_replacements.get(model_name, model_name)
        ax.plot(val_and_test_curves_data[ds_index][2], val_and_test_curves_data[ds_index][1],
                label=label,
                lw=2, alpha=.8, color=colors[i])
        i += 1
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.savefig(f'{out_dir}/{label_col}/all_models_{dataset_partition}_prc_curves'.replace(' ', '_'),
                bbox_inches='tight', dpi=dpi)
    plt.close()


def plot_summary_roc_pr(all_model_metrics, out_dir, label_col, y):
    """
    Plot both ROC and PR curves for all models in one figure.
    Parameters
    ----------
    all_model_metrics:
    out_dir: str
    label_col:
    y:
    """
    # Plot mean val & test ROC and PR Curve for all models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    fig.suptitle(f'Predicting {label_col} ROC & PR Curves')
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey',
             label='Chance', alpha=.8)
    ax2.plot([0, 1], [(len(y[y == 1]) / len(y)), (len(y[y == 1]) / len(y))], linestyle='--', lw=2, color='grey',
             label='Chance', alpha=.8)
    x_linspace = np.linspace(0, 1, 101)
    roc_prc_data = {model_name: entry[2] for model_name, entry in all_model_metrics.items()}
    for model_name, (val_curves_data, test_curves_data) in roc_prc_data.items():
        if model_name == 'DummyClassifier':
            continue
        ax1.plot(x_linspace, val_curves_data[0],
                 label='{} (AUROC = {:.2f})'.format(model_name_replacements.get(model_name, model_name),
                                                    np.mean(all_model_metrics[model_name][0]['roc_auc'])),
                 lw=1, alpha=.8)
        ax2.plot(val_curves_data[2], val_curves_data[1],
                 label='{} (AUPRC = {:.2f})'.format(model_name_replacements.get(model_name, model_name),
                                                    np.mean(all_model_metrics[model_name][0]['prc_auc'])),
                 lw=1, alpha=.8)
    ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
            title='ROC')
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax2.set(xlim=[-0.05, 1.05], ylim=[0.0, 1.05],
            title='Precision-Recall')
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.savefig(f'{out_dir}/{label_col}/all_models_roc_prc_curves'.replace(' ', '_'),
                bbox_inches='tight', dpi=dpi)
    plt.close()


def plot_val_mean_roc(ax, tprs, aucs, x_linspace):
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(x_linspace, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(x_linspace, mean_tpr, color='b',
            label=r'Mean ROC (AUROC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(x_linspace, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title='ROC')
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))

    return mean_tpr


def plot_val_mean_prec_rec(ax, ys_real, ys_proba, pos_class_fraction):
    ax.plot([0, 1], [pos_class_fraction, pos_class_fraction], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    overall_precision, overall_recall, _ = precision_recall_curve(ys_real, ys_proba)
    overall_prc_auc = auc(overall_recall, overall_precision)
    overall_prc_ap = average_precision_score(ys_real, ys_proba)
    ax.plot(overall_recall, overall_precision, color='b',
            label=r'Overall Prec_Rec (AP = %0.2f, AUC = %.2f)' % (overall_prc_ap, overall_prc_auc),
            lw=2, alpha=.8)

    ax.set(xlim=[-0.05, 1.05], ylim=[0.0, 1.05],
           title='Precision-Recall')

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))

    return overall_precision, overall_recall, overall_prc_auc


def data_exploration_comparison(train_data, test_data, endpoint, file_path):
    exploration_report = compare([train_data, "Training data"], [test_data, "External Test Data"], endpoint)
    exploration_report.show_html(filepath=pth.join(file_path, f"dataset_comparison_{endpoint}.html"),
                                 open_browser=False,
                                 layout='widescreen',
                                 scale=None)


def data_exploration(data, endpoint, file_path):
    exploration_report = analyze([data, "Source Data"], endpoint)
    exploration_report.show_html(filepath=pth.join(file_path, f"dataset_exploration_{endpoint}.html"),
                                 open_browser=False,
                                 layout='widescreen',
                                 scale=None)


def plot_roc_pr_curve(X_test, y_test, endpoint, model, model_name, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax1.set(xlim=[-0.05, 1.05], ylim=[0.0, 1.05])
    ax2.set(xlim=[-0.05, 1.05], ylim=[0.0, 1.05])
    fig.suptitle(f'{model_name} predicting {endpoint}')
    # ROC
    roc_plot = plot_roc_curve(model, X_test, y_test,
                              name='ROC curve', lw=1, ax=ax1)
    ax2.set(xlim=[-0.05, 1.05], ylim=[0.0, 1.05])
    prc_plot = PrecisionRecallDisplay.from_estimator(model, X_test, y_test,
                                                     name='PR curve', lw=1, ax=ax2)
    plt.savefig(f'{out_dir}/{endpoint}/test/{model_name}_roc_prc_curves'.replace(' ', '_'), bbox_inches='tight', dpi=dpi)

    return roc_plot, prc_plot


def plot_confusion_matrix(label, confusion_matrix, model_name, out_dir, phase):
    cm_fig, ax = plt.subplots()
    cm_fig.suptitle(f'{model_name} predicting {label}')
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                  display_labels=[0, 1])
    disp.plot(include_values=True, cmap='Blues', ax=ax,
              xticks_rotation='horizontal', values_format='d')
    plt.savefig(f'{out_dir}/{label}/{phase}/{model_name}_cm'.replace(' ', '_'))
    plt.close()


def plot_shap_values(X_test, X_train, y_train, cv, model, model_name, out_dir, select_features):
    def get_shap_values(test_data, train_data):
        if select_features:
            train_data = model.named_steps['selector'].transform(train_data)
            test_data = model.named_steps['selector'].transform(test_data)

        # Get SHAP values function
        if model_name == 'LogisticRegression':
            explainer = shap.LinearExplainer(model.named_steps['model'], train_data)
        elif model_name in ['DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']:
            explainer = shap.TreeExplainer(model.named_steps['model'], train_data)
        else:
            if hasattr(model, "predict_proba"):
                f = lambda x: model.predict_proba(x)[:, 1]
                explainer = shap.KernelExplainer(f, train_data)
            else:
                explainer = shap.KernelExplainer(model.decision_function, train_data)
        return explainer.shap_values(test_data)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Test SHAP values
        model.fit(X_train, y_train)
        test_shap_values = get_shap_values(X_test, X_train)
        shap.summary_plot(test_shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{y_train.name}/test/{model_name}_SHAP.png'.replace(' ', '_'), dpi=dpi)
        plt.close()

        shap.summary_plot(test_shap_values, X_test, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{y_train.name}/test/{model_name}_SHAP_bars.png'.replace(' ', '_'), dpi=dpi)
        plt.close()

        # CV SHAP values
        shap_values_list = []
        test_ixs = []
        for train_ix, test_ix in cv.split(X_train, y_train):
            test_ixs.append(test_ix)

            X_tr = X_train.iloc[train_ix]
            y_tr = y_train.iloc[train_ix]
            X_te = X_train.iloc[test_ix]
            model.fit(X_tr, y_tr)
            shap_values = get_shap_values(X_te, X_tr)
            for shap_value in shap_values:
                shap_values_list.append(shap_value)

        new_index = [ix for ix_test_fold in test_ixs for ix in ix_test_fold]
        shap.summary_plot(np.array(shap_values_list), X_train.reindex(new_index), show=False)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{y_train.name}/val/{model_name}_SHAP.png'.replace(' ', '_'), dpi=dpi)
        plt.close()

        shap.summary_plot(np.array(shap_values_list), X_train.reindex(new_index), plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{y_train.name}/val/{model_name}_SHAP_bars.png'.replace(' ', '_'), dpi=dpi)
        plt.close()


def plot_calibration_curves(X_test, y_test, endpoint, model, model_name, out_dir):
    sns.set(style='darkgrid', context='talk', palette='rainbow')
    fig, ax = plt.subplots()
    fig.suptitle(f'{model_name} predicting {endpoint}')
    viz = CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=10, name=model_name, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{endpoint}/test/{model_name}_calibration'.replace(' ', '_'), dpi=dpi)
    plt.close()
    sns.reset_orig()

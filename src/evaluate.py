import os
import logging as log
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.metrics import ConfusionMatrixDisplay, auc, PrecisionRecallDisplay, RocCurveDisplay, make_scorer, \
    average_precision_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold, RandomizedSearchCV
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.preprocess import model_feature_selection
from src.test import test_classification_model
from src.utils.metrics import all_classification_metrics_list, compute_classification_metrics
from src.utils.plot import plot_val_mean_prec_rec, plot_val_mean_roc, plot_confusion_matrix, plot_shap_values


def evaluate_single_cv_split(model, param_grid,
                             X_train, y_train, X_test, y_test,
                             cv_splits=5, cv_scoring='log_loss', select_features=False, shap_value_eval=False,
                             cm_agg_type='sum', out_dir='results/default', sample_balancing=None, seed=42,
                             search_method='random'):
    os.makedirs(f'{out_dir}/val/', exist_ok=True)
    os.makedirs(f'{out_dir}/test/', exist_ok=True)
    model_name = str(model.__class__.__name__)

    # ================= SETTING UP K-FOLD OR LOO CV =================
    if cv_splits > 0:
        log.info(f'Evaluating model {model_name} with {cv_splits}-fold CV')
        log.info(
            f'Total split into Train/Val/Test: {round(100 * (cv_splits - 1) / cv_splits * len(y_train) / (len(y_train) + len(y_test)))}/' +
            f'{round(100 / cv_splits * len(y_train) / (len(y_train) + len(y_test)))}/{round(100 * len(y_test) / (len(y_train) + len(y_test)))}' +
            f' - Absolute Samples: {len(y_train) - round(len(y_train) / cv_splits)}/{round(len(y_train) / cv_splits)}/{len(y_test)}')

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    else:
        log.info(f'Evaluating model {model_name} with LOO-CV')
        log.info(
            f'Total split into Train+Val/Test: {round(100 * len(y_train) / (len(y_train) + len(y_test)))}/' +
            f'{round(100 * len(y_test) / (len(y_train) + len(y_test)))}' +
            f' - Absolute Samples: {len(y_train) - 1}/1/{len(y_test)}')

        cv = LeaveOneOut()

    # Define list with steps for the pipeline
    pipeline_steps = []

    # ================= ADD BALANCING TO PIPELINE IF SELECTED =================
    if sample_balancing in ['random_oversampling', 'SMOTE', 'ADASYN']:
        log.info(f'Performing random oversampling via {sample_balancing} algorithm.')
        log.info(f'n samples before: {len(y_train[y_train == 0])} vs. {len(y_train[y_train == 1])}')
        if sample_balancing == 'random_oversampling':
            over_sampler = RandomOverSampler(random_state=seed)  # todo possibly reduce ratio to sth like 0.5
        elif sample_balancing == 'SMOTE':
            over_sampler = SMOTE(n_jobs=-1, random_state=seed)
        elif sample_balancing == 'ADASYN':
            over_sampler = ADASYN(n_jobs=-1, random_state=seed)
        else:
            raise ValueError(f'Unknown balancing option {sample_balancing}')

        # log.info(f'n samples after:  {len(endpoint[endpoint == 0])} vs. {len(endpoint[endpoint == 1])}')
        pipeline_steps.append(('over_sampling', over_sampler))

    # ================= SELECT OPTIMAL MODEL AND FEATURE SET THROUGH CV =================

    # prepare param_grid
    param_grid = {'model__' + key: value for (key, value) in param_grid.items()}
    if select_features:
        # Feature selection for each endpoint only on training data
        param_grid['selector'] = [model_feature_selection(X_test, X_train, y_train, min_feature_fraction=0.5, cores=1,
                                                          scoring=make_scorer(average_precision_score))]
        # Use XGBClassifier for feature selection
        param_grid['selector'] = [SelectFromModel(XGBClassifier())]

        pipeline_steps.extend([('selector', 'passthrough'), ('model', model)])
    else:
        pipeline_steps.append(('model', model))

    pipeline = Pipeline(pipeline_steps)
    log.info(f"Using pipeline {pipeline}")

    # Define metrics used
    all_metrics_list = all_classification_metrics_list

    if search_method == 'grid':
        grid_model = GridSearchCV(pipeline, param_grid=param_grid, scoring=cv_scoring, verbose=False, cv=cv,
                                  n_jobs=-1,
                                  error_score=0)
    elif search_method == 'random':
        grid_model = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring=cv_scoring, verbose=False, cv=cv,
                                        n_jobs=-1,
                                        error_score=0, n_iter=10)
    else:
        raise ValueError(f'Unknown method {search_method}')
    grid_model.fit(X_train, y_train)
    log.info("Fitted model")

    try:
        pass
    except ValueError as ve:
        log.info(ve)
        with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
            f.write('\n' + model_name)
            f.write(f'GridSearch Failed due to incompatible options in best selected model.\n')
        log.warning("GridSearch Failed due to incompatible options in best selected model.")
        empty_cm = np.zeros((2, 2))
        return {metric: ([0.0] if metric != 'confusion_matrix' else [empty_cm] * cv_splits) for metric in all_metrics_list}, \
            {metric: (0.0 if metric != 'confusion_matrix' else empty_cm) for metric in all_metrics_list}, \
            (([0] * 101, [0] * 101, [0] * 101), ([0] * 101, [0] * 101, [0] * 101))
    with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
        f.write('\n' + model_name)
        f.write(f'\nBest Params: {grid_model.best_params_}\n')
    log.info(f'Best Params: {grid_model.best_params_} - {cv_scoring}: {grid_model.best_score_}')

    best_model = grid_model.best_estimator_

    cv_metrics, mean_tpr, overall_precision, overall_recall = independent_validation(X_test, X_train, all_metrics_list,
                                                                                     best_model, cm_agg_type, cv, cv_splits,
                                                                                     model_name, out_dir, select_features,
                                                                                     shap_value_eval, y_test, y_train)
    # =================== Final Model Testing ===============
    test_metrics, test_curves = test_classification_model(best_model, X_train, y_train, X_test, y_test,
                                                          model_name, select_features, out_dir)
    return cv_metrics, test_metrics, ((mean_tpr, overall_precision, overall_recall), test_curves)


def evaluate_single_model(model, param_grid,
                          X_train, y_train, X_test, y_test,
                          cv_splits=5, cv_scoring='average_precision', select_features=False, shap_value_eval=False,
                          cm_agg_type='sum', out_dir='results/default', sample_balancing=None, seed=42,
                          search_method='random'):
    os.makedirs(f'{out_dir}/val/', exist_ok=True)
    os.makedirs(f'{out_dir}/test/', exist_ok=True)
    model_name = str(model.__class__.__name__)

    # ================= SETTING UP K-FOLD OR LOO CV =================
    if cv_splits > 0:
        log.info(f'Evaluating model {model_name} with {cv_splits}-fold CV')
        log.info(
            f'Total split into Train/Val/Test: {round(100 * (cv_splits - 1) / cv_splits * len(y_train) / (len(y_train) + len(y_test)))}/' +
            f'{round(100 / cv_splits * len(y_train) / (len(y_train) + len(y_test)))}/{round(100 * len(y_test) / (len(y_train) + len(y_test)))}' +
            f' - Absolute Samples: {len(y_train) - round(len(y_train) / cv_splits)}/{round(len(y_train) / cv_splits)}/{len(y_test)}')

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    else:
        log.info(f'Evaluating model {model_name} with LOO-CV')
        log.info(
            f'Total split into Train+Val/Test: {round(100 * len(y_train) / (len(y_train) + len(y_test)))}/' +
            f'{round(100 * len(y_test) / (len(y_train) + len(y_test)))}' +
            f' - Absolute Samples: {len(y_train) - 1}/1/{len(y_test)}')

        cv = LeaveOneOut()

    # Define list with steps for the pipeline
    pipeline_steps = []
    # ================= ADD BALANCING TO PIPELINE IF SELECTED =================
    if sample_balancing in ['random_oversampling', 'SMOTE', 'ADASYN']:
        log.info(f'Performing random oversampling via {sample_balancing} algorithm.')
        log.info(f'n samples before: {len(y_train[y_train == 0])} vs. {len(y_train[y_train == 1])}')
        if sample_balancing == 'random_oversampling':
            over_sampler = RandomOverSampler(random_state=seed)  # todo possibly reduce ratio to sth like 0.5
        elif sample_balancing == 'SMOTE':
            over_sampler = SMOTE(n_jobs=-1, random_state=seed)
        elif sample_balancing == 'ADASYN':
            over_sampler = ADASYN(n_jobs=-1, random_state=seed)
        else:
            raise ValueError(f'Unknown balancing option {sample_balancing}')

        # log.info(f'n samples after:  {len(endpoint[endpoint == 0])} vs. {len(endpoint[endpoint == 1])}')
        pipeline_steps.append(('over_sampling', over_sampler))

    # ================= SELECT OPTIMAL MODEL AND FEATURE SET THROUGH CV =================
    pretune = True
    # Define metrics used
    all_metrics_list = all_classification_metrics_list

    if not pretune:
        # prepare param_grid
        param_grid = {'model__' + key: value for (key, value) in param_grid.items()}
        select_features = False
        if select_features:
            # Feature selection for each endpoint only on training data
            param_grid['selector'] = [model_feature_selection(X_test, X_train, y_train, min_feature_fraction=0.5, cores=1,
                                                              scoring=make_scorer(average_precision_score))]
            param_grid['selector'] = [SelectFromModel(XGBClassifier())]
            # param_grid['selector'] = [SelectKBest(k='all'), SelectKBest(k=25),
            #                           SelectFromModel(LinearSVC(C=1, penalty="l1", dual=False, max_iter=5000))]

            pipeline_steps.extend([('selector', 'passthrough'), ('model', model)])
        else:
            pipeline_steps.append(('model', model))

        pipeline = Pipeline(pipeline_steps)
        log.info(f"Using pipeline {pipeline}")

        if search_method == 'grid':
            grid_model = GridSearchCV(pipeline, param_grid=param_grid, scoring=cv_scoring, verbose=False, cv=cv,
                                      n_jobs=-1,
                                      error_score=0)
        elif search_method == 'random':
            grid_model = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring=cv_scoring, verbose=False, cv=cv,
                                            n_jobs=1,
                                            error_score=0, n_iter=10)
        else:
            raise ValueError(f'Unknown method {search_method}')
        grid_model.fit(X_train, y_train)
        best_model = grid_model.best_estimator_
    else:
        grid_model = model
        best_model = grid_model.fit(X_train, y_train)
    log.info("Fitted model")

    try:
        pass
    except ValueError as ve:
        log.info(ve)
        with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
            f.write('\n' + model_name)
            f.write(f'GridSearch Failed due to incompatible options in best selected model.\n')
        log.warning("GridSearch Failed due to incompatible options in best selected model.")
        empty_cm = np.zeros((2, 2))
        return {metric: ([0.0] if metric != 'confusion_matrix' else [empty_cm] * cv_splits) for metric in all_metrics_list}, \
            {metric: (0.0 if metric != 'confusion_matrix' else empty_cm) for metric in all_metrics_list}, \
            (([0] * 101, [0] * 101, [0] * 101), ([0] * 101, [0] * 101, [0] * 101))
    # with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
    #     f.write('\n' + model_name)
    #     f.write(f'\nBest Params: {grid_model.best_params_}\n')
    # log.info(f'Best Params: {grid_model.best_params_} - {cv_scoring}: {grid_model.best_score_}')

    cv_metrics, mean_tpr, overall_precision, overall_recall = independent_validation(X_test, X_train, all_metrics_list,
                                                                                     best_model, cm_agg_type, cv, cv_splits,
                                                                                     model_name, out_dir, select_features,
                                                                                     shap_value_eval, y_test, y_train)
    # =================== Final Model Testing ===============
    test_metrics, test_curves, shaps = test_classification_model(best_model, X_train, y_train, X_test, y_test,
                                                                 model_name, select_features, out_dir)
    return cv_metrics, test_metrics, ((mean_tpr, overall_precision, overall_recall), test_curves), best_model, (
    shaps, list(X_test.index))


def independent_validation(X_test, X_train, all_metrics_list, best_model, cm_agg_type, cv, cv_splits, model_name, out_dir,
                           select_features, shap_value_eval, y_test, y_train):
    """
        Performs cross-validation with selected model on the test set and plots the ROC and Precision-Recall curves.

    Args:
        X_test:
        X_train:
        all_metrics_list:
        best_model:
        cm_agg_type:
        cv:
        cv_splits:
        model_name:
        out_dir:
        select_features:
        shap_value_eval:
        y_test:
        y_train:

    Returns: cv_metrics, mean_tpr, overall_precision, overall_recall

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    fig.suptitle(f'{model_name} predicting {y_test.name}')
    # ROC values
    tprs = []
    x_linspace = np.linspace(0, 1, 101)
    cv_metrics = {metric: [] for metric in all_metrics_list}
    ys_real = []
    ys_proba = []
    # Perform k-fold cross validation
    for i, (train, val) in enumerate(cv.split(X_train, y_train)):
        cv_X_train = X_train.iloc[train]
        cv_y_train = y_train.iloc[train]
        cv_X_val = X_train.iloc[val]
        cv_y_val = y_train.iloc[val]

        # Save performance values
        best_model.fit(cv_X_train, cv_y_train)

        y_pred = best_model.predict(cv_X_val)

        model_metrics = compute_classification_metrics(cv_y_val, y_pred)

        # Save predict probabilities for total PRC
        ys_real.append(cv_y_val)
        if model_name in ['LinearSVC', 'SVC']:
            ys_proba.append(best_model.decision_function(cv_X_val))
        else:
            ys_proba.append(best_model.predict_proba(cv_X_val)[:, 1])

        # save ROC values
        viz = RocCurveDisplay.from_estimator(best_model, cv_X_val, cv_y_val,
                                             name='ROC fold {}'.format(i),
                                             alpha=0.3, lw=1, ax=ax1)
        interp_tpr = np.interp(x_linspace, viz.fpr, viz.tpr, left=0.0)
        tprs.append(interp_tpr)

        model_metrics['roc_auc'] = viz.roc_auc

        # save Precision_Recall values
        viz2 = PrecisionRecallDisplay.from_estimator(best_model, cv_X_val, cv_y_val,
                                                     name='PRC fold {}'.format(i),
                                                     alpha=0.3, lw=1, ax=ax2)

        model_metrics['avg_precision'] = viz2.average_precision
        aucprs = auc(viz2.recall, viz2.precision)
        model_metrics['prc_auc'] = aucprs

        cv_metrics = {key: value + [model_metrics[key]] for (key, value) in cv_metrics.items()}
    # Plot ROC
    mean_tpr = plot_val_mean_roc(ax1, tprs, cv_metrics['roc_auc'], x_linspace)
    # Plot Precision_Recall
    overall_precision, overall_recall, overall_prc_auc = plot_val_mean_prec_rec(ax2, np.concatenate(ys_real),
                                                                                np.concatenate(ys_proba),
                                                                                (sum(y_train == 1) / len(y_train)))
    # Plot SHAP values
    if shap_value_eval:
        plot_shap_values(X_test, X_train, y_train, cv, best_model, model_name, out_dir, select_features)
    # all_model_metrics['prc_auc'] = overall_prc_auc
    plt.savefig(f'{out_dir}/val/{model_name}_roc_prc_curves'.replace(' ', '_'), bbox_inches='tight')
    plt.close()
    # Plot confusion matrix
    val_cm = sum(cv_metrics['confusion_matrix'])
    if cm_agg_type == 'mean':
        val_cm = val_cm / cv_splits
    plot_confusion_matrix(y_train.name, val_cm, model_name, out_dir, "val")
    cm_fig, ax = plt.subplots()

    cm_fig.suptitle(f'{model_name} predicting {y_test.name}')
    disp = ConfusionMatrixDisplay(confusion_matrix=val_cm,
                                  display_labels=[0, 1])
    disp.plot(include_values=True, cmap='Blues', ax=ax,
              xticks_rotation='horizontal', values_format='d')
    plt.savefig(f'{out_dir}/val/{model_name}_cm'.replace(' ', '_'))
    plt.close()
    return cv_metrics, mean_tpr, overall_precision, overall_recall

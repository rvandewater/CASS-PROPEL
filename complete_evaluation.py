import argparse
import os
import json
import time
import random
from array import array
from datetime import datetime
import wandb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data import get_data_from_name
from src.preprocess import get_preprocessed_data
from src.models import get_classification_model_grid
from src.evaluate import evaluate_single_model
from src.utils.metrics import all_classification_metrics_list
from src.utils.plot import boxplot, plot_summary_roc_pr, plot_summary_roc, plot_summary_prc
from sklearn.metrics import average_precision_score, roc_auc_score, make_scorer
import logging as log
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier

# import feature_engine
bars = '====================='


def evaluation(seed, out_dir, dataset, feature_set, external_test_data, imputer, normaliser, feature_selectors, drop_features,
               select_features, cv_splits, shap_eval, test_fraction, balancing_option, drop_missing_value, missing_threshold,
               correlation_threshold, data_exploration, cores):
    # Setup output directory
    seed = seed
    np.random.seed(seed)
    random.seed(seed)
    if out_dir is None:
        feature_set_string = '' if feature_set is None else f'_{"_".join(feature_set)}'
        out_dir = f'results_{dataset}{feature_set_string}_{str(datetime.now().strftime("%Y-%m-%d_T_%H-%M-%S"))}_seed_{seed}'
    else:
        feature_set_string = '' if feature_set is None else f'_{"_".join(feature_set)}'
        out_dir = f'{out_dir}/results_{dataset}{feature_set_string}_{str(datetime.now().strftime("%Y-%m-%d_T_%H-%M-%S"))}_seed_{seed}'
    log.info(f"Logging results to: {out_dir}")
    os.makedirs(f'{out_dir}', exist_ok=True)
    os.makedirs(f'{out_dir}/data_frames', exist_ok=True)

    # Get DataInformation object for the specified task
    data = get_data_from_name(dataset)

    # Parse data
    data.parse(drop_columns=drop_features, feature_set=feature_set, drop_missing_value=drop_missing_value,
               out_dir=out_dir, exploration=data_exploration, external_validation=external_test_data)

    # Preprocess data
    X, Y = get_preprocessed_data(data,
                                 # fs_operations=feature_selectors,
                                 # missing_threshold=missing_threshold,
                                 # correlation_threshold=correlation_threshold,
                                 imputer=imputer,
                                 normaliser=normaliser,
                                 verbose=True,
                                 validation=False)
    log.info(f"{bars} Preprocessing complete {bars}")
    # Preprocess external validation data
    if external_test_data:
        X_val, Y_val = get_preprocessed_data(data,
                                             missing_threshold=missing_threshold,
                                             correlation_threshold=correlation_threshold,
                                             verbose=True, validation=True)
        log.info(X_val.columns.difference(X.columns))
        # Get rid of extra columns introduced by values in validation dataset
        log.info(f'Dropping columns in val data since they are missing in train data: {X_val.columns.difference(X.columns)}')
        X_val = X_val.drop(set(X_val.columns.difference(X.columns)), axis=1)
        assert len(X.columns.difference(
            X_val.columns)) == 0, f'Error: Train data includes columns {X.columns.difference(X_val.columns)} that are missing in val data'

    all_metrics_list = all_classification_metrics_list

    all_test_metric_dfs = {metric: pd.DataFrame(dtype=np.float64) for metric in all_metrics_list if
                           metric != 'confusion_matrix'}

    with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
        f.write(f'\n========== New Trial at {time.strftime("%d.%m.%Y %H:%M:%S")} ==========\n')
        # f.write(str(vars(args)))
        f.write('\n')

    for k, label_col in enumerate(Y.columns):
        log.info(f'Predicting {label_col}')
        with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
            f.write(f'=====\n{label_col}\n=====')

        # Set endpoint for iteration
        y = Y[label_col]

        log.info(Y.info())

        # If we do not have an external validation dataset, we split the original dataset
        if not external_test_data:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction,
                                                                random_state=seed, shuffle=True, stratify=y)
        else:
            # Set endpoint for iteration
            y_val = Y_val[label_col]

            # Set train and test
            X_train = X
            y_train = y
            X_test = X_val
            y_test = y_val

        all_model_metrics = {}

        # Feature selection for each endpoint only on training data
        X_test, X_train = model_feature_selection(X_test, X_train, cores, y_train)

        # model grid
        model_grid = get_classification_model_grid('balanced' if balancing_option == 'class_weight' else None, seed=seed)
        for j, (model, param_grid) in enumerate(model_grid):
            val_metrics, test_metrics, curves = evaluate_single_model(model, param_grid,
                                                                      X_train, y_train, X_test, y_test,
                                                                      cv_splits=cv_splits,
                                                                      select_features=select_features,
                                                                      shap_value_eval=shap_eval,
                                                                      out_dir=out_dir,
                                                                      sample_balancing=balancing_option,
                                                                      seed=seed)
            all_model_metrics[str(model.__class__.__name__)] = (val_metrics, test_metrics, curves)

        # ===== Save aggregate plots across models =====
        # Generate Boxplots for Metrics
        json_metric_data = {}
        for metric_name in all_model_metrics[str(model.__class__.__name__)][0].keys():
            if metric_name == 'confusion_matrix':
                json_metric_data[metric_name] = {
                    model_name: ([cv_cm.tolist() for cv_cm in val_metrics[metric_name]], test_metrics[metric_name].tolist())
                    for model_name, (val_metrics, test_metrics, _) in all_model_metrics.items()}
                continue
            metric_data = {model_name: (val_metrics[metric_name], test_metrics[metric_name])
                           for model_name, (val_metrics, test_metrics, _) in all_model_metrics.items()}
            json_metric_data[metric_name] = metric_data
            boxplot(out_dir, metric_data, metric_name, label_col, ymin=(-1 if metric_name == 'mcc' else 0))
        json.dump(json_metric_data, open(f'{out_dir}/{label_col}/all_model_metrics.json', 'w'), indent=4)

        # Plot roc pr for all models
        plot_summary_roc(all_model_metrics, out_dir, label_col, dataset_partition='val', legend=True,
                         value_in_legend=False)
        plot_summary_roc(all_model_metrics, out_dir, label_col, dataset_partition='test', legend=True,
                         value_in_legend=False)
        plot_summary_prc(all_model_metrics, out_dir, label_col, y, dataset_partition='val', legend=True,
                         value_in_legend=False)
        plot_summary_prc(all_model_metrics, out_dir, label_col, y, dataset_partition='test', legend=True,
                         value_in_legend=False)
        plot_summary_roc_pr(all_model_metrics, out_dir, label_col, y)

        # save results in DF
        for model_name, test_data in {model_name: entry[1] for model_name, entry in all_model_metrics.items()}.items():
            for metric, value in test_data.items():
                if metric == 'confusion_matrix':
                    continue
                all_test_metric_dfs[metric].loc[model_name, label_col.replace(' ', '_')] = value

        for metric, df in all_test_metric_dfs.items():
            df.to_csv(f'{out_dir}/data_frames/{metric}.csv')


def model_feature_selection(X_test, X_train, cores, y_train, min_feature_fraction=0.5, scoring=make_scorer(average_precision_score)):
    xgb_model = XGBClassifier()
    n_features = int(len(X_train.columns) * min_feature_fraction)

    # Cross validated feature selection
    select = RFECV(estimator=xgb_model, min_features_to_select=n_features, verbose=0, cv=5, scoring=scoring, n_jobs=cores)
    select = select.fit(X_train, y_train)

    # Get the selection mask
    mask = select.get_support()
    X_train = X_train.loc[:, mask]  # select.transform(X_train)
    X_test = X_test.loc[:, mask]
    log.info(f"Columns after feature selection: {X_train.columns}")
    return X_test, X_train

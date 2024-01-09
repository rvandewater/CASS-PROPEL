import os
import json
import time
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import get_data_from_name
from src.preprocess import common_preprocessing
from src.models import get_classification_model_grid
from src.evaluate import evaluate_single_model
from src.utils.metrics import all_classification_metrics_list
from src.utils.plot import boxplot, plot_summary_roc_pr, plot_summary_roc, plot_summary_prc
import logging as log

# import feature_engine
bars = '====================='


def evaluation(seeds, out_dir, dataset, feature_set, external_test_data, imputer, normaliser, drop_features,
               select_features, cv_splits, shap_eval, test_fraction, balancing_option, drop_missing_value, data_exploration,
               missing_threshold, correlation_threshold, cores):
    """
    Main function for evaluation of classical ML models on post-operative complications dataset.

    Args:
        seed:
        out_dir:
        dataset:
        feature_set:
        external_test_data:
        imputer:
        normaliser:
        drop_features:
        select_features:
        cv_splits:
        shap_eval:
        test_fraction:
        balancing_option:
        drop_missing_value:
        data_exploration:
        missing_threshold:
        correlation_threshold:
        cores:
    """
    cores = 1
    out_dir = setup_output_directory(dataset, feature_set, out_dir)

    # Get DataInformation object for the specified task
    data = get_data_from_name(dataset)

    # Parse data
    data.parse(drop_columns=drop_features, feature_set=feature_set, drop_missing_value=drop_missing_value,
               out_dir=out_dir, exploration=data_exploration, external_validation=external_test_data)

    # Preprocess data
    X, Y = common_preprocessing(data, imputer=imputer, normaliser=normaliser, validation=False,
                                missing_threshold=missing_threshold, corr_threshold=correlation_threshold)
    log.info(f"{bars} Preprocessing complete {bars}")

    # Preprocess external validation data
    if external_test_data:
        X_val, Y_val = common_preprocessing(data, imputer=imputer, normaliser=normaliser, validation=True)
        log.info(X_val.columns.difference(X.columns))
        # Get rid of extra columns introduced by values in validation dataset
        log.info(f'Dropping columns in val data since they are missing in train data: {X_val.columns.difference(X.columns)}')
        X_val = X_val.drop(set(X_val.columns.difference(X.columns)), axis=1)
        assert len(X.columns.difference(
            X_val.columns)) == 0, (f'Error: Train data includes columns {X.columns.difference(X_val.columns)} that are '
                                   f'missing in val data')



    with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
        f.write(f'\n {bars} New Trial at {time.strftime("%d.%m.%Y %H:%M:%S")} {bars} \n')
        f.write(str(locals()))
        f.write('\n')
    seeds = [111, 222, 333, 444, 555, 666, 777, 888, 999]
    # seeds = [111,222]
    all_metrics_list = all_classification_metrics_list

    # For each endpoint
    all_model_metrics = pd.DataFrame(columns=["endpoint", "seed", "model", "validation", "test", "curves"])
    for k, endpoint in enumerate(Y.columns):
        # For each endpoint, we create dictionary for each seed
        # all_model_metrics[endpoint] = {seed: {} for seed in seeds}
        out_dir_endpoint = f'{out_dir}/{endpoint.replace(" ", "_")}'

        log.info(f'Predicting {endpoint}')
        # Set endpoint for iteration
        y = Y[endpoint]

        for seed in seeds:
            # all_test_metric_dfs[seed] = {metric: pd.DataFrame(dtype=np.float64) for metric in all_metrics_list if
            #                        metric != 'confusion_matrix'}
            # all_test_metric_dfs[seed] = {metric: pd.DataFrame(dtype=np.float64) for metric in all_metrics_list if
            #                        metric != 'confusion_matrix'}
            out_dir_seed = f'{out_dir_endpoint}/{seed}'
            os.makedirs(out_dir_seed, exist_ok=True)
            seed = seed
            np.random.seed(seed)
            random.seed(seed)
            with open(f'{out_dir_seed}/best_parameters.txt', 'a+') as f:
                f.write(f'=====\n{endpoint}\n=====')

            log.info(Y.info())

            # If we do not have an external validation dataset, we split the original dataset
            if not external_test_data:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction,
                                                                    random_state=seed, shuffle=True, stratify=y)
            else:
                # Set endpoint for iteration
                y_val = Y_val[endpoint]

                # Set train and test
                X_train = X
                y_train = y
                X_test = X_val
                y_test = y_val

            # Feature selection for each endpoint only on training data
            # X_test, X_train = model_feature_selection(X_test, X_train, y_train, min_feature_fraction=0.5, cores=cores,
            #                                           scoring=make_scorer(average_precision_score))

            # model grid
            model_grid = get_classification_model_grid('balanced' if balancing_option == 'class_weight' else None, seed=seed)
            for j, (model, param_grid) in enumerate(model_grid):
                val_metrics, test_metrics, curves = evaluate_single_model(model, param_grid,
                                                                          X_train, y_train, X_test, y_test,
                                                                          cv_splits=cv_splits,
                                                                          select_features=select_features,
                                                                          shap_value_eval=shap_eval,
                                                                          out_dir=out_dir_seed,
                                                                          sample_balancing=balancing_option,
                                                                          seed=seed)
                model_label = str(model.__class__.__name__)
                # all_model_metrics[endpoint][seed][model_label]= {"validation":val_metrics, "test": test_metrics, "curves": curves}
                all_model_metrics.loc[len(all_model_metrics.index)]= [endpoint, seed, model_label,val_metrics, test_metrics, curves]


            if (len(model_grid) > 1):
                # Save summary plots across models
                generate_summary_plots(all_model_metrics, endpoint, model, out_dir_seed, y)

        metrics = all_model_metrics
        # Unpack the test and validation results
        results = pd.concat([metrics.drop('test', axis=1), pd.DataFrame(metrics['test'].tolist()).add_suffix("_test")], axis=1)
        results = pd.concat(
            [results.drop('validation', axis=1), pd.DataFrame(metrics['validation'].tolist()).add_suffix("_validation")], axis=1)
        results.to_csv(f'{out_dir}/individual_metrics.csv')

        # Metrics to be aggregated
        metrics = ['balanced_accuracy', 'recall', 'precision', 'mcc', 'f1_score', 'roc_auc', 'avg_precision', 'prc_auc']
        # Aggregate results over seeds
        aggregated = results.groupby(['endpoint', 'model']).agg({item + "_test": ['mean', 'std','count','min','max'] for item in metrics})
        aggregated.to_csv(f'{out_dir}/aggregated_metrics.csv')

        # filehandler = open(f'{out_dir}/all_model_metrics.pkl', "wb")
        # pickle.dump(all_model_metrics, filehandler)
        # filehandler.close()

        # all_test_metric_dfs = {seed: {} for seed in seeds}

        # # Save summary results in dataframe
        # for seed in seeds:
        #     for model_name, test_data in {model_name: entry[1] for model_name, entry in all_model_metrics[seed].items()}.items():
        #         for metric, value in test_data.items():
        #             if metric == 'confusion_matrix':
        #                 continue
        #             all_test_metric_dfs[seed][metric].loc[model_name, label_col.replace(' ', '_')] = value
        #
        #
        # for metric, df in all_test_metric_dfs.items():
        #     df.to_csv(f'{out_dir}/data_frames/{metric}.csv')


def generate_summary_plots(all_model_metrics, label_col, model, out_dir, y):
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


def setup_output_directory(dataset, feature_set, out_dir):
    # Setup output directory
    # seed = seed
    # np.random.seed(seed)
    # random.seed(seed)
    now = str(datetime.now().strftime("%Y-%m-%d_T_%H-%M-%S"))
    if out_dir is None:
        # Set up an extra directory for this dataset
        feature_set_string = '' if feature_set is None else f'_{"_".join(feature_set)}'
        #out_dir = f'results_{dataset}{feature_set_string}_{now}_seed_{seed}'
        out_dir = f'results_{dataset}{feature_set_string}_{now}'
    else:
        # Output directory already exists, make subdirectory for this run
        feature_set_string = '' if feature_set is None else f'_{"_".join(feature_set)}'
        # out_dir = f'{out_dir}/results_{dataset}{feature_set_string}_{now}_seed_{seed}'
        out_dir = f'{out_dir}/results_{dataset}{feature_set_string}_{now}'
    log.info(f"Logging results to: {out_dir}")
    os.makedirs(f'{out_dir}', exist_ok=True)
    # os.makedirs(f'{out_dir}/data_frames', exist_ok=True)
    return out_dir

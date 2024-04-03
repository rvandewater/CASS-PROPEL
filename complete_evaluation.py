import os
import json
import pickle
import time
import random
from datetime import datetime

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

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
               select_features, cv_splits, shap_eval, test_fraction, artificial_balancing_option, drop_missing_value,
               data_exploration, missing_threshold, correlation_threshold, cores=1, offset=12):
    """
    Main function for evaluation of classical ML models on post-operative complications dataset.

    Args:
        seeds: Seeds to use for reproducibility
        out_dir: Output directory
        dataset: The dataset to process
        feature_set: If given, processes only features from all provided feature sets
        external_test_data: Use external validation dataset
        imputer: Imputer to use for missing values
        normaliser: Normalise features
        drop_features: Whether to drop predefined features
        select_features: Whether to perform feature selection
        cv_splits: Number of cross_validation splits; 1 denotes LOO-CV
        shap_eval: Turn on SHAP evaluation (Warning: Increases runtime)
        test_fraction: Size of the test set in fraction of total samples
        artificial_balancing_option: Artificial oversampling option
        drop_missing_value: Drop rows missing this percentage of columns
        data_exploration: Generate data exploration plots
        missing_threshold: Missing threshold for removing features in preprocessing
        correlation_threshold: Correlation threshold for removing features in preprocessing
        cores: Amount of cores to use for parallel processing
        offset: Temporal offset for dataset if applicable
    """
    out_dir, start_time = setup_output_directory(dataset, feature_set, out_dir, offset=offset)

    log.debug(f"Arguments: {locals()}")
    # Get DataInformation object for the specified task
    data = get_data_from_name(dataset, offset)

    if external_test_data:
        log.info(f"Using external test data from {data.validation_data_path}")

    # Parse data
    data.parse(drop_columns=drop_features, feature_set=feature_set, drop_missing_value=drop_missing_value,
               out_dir=out_dir, exploration=data_exploration, external_validation=external_test_data)

    # Preprocess data
    X, Y = common_preprocessing(data, imputer=imputer, normaliser=normaliser, validation=False,
                                missing_threshold=missing_threshold, corr_threshold=correlation_threshold)
    log.info(f"{bars} Preprocessing complete {bars}")
    log.info(list(X.columns))

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

    all_metrics_list = all_classification_metrics_list
    # For each endpoint
    all_model_metrics = pd.DataFrame(
        columns=["endpoint", "seed", "model", "fold", "validation", "test", "curves", "best_model", "shaps"])
    log.info(f"Seeds: {seeds}")
    # Iterate over endpoints
    for k, endpoint in enumerate(Y.columns):
        # all_model_metrics = pd.DataFrame(columns=["endpoint", "seed", "model", "fold", "validation", "test", "curves", "best_model", "shaps"])
        # For each endpoint, we create dictionary for each seed
        # all_model_metrics[endpoint] = {seed: {} for seed in seeds}
        out_dir_endpoint = f'{out_dir}/{endpoint.replace(" ", "_")}'
        log.info(f'Predicting {endpoint}')
        # Set endpoint for iteration
        y = Y[endpoint]
        # pretune = True
        # # # The amount of independent trails we want to do
        #
        # if pretune:
        #     model_scores = {}
        #     model_params = {}
        #     for seed in seeds:
        #         model_grid = get_classification_model_grid('balanced' if artificial_balancing_option == 'class_weight' else None, seed=seed)
        #         grid_model = pre_tune(X, y, model_grid[0])
        #         # grid_model.best_model.score(X,y)
        #         model_scores[seed] = grid_model.best_score_
        #         model_params[seed] = grid_model
        #     log.info(model_scores)
        #     pretune_model = max(model_scores)
        #     pretune_model = model_params[pretune_model]
        # model_scores = {111: ({'n_estimators': 750, 'min_child_weight': 0.5, 'max_depth': 3, 'learning_rate': 0.005, 'colsample_bytree': 0.1}, 0.4165744284536581), 222: ({'n_estimators': 100, 'min_child_weight': 0.5, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 1.0}, 0.40532207472238657), 333: ({'n_estimators': 50, 'min_child_weight': 0.5, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.5}, 0.405838096897264), 444: ({'n_estimators': 250, 'min_child_weight': 0.5, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.1}, 0.4085006083226691), 555: ({'n_estimators': 500, 'min_child_weight': 0.5, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.25}, 0.3923515498500847), 666: ({'n_estimators': 500, 'min_child_weight': 0.5, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 0.5}, 0.39398412387226495), 777: ({'n_estimators': 750, 'min_child_weight': 0.5, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 1.0}, 0.3968773189038742), 888: ({'n_estimators': 500, 'min_child_weight': 0.5, 'max_depth': 5, 'learning_rate': 0.01, 'colsample_bytree': 0.25}, 0.42302872919384454), 999: ({'n_estimators': 100, 'min_child_weight': 0.5, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 1.0}, 0.41035040320598526)}
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
            # with open(f'{out_dir_seed}/best_parameters.txt', 'a+') as f:
            #     f.write(f'=====\n{endpoint}\n=====')

            log.info(Y.info())
            # cv_splits = 3
            nested_cv = False

            if nested_cv:
                # log.info(f"Using pretuned model: {pretune_model}")
                model = nested_crossval(X, all_model_metrics, artificial_balancing_option, cv_splits, endpoint, out_dir_seed,
                                        seed,
                                        select_features, shap_eval, y)  # , pretune_model=pretune_model.best_estimator_)
            else:
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
                # model_grid = get_classification_model_grid('balanced' if artificial_balancing_option == 'class_weight' else None, seed=seed)
                model_grid = get_classification_model_grid(
                    'balanced' if artificial_balancing_option == 'class_weight' else None, seed=seed)

                for j, (model, param_grid) in enumerate(model_grid):
                    val_metrics, test_metrics, curves, best_model, shaps = evaluate_single_model(model, param_grid,
                                                                                                 X_train, y_train, X_test,
                                                                                                 y_test,
                                                                                                 cv_splits=cv_splits,
                                                                                                 select_features=select_features,
                                                                                                 shap_value_eval=shap_eval,
                                                                                                 out_dir=out_dir_seed,
                                                                                                 sample_balancing=artificial_balancing_option,
                                                                                                 seed=seed)

                    model_label = str(model.__class__.__name__)
                    # all_model_metrics[endpoint][seed][model_label]= {"validation":val_metrics, "test": test_metrics, "curves": curves}
                    # 0 for fold name
                    all_model_metrics.loc[len(all_model_metrics.index)] = [endpoint, seed, model_label, 0, val_metrics,
                                                                           test_metrics, curves, best_model, shaps]
                    # all_model_metrics.loc[len(all_model_metrics.index)] = [endpoint, seed, model_label, fold_iter, val_metrics,
                    #                                                        test_metrics, curves, best_model, shaps]

                # if (len(model_grid) > 1):
                #     # Save summary plots across models
                #     generate_summary_plots(all_model_metrics, endpoint, model, out_dir_seed, y)
                # pickle.dump(all_model_metrics, open(f'{out_dir_seed}/{model_label}_all_model_metrics.pkl', 'wb'))
            shap_tuples = all_model_metrics[all_model_metrics['model'] == model_label]['shaps']
            Shap_summary(X, model_label, out_dir_seed, shap_tuples)

        # Optional: save best hyperparams per model and endpoint
        metrics = all_model_metrics
        pickle.dump(metrics, open(f'{out_dir_seed}/all_model_metrics.pkl', 'wb'))
        # Unpack the test and validation results
        results = pd.concat([metrics.drop('test', axis=1), pd.DataFrame(metrics['test'].tolist()).add_suffix("_test")], axis=1)
        results = pd.concat(
            [results.drop('validation', axis=1), pd.DataFrame(metrics['validation'].tolist()).add_suffix("_validation")],
            axis=1)
        results.to_csv(f'{out_dir}/individual_metrics_{start_time}_offset_{offset}.csv')

        # Metrics to be aggregated
        metrics = ['balanced_accuracy', 'recall', 'precision', 'mcc', 'f1_score', 'roc_auc', 'avg_precision', 'prc_auc']
        # Convert list to numpy array

        # metrics.to_csv(f'{out_dir}/all_metrics_{start_time}_offset_{offset}.csv')
        # Aggregate results over seeds
        aggregated = results.groupby(['endpoint', 'model', 'seed']).agg(
            {item + "_test": ['mean', 'std', 'count', 'min', 'max'] for item in metrics})
        if offset is not None:
            if feature_set is not None:
                aggregated.to_csv(f'{out_dir}/aggregated_metrics_{start_time}_{"_".join(feature_set)}_offset_{offset}.csv')
            else:
                aggregated.to_csv(f'{out_dir}/aggregated_metrics_{start_time}_offset_{offset}.csv')
        else:
            aggregated.to_csv(f'{out_dir}/aggregated_metrics_{start_time}.csv')

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


def pre_tune(X, y, param_grid):
    # model_name = str(model.__class__.__name__)
    pipeline_steps = []
    # prepare param_grid
    # param_grid = {'model__' + key: value for (key, value) in param_grid.items()}
    select_features = False
    # if select_features:
    #     # Feature selection for each endpoint only on training data
    #     param_grid['selector'] = [
    #         model_feature_selection(X_test, X_train, y_train, min_feature_fraction=0.5, cores=1,
    #                                 scoring=make_scorer(average_precision_score))]
    #     param_grid['selector'] = [SelectFromModel(XGBClassifier())]
    #     # param_grid['selector'] = [SelectKBest(k='all'), SelectKBest(k=25),
    #     #                           SelectFromModel(LinearSVC(C=1, penalty="l1", dual=False, max_iter=5000))]
    #
    #     pipeline_steps.extend([('selector', 'passthrough'), ('model', model)])
    # else:
    model = param_grid[0]
    pipeline_steps.append(('model', model))
    pipeline = Pipeline(pipeline_steps)
    cv_scoring = 'average_precision'
    cv = 5
    log.info(f"Using pipeline {pipeline}")
    grid_model = RandomizedSearchCV(param_grid[0], param_distributions=param_grid[1], scoring=cv_scoring, verbose=False, cv=cv,
                                    n_jobs=-1,
                                    error_score=0, n_iter=30)
    grid_model.fit(X, y)
    return grid_model


def nested_crossval(X, all_model_metrics, balancing_option, cv_splits, endpoint, out_dir_seed, seed, select_features,
                    shap_eval, y, pretune_model=None):
    # Nested CV
    # model_grid = get_classification_model_grid('balanced' if artificial_balancing_option == 'class_weight' else None, seed=seed)
    model_grid = get_classification_model_grid(seed=seed)
    for j, (model, param_grid) in enumerate(model_grid):
        # if (pretune_model is not None):
        #     model = max(pretune_model)
        # stratified_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        skf = StratifiedShuffleSplit(n_splits=cv_splits, random_state=seed)
        model_label = str(model.__class__.__name__)
        fold_iter = 0
        for train_i, test_i in skf.split(X, y):
            log.debug(f'Outer fold: {fold_iter}')
            out_dir_cv = f'{out_dir_seed}/fold_{fold_iter}'
            X_train, X_test = X.iloc[train_i], X.iloc[test_i]
            y_train, y_test = y.iloc[train_i], y.iloc[test_i]
            val_metrics, test_metrics, curves, best_model, shaps = evaluate_single_model(model, param_grid,
                                                                                         X_train, y_train, X_test, y_test,
                                                                                         cv_splits=cv_splits,
                                                                                         select_features=select_features,
                                                                                         shap_value_eval=shap_eval,
                                                                                         out_dir=out_dir_cv,
                                                                                         sample_balancing=balancing_option,
                                                                                         seed=seed)

            # all_model_metrics[endpoint][seed][model_label]= {"validation":val_metrics, "test": test_metrics, "curves": curves}
            all_model_metrics.loc[len(all_model_metrics.index)] = [endpoint, seed, model_label, fold_iter, val_metrics,
                                                                   test_metrics, curves, best_model, shaps]
            fold_iter += 1
        pickle.dump(all_model_metrics, open(f'{out_dir_seed}/{model_label}_all_model_metrics.pkl', 'wb'))
        # shap_tuples = all_model_metrics[all_model_metrics['model'] == model_label]['shaps']
        # Shap_summary(X, model_label, out_dir_seed, shap_tuples)
    return model


def Shap_summary(X, model_label, out_dir_seed, shap_tuples):
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


def setup_output_directory(dataset, feature_set, out_dir, offset=0):
    # Setup output directory
    # seed = seed
    # np.random.seed(seed)
    # random.seed(seed)
    now = str(datetime.now().strftime("%Y-%m-%d_T_%H-%M-%S"))
    if out_dir is None:
        # Set up an extra directory for this dataset
        feature_set_string = '' if feature_set is None else f'_{"_".join(feature_set)}'
        # out_dir = f'results_{dataset}{feature_set_string}_{now}_seed_{seed}'
        out_dir = f'results_{dataset}{feature_set_string}_offset_{offset}_{now}'
    else:
        # Output directory already exists, make subdirectory for this run
        feature_set_string = '' if feature_set is None else f'_{"_".join(feature_set)}'
        # out_dir = f'{out_dir}/results_{dataset}{feature_set_string}_{now}_seed_{seed}'
        out_dir = f'{out_dir}/results_{dataset}{feature_set_string}_offset_{offset}_{now}'
    log.info(f"Logging results to: {out_dir}")
    os.makedirs(f'{out_dir}', exist_ok=True)
    # os.makedirs(f'{out_dir}/data_frames', exist_ok=True)
    return out_dir, now

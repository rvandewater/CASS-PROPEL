import os
import pickle
import time
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.data import get_data_from_name
from src.preprocess import common_preprocessing
from src.models import get_classification_model_grid
from src.evaluate import evaluate_single_model
from src.utils.file_creation import setup_output_directory
from src.utils.metrics import all_classification_metrics_list, Shap_summary
import logging as log

# import feature_engine
bars = '====================='


def evaluation(seeds, out_dir, dataset, feature_set, external_test_data, imputer, normaliser, drop_features,
               select_features, cv_splits, shap_eval, test_fraction, artificial_balancing_option, drop_missing_value,
               data_exploration, missing_threshold, correlation_threshold, nested_cv=True, cores=1, offset=12):
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
        nested_cv: Use nested CV (only possible if external_test_data is False)
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
    # For each endpoint create a dataframe to store all metrics
    all_model_metrics = pd.DataFrame(
        columns=["endpoint", "seed", "model", "fold", "validation", "test", "curves", "best_model", "shaps"])

    # Iterate over endpoints
    for k, endpoint in enumerate(Y.columns):
        out_dir_endpoint = f'{out_dir}/{endpoint.replace(" ", "_")}'
        log.info(f'Predicting {endpoint}')
        # Set endpoint for iteration
        y = Y[endpoint]
        pretune_model = None
        if pretune:
            pretune_model = (X, artificial_balancing_option, seeds, y)
        # The amount of independent trails we want to do
        for seed in seeds:
            out_dir_seed = f'{out_dir_endpoint}/{seed}'
            os.makedirs(out_dir_seed, exist_ok=True)
            seed = seed
            np.random.seed(seed)
            random.seed(seed)
            # with open(f'{out_dir_seed}/best_parameters.txt', 'a+') as f:
            #     f.write(f'=====\n{endpoint}\n=====')

            log.info(Y.info())

            if nested_cv:
                if not external_test_data:
                    model = nested_crossval(X, all_model_metrics, artificial_balancing_option, cv_splits, endpoint,
                                            out_dir_seed, seed, select_features, shap_eval, y, pretune_model=pretune_model)
                else:
                    raise ValueError('Nested CV is not possible with external validation data')
            else:
                # If we do not have an external validation dataset, we split the original dataset
                if not external_test_data:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction,
                                                                        random_state=seed, shuffle=True, stratify=y)
                else:
                    # If we have an external validation dataset, we use the original dataset as training data
                    X_train = X
                    y_train = y
                    # Set test data to be external validation data
                    X_test = X_val
                    y_test = Y_val

                # Feature selection for each endpoint only on training data
                # X_test, X_train = model_feature_selection(X_test, X_train, y_train, min_feature_fraction=0.5, cores=cores,
                #                                           scoring=make_scorer(average_precision_score))
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
                    # 0 for fold name
                    all_model_metrics.loc[len(all_model_metrics.index)] = [endpoint, seed, model_label, 0, val_metrics,
                                                                           test_metrics, curves, best_model, shaps]
                    # all_model_metrics.loc[len(all_model_metrics.index)] = [endpoint, seed, model_label, fold_iter, val_metrics,
                    #                                                        test_metrics, curves, best_model, shaps]

                # if (len(model_grid) > 1):
                #     # Save summary plots across models
                #     generate_summary_plots(all_model_metrics, endpoint, model, out_dir_seed, y)

            # We take the shap values from all models and generate one SHAP plot per seed
            for model_name in all_model_metrics['model'].unique():
                shap_tuples = all_model_metrics[all_model_metrics['model'] == model_name]['shaps']
                Shap_summary(X, model_name, out_dir_seed, shap_tuples)

        metric_summarization(all_model_metrics, feature_set, offset, out_dir, start_time)


def metric_summarization(all_model_metrics, feature_set, offset, out_dir, start_time):
    metrics = all_model_metrics
    # Unpack the test and validation results
    results = pd.concat([metrics.drop('test', axis=1), pd.DataFrame(metrics['test'].tolist()).add_suffix("_test")], axis=1)
    results = pd.concat(
        [results.drop('validation', axis=1), pd.DataFrame(metrics['validation'].tolist()).add_suffix("_validation")],
        axis=1)
    results.to_csv(f'{out_dir}/individual_metrics_{start_time}_offset_{offset}.csv')
    # Metrics to be aggregated and appear in result sheet
    metrics = ['balanced_accuracy', 'recall', 'precision', 'mcc', 'f1_score', 'roc_auc', 'avg_precision', 'prc_auc']
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
    pickle.dump(metrics, open(f'{out_dir}/all_metrics.pkl', 'wb'))
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


def pretune(X, artificial_balancing_option, seeds, y):
    model_scores = {}
    model_params = {}
    for seed in seeds:
        model_grid = get_classification_model_grid('balanced' if artificial_balancing_option == 'class_weight' else None,
                                                   seed=seed)
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
        model = model_grid[0]
        pipeline_steps.append(('model', model))
        pipeline = Pipeline(pipeline_steps)
        cv_scoring = 'average_precision'
        cv = 5
        log.info(f"Using pipeline {pipeline}")
        grid_model = RandomizedSearchCV(model_grid[0], param_distributions=model_grid[1], scoring=cv_scoring, verbose=False,
                                        cv=cv,
                                        n_jobs=-1,
                                        error_score=0, n_iter=30)
        grid_model.fit(X, y)
        # grid_model.best_model.score(X,y)
        model_scores[seed] = grid_model.best_score_
        model_params[seed] = grid_model
    log.info(model_scores)
    pretune_model = max(model_scores)
    pretune_model = model_params[pretune_model]
    log.info(f"Using pretuned model: {pretune_model}")
    return pretune_model.best_estimator_


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

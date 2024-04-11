import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer, MissingIndicator
from xgboost import XGBClassifier
import warnings
# from src.utils.feature_selector import FeatureSelector
from src.data.abstract_dataset import Dataset
import logging as log


def common_preprocessing(data: Dataset,
                         imputer="knn",
                         normaliser='standard',
                         missing_threshold=0,
                         corr_threshold=0,
                         validation=False):
    """
    Preprocesses the data by applying one-hot-encoding to categorical features, imputing missing values and scaling

    Args:
        data: Dataframe to preprocess.
        imputer: Imputation method for missing values.
        normaliser: Scaling method for numerical features.
        validation: Whether to use the validation dataset instead of the training dataset.
    Returns:
        x: Preprocessed feature df.
        Y: Preprocessed label df.

    """
    # Choice of validation dataset
    if not validation:
        X, Y = data.get_data()
    else:
        X, Y = data.get_val_data()
        if validation:
            validation_samples = X.shape[0]
        # Get original data and append (in order for proper one-hot encoding)
        X_or, Y_or = data.get_data()
        X = pd.concat([X, X_or], ignore_index=True)
        Y = pd.concat([Y, Y_or], ignore_index=True)

    # Drop rows with missing values
    if missing_threshold > 0:
        log.info(f"Shape of x: {X.shape}, dropping rows with missing values {missing_threshold}")
        row_indices = X.dropna(thresh = missing_threshold, axis = 0)
        X.dropna(thresh=int(len(X.columns) * missing_threshold), axis=0, inplace=True)
        log.info(f"Shape of x after dropping: {X.shape}")
        intersection = row_indices.index.intersection(X.index)
        Y = Y.loc[intersection]

    log.debug("Categorical features:")
    log.debug(data.get_categorical_features())
    categorical_features = [col for col in X.columns if col in data.get_categorical_features()]
    X[categorical_features] = X[categorical_features].fillna(value=0)

    # One-hot-encode categorical features
    X = pd.get_dummies(X, columns=categorical_features, dummy_na=True, dtype="float64")


    # Apply FeatureSelector functionality
    # if len(fs_operations) > 0:
    #     fs = FeatureSelector()
    #     if 'single_unique' in fs_operations:
    #         fs.identify_single_unique(x)
    #     if 'missing' in fs_operations:
    #         fs.identify_missing(x, missing_threshold=missing_threshold)
    #     if 'collinear' in fs_operations:
    #         fs.identify_collinear(x, correlation_threshold=correlation_threshold)
    #         log.info(fs.removal_ops['collinear'])
    #     x = fs.remove(x, fs_operations, one_hot=False)
    if corr_threshold > 0:
        X = drop_correlated(X, corr_threshold)

    # Fix strings in Binary columns
    for binary_col in data.get_binary_features():
        if binary_col in X.columns:
            unique_vals = list(np.unique(X[binary_col].values))
            if len(unique_vals) < 2:
                warnings.warn(f"Binary column {binary_col} has less than 2 unique values.", UserWarning)
            if len(unique_vals) != 1 and unique_vals != [0, 1]:
                log.debug(f'Renaming entries from {binary_col}: {unique_vals[0]} -> 0; {unique_vals[1]} -> 1')
                X[binary_col].replace({unique_vals[0]: 0,
                                       unique_vals[1]: 1}, inplace=True)


    X_numerical = X[[col for col in X.columns if col in data.get_numerical_features()]]
    X_binary = X.drop(columns=[col for col in X.columns if col in data.get_numerical_features()])
    X_numerical_feature_names = X_numerical.columns



    # Interpolate numerical features
    if imputer is not None:
        X_miss_numerical = generate_missing_indicators(X_numerical)
        X_miss_binary = generate_missing_indicators(X_binary)
        log.info(f'Running {imputer} imputer...')
        if imputer == 'knn':
            imputer = KNNImputer(n_neighbors=5, weights='uniform')
        elif imputer in ['mean', 'median']:
            imputer = SimpleImputer(strategy=imputer)
        else:
            raise ValueError(f'Imputer type {imputer} not supported')
        X_numerical = imputer.fit_transform(X_numerical)
        X_numerical = pd.DataFrame(X_numerical, columns=X_numerical_feature_names)
        X_binary = X_binary.fillna(value=0)

    if normaliser is not None:
        # Normalise numerical features
        log.info('Normalising numerical features...')
        if normaliser == 'standard':
            scaler = StandardScaler()
        elif normaliser == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f'Normaliser type {normaliser} not supported')
        X_numerical = scaler.fit_transform(X_numerical)
        standardScaler = StandardScaler()
        X_numerical = standardScaler.fit_transform(X_numerical)
        X_numerical = pd.DataFrame(X_numerical, columns=X_numerical_feature_names)

    # Build x and y arrays
    # x = pd.concat([X_binary, data], axis=1)
    log.info(f"Intersection:{len(X_numerical.index.intersection(X_binary.index))}")
    X = X_binary.join(X_numerical, how="outer")
    # x = x.loc[row_indices.index]

    if validation:
        X = X.head(validation_samples)
        Y = Y.head(validation_samples)
        Y = Y.fillna(value=0)

    log.info(f'Class distributions for {len(Y)} data points, validation={validation}:')
    for y_col in Y:
        log.info(f'Endpoint {y_col}:')
        if y_col in data.get_numerical_endpoints():
            log.debug("Endpoints:")
            log.debug(Y[y_col].describe())
        else:
            abs_value_counts = Y[y_col].value_counts()
            rel_value_counts = Y[y_col].value_counts(normalize=True)
            for i in range(len(abs_value_counts.index)):
                log.info(
                    f'\tClass "{abs_value_counts.index[i]}":\t{abs_value_counts.iloc[i]} ({rel_value_counts.iloc[i]:.3f})')
        log.info('\n')
    if imputer is not None:
        # Add missing indicator if imputation was performed
        X = X.join(X_miss_numerical)
        X = X.join(X_miss_binary)
    return X, Y


def generate_missing_indicators(data):
    indicator = MissingIndicator(features='all')
    data_miss = pd.DataFrame()
    if len(data.columns) > 0:
        data_miss = indicator.fit_transform(data)
        data_miss = pd.DataFrame(data_miss, columns=[f'{col}_missing' for col in data.columns])
        data_miss = data_miss.astype("int")
    return data_miss


def drop_correlated(df_in, threshold):
    df_corr = df_in.corr(method='pearson', min_periods=1)
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)] * 2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = df_in[un_corr_idx]
    log.info(f'Dropped {len(df_in.columns) - len(df_out.columns)} columns due to high correlation')
    return df_out


def model_feature_selection(X_test: pd.DataFrame,
                            X_train: pd.DataFrame,
                            y_train: pd.DataFrame,
                            min_feature_fraction: float = 0.5,
                            cores: int = -1,
                            scoring: sklearn.metrics = make_scorer(average_precision_score)) -> pd.DataFrame:
    """
    Performs feature selection on the training data and applies the same selection to the test data.
    Args:
        X_test:
        X_train:
        cores:
        y_train:
        min_feature_fraction: Minimal fraction of features to keep
        scoring: Which scoring function to use for feature selection

    Returns:

    """
    log.debug(f"Columns before feature selection: {X_train.columns}")
    xgb_model = XGBClassifier()
    n_features = int(len(X_train.columns) * min_feature_fraction)

    # Cross validated feature selection
    select = RFECV(estimator=xgb_model, min_features_to_select=n_features, verbose=0, cv=5, scoring=scoring, n_jobs=cores)
    return select
    # select = select.fit(x_train, y_train)
    #
    # # Get the selection mask
    # mask = select.get_support()
    # x_train = x_train.loc[:, mask]  # select.transform(x_train)
    # x_test = x_test.loc[:, mask]
    # log.info(f"Columns after feature selection: {x_train.columns}")
    # return x_test, x_train

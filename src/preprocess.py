import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer

# from src.utils.feature_selector import FeatureSelector
from src.data.abstract_dataset import Dataset
import logging as log


def common_preprocessing(data: Dataset,
                         imputer="knn",
                         normaliser='standard',
                         missing_threshold=0.5,
                         corr_threshold=0.95,
                         validation=False):
    """
    Preprocesses the data by applying one-hot-encoding to categorical features, imputing missing values and scaling

    Args:
        data: Dataframe to preprocess.
        imputer: Imputation method for missing values.
        normaliser: Scaling method for numerical features.
        validation: Whether to use the validation dataset instead of the training dataset.
    Returns:
        X: Preprocessed feature df.
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

    X.dropna(thresh = len(X)*missing_threshold, axis = 1, inplace = True)

    log.info("Categorical features:")
    log.info(data.get_categorical_features())
    categorical_features = [col for col in X.columns if col in data.get_categorical_features()]
    X[categorical_features] = X[categorical_features].fillna(value=0)
    # if "Vorbehandlung" in X.columns:
    #     X["Vorbehandlung"] = X["Vorbehandlung"].astype("int")
    # if "Histologie" in X.columns:
    #     X["Histologie"] = X["Histologie"].astype("int")

    # One-hot-encode categorical features
    X = pd.get_dummies(X, columns=categorical_features, dummy_na=False)


    # Apply FeatureSelector functionality
    # if len(fs_operations) > 0:
    #     fs = FeatureSelector()
    #     if 'single_unique' in fs_operations:
    #         fs.identify_single_unique(X)
    #     if 'missing' in fs_operations:
    #         fs.identify_missing(X, missing_threshold=missing_threshold)
    #     if 'collinear' in fs_operations:
    #         fs.identify_collinear(X, correlation_threshold=correlation_threshold)
    #         log.info(fs.removal_ops['collinear'])
    #     X = fs.remove(X, fs_operations, one_hot=False)

    X = drop_correlated(X, corr_threshold)

    # Fix strings in Binary columns
    for binary_col in data.get_binary_features():
        if binary_col in X.columns:
            unique_vals = list(np.unique(X[binary_col].values))
            if len(unique_vals) < 2:
                raise AssertionError(f"Binary column {binary_col} has less than 2 unique values.")
            if len(unique_vals) != 1 and unique_vals != [0, 1]:
                log.warning(f'Renaming entries from {binary_col}: {unique_vals[0]} -> 0; {unique_vals[1]} -> 1')
                X[binary_col].replace({unique_vals[0]: 0,
                                       unique_vals[1]: 1}, inplace=True)

    X_numerical = X[[col for col in X.columns if col in data.get_numerical_features()]]
    X_binary = X.drop(columns=[col for col in X.columns if col in data.get_numerical_features()])
    X_numerical_feature_names = X_numerical.columns

    # Interpolate numerical features
    if imputer is not None:
        log.info(f'Running {imputer} Imputer...')
        # Unstable, so pro
        # if imputer == 'iterative':
        #     imputer = IterativeImputer(max_iter=1000)
        if imputer == 'knn':
            imputer = KNNImputer(n_neighbors=5, weights='uniform')
        elif imputer in ['mean', 'median']:
            imputer = SimpleImputer(strategy=imputer)
        else:
            raise ValueError(f'Imputer type {imputer} not supported')
        X_numerical = imputer.fit_transform(X_numerical)
        X_numerical = pd.DataFrame(X_numerical, columns=X_numerical_feature_names)

        log.info("Imputing binary features...")
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

    # Build X and y arrays
    X = pd.concat([X_binary, X_numerical], axis=1)

    if validation:
        X = X.head(validation_samples)
        Y = Y.head(validation_samples)
        Y = Y.fillna(value=0)

    log.info(f'Class distributions for {len(Y)} data points, validation={validation}:')
    for y_col in Y:
        log.info(f'Endpoint {y_col}:')
        if y_col in data.get_numerical_endpoints():
            log.info(Y[y_col].describe())
        else:
            abs_value_counts = Y[y_col].value_counts()
            rel_value_counts = Y[y_col].value_counts(normalize=True)
            for i in range(len(abs_value_counts.index)):
                log.info(
                    f'\tClass "{abs_value_counts.index[i]}":\t{abs_value_counts.iloc[i]} ({rel_value_counts.iloc[i]:.3f})')
        log.info('\n')
    return X, Y

def drop_correlated(df_in, threshold):
    df_corr = df_in.corr(method='pearson', min_periods=1)
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)] * 2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = df_in[un_corr_idx]
    log.info(f'Dropped {len(df_in.columns) - len(df_out.columns)} columns due to high correlation')
    return df_out

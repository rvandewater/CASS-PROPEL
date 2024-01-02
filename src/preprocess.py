import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from src.utils.feature_selector import FeatureSelector
from src.data.abstract_dataset import Dataset


def get_preprocessed_data(data: Dataset,
                          fs_operations=None,
                          missing_threshold=0.5,
                          correlation_threshold=0.95,
                          imputer="knn",
                          normaliser='standard',
                          verbose=False,
                          validation=False):
    """Return X and Y after applying specified preprocessing steps

    Parameters
    ----------
    validation
    data : Dataset
        The parsed data
    fs_operations : list, default=['missing', 'single_unique', 'collinear']
        The feature selection operations to perform with FeatureSelector instance
    missing_threshold : float, default=0.5
        The threshold for removing features with missing values
    correlation_threshold : float, default=0.95
        The threshold for removing collinear features
    impute : bool, default=True
        Whether to impute missing values
    normalise : bool, default=True
        Whether to normalise numerical values
    verbose : bool, default=False
        Turns on verbose output
    validation : bool
        Whether to process the validation dataset
    Returns
    -------
    X : DataFrame
        The feature values
    Y : DataFrame
        The endpoints
    """
    if fs_operations is None:
        fs_operations = [] #['missing', 'single_unique', 'collinear']

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

    # Apply FeatureSelector functionality
    if len(fs_operations) > 0:
        fs = FeatureSelector()
        if 'single_unique' in fs_operations:
            fs.identify_single_unique(X)
        if 'missing' in fs_operations:
            fs.identify_missing(X, missing_threshold=missing_threshold)
        if 'collinear' in fs_operations:
            fs.identify_collinear(X, correlation_threshold=correlation_threshold)
            print(fs.removal_ops['collinear'])
        X = fs.remove(X, fs_operations, one_hot=False)

    # Fix strings in Binary columns
    for binary_col in data.get_binary_features():
        if binary_col in X.columns:
            unique_vals = list(np.unique(X[binary_col].values))
            if len(unique_vals) < 2:
                raise AssertionError(f"Binary column {binary_col} has less than 2 unique values.")
            if len(unique_vals) != 1 and unique_vals != [0, 1]:
                if verbose:
                    print(f'Renaming entries from {binary_col}: {unique_vals[0]} -> 0; {unique_vals[1]} -> 1')
                X[binary_col].replace({unique_vals[0]: 0,
                                       unique_vals[1]: 1}, inplace=True)

    print(data.get_categorical_features())
    categorical_features = [col for col in X.columns if col in data.get_categorical_features()]
    X[categorical_features] = X[categorical_features].fillna(value=0)
    # if "Vorbehandlung" in X.columns:
    #     X["Vorbehandlung"] = X["Vorbehandlung"].astype("int")
    # if "Histologie" in X.columns:
    #     X["Histologie"] = X["Histologie"].astype("int")

    # One-hot-encode categorical features
    X = pd.get_dummies(X, columns=categorical_features, dummy_na=False)

    X_numerical = X[[col for col in X.columns if col in data.get_numerical_features()]]
    X_binary = X.drop(columns=[col for col in X.columns if col in data.get_numerical_features()])

    X_numerical_feature_names = X_numerical.columns
    # Interpolate numerical features

    if imputer is not None:
        print(f'Running {imputer} Imputer...')
        if imputer == 'iterative':
            imputer = IterativeImputer(max_iter=1000, verbose=verbose)
        elif imputer == 'knn':
            imputer = KNNImputer(n_neighbors=5, weights='uniform')
        elif imputer in ['mean', 'median']:
            imputer = SimpleImputer(strategy=imputer)
        else:
            raise ValueError(f'Imputer type {imputer} not supported')
        X_numerical = imputer.fit_transform(X_numerical)
        X_numerical = pd.DataFrame(X_numerical, columns=X_numerical_feature_names)
        print("Imputing binary features...")
        X_binary = X_binary.fillna(value=0)

    if normaliser is not None:
        # Normalise numerical features
        print('Normalising numerical features...')
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

    print(f'Class distributions for {len(Y)} data points, validation={validation}:')
    for y_col in Y:
        print(f'Endpoint {y_col}:')
        if y_col in data.get_numerical_endpoints():
            print(Y[y_col].describe())
        else:
            abs_value_counts = Y[y_col].value_counts()
            rel_value_counts = Y[y_col].value_counts(normalize=True)
            for i in range(len(abs_value_counts.index)):
                print(f'\tClass "{abs_value_counts.index[i]}":\t{abs_value_counts.iloc[i]} ({rel_value_counts.iloc[i]:.3f})')
        print('\n')
    return X, Y

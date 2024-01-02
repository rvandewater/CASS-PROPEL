import pandas as pd
from src.utils.plot import data_exploration_comparison, data_exploration


class Dataset:
    def __init__(self,
                 data_path,
                 data_infos_path,
                 validation_data_path=None):
        self.path = data_path
        self.data_infos = pd.read_csv(data_infos_path)
        # External validation data
        self.validation_data_path = validation_data_path

        self.feature_set = None
        self.X = None
        self.Y = None

    def get_all_features(self, include_drop_columns=True):
        """Return list of names of all features

        Parameters
        ----------
        include_drop_columns : bool, default=True
            Include columns set as drop columns in data_infos

        Returns
        -------
        list
            a list of feature names
        """
        return list(self.data_infos.loc[~self.data_infos['endpoint'] & (
                    include_drop_columns | ~self.data_infos['drop']), 'column_name'])

    def get_binary_features(self, include_drop_columns=True):
        """Return list of names of all binary features

        Parameters
        ----------
        include_drop_columns : bool, default=True
            Include columns set as drop columns in data_infos

        Returns
        -------
        list
            a list of feature names
        """
        return list(self.data_infos.loc[~self.data_infos['endpoint'] &
                                        (include_drop_columns | ~self.data_infos['drop']) &
                                        (self.data_infos['data_type'] == 'B'), 'column_name'])

    def get_categorical_features(self, include_drop_columns=True):
        """Return list of names of all categorical features

        Parameters
        ----------
        include_drop_columns : bool, default=True
            Include columns set as drop columns in data_infos

        Returns
        -------
        list
            a list of feature names
        """
        return list(self.data_infos.loc[~self.data_infos['endpoint'] &
                                        (include_drop_columns | ~self.data_infos['drop']) &
                                        (self.data_infos['data_type'] == 'C'), 'column_name'])

    def get_numerical_features(self, include_drop_columns=True):
        """Return list of names of all numerical features

        Parameters
        ----------
        include_drop_columns : bool, default=True
            Include columns set as drop columns in data_infos

        Returns
        -------
        list
            a list of feature names
        """
        return list(self.data_infos.loc[~self.data_infos['endpoint'] &
                                        (include_drop_columns | ~self.data_infos['drop']) &
                                        (self.data_infos['data_type'] == 'N'), 'column_name'])

    def get_all_endpoints(self, include_drop_columns=True):
        """Return list of names of all endpoints

        Parameters
        ----------
        include_drop_columns : bool, default=True
            Include columns set as drop columns in data_infos

        Returns
        -------
        list
            a list of endpoint names
        """
        return list(self.data_infos.loc[
                        self.data_infos['endpoint'] & (include_drop_columns | ~self.data_infos['drop']), 'column_name'])

    def get_binary_endpoints(self, include_drop_columns=True):
        """Return list of names of all binary endpoints

        Parameters
        ----------
        include_drop_columns : bool, default=True
            Include columns set as drop columns in data_infos

        Returns
        -------
        list
            a list of endpoint names
        """
        return list(self.data_infos.loc[self.data_infos['endpoint'] &
                                        (include_drop_columns | ~self.data_infos['drop']) &
                                        (self.data_infos['data_type'] == 'B'), 'column_name'])

    def get_categorical_endpoints(self, include_drop_columns=True):
        """Return list of names of all categorical endpoints

        Parameters
        ----------
        include_drop_columns : bool, default=True
            Include columns set as drop columns in data_infos

        Returns
        -------
        list
            a list of endpoint names
        """
        return list(self.data_infos.loc[self.data_infos['endpoint'] &
                                        (include_drop_columns | ~self.data_infos['drop']) &
                                        (self.data_infos['data_type'] == 'C'), 'column_name'])

    def get_numerical_endpoints(self, include_drop_columns=True):
        """Return list of names of all numerical endpoints

        Parameters
        ----------
        include_drop_columns : bool, default=True
            Include columns set as drop columns in data_infos

        Returns
        -------
        list
            a list of endpoint names
        """
        return list(self.data_infos.loc[self.data_infos['endpoint'] &
                                        (include_drop_columns | ~self.data_infos['drop']) &
                                        (self.data_infos['data_type'] == 'N'), 'column_name'])

    def get_X(self):
        """Return data X if parse() has been called"""
        if self.X is None:
            raise RuntimeError('Please call parse() before accessing data')
        return self.X

    def get_Y(self):
        """Return endpoints Y if parse() has been called"""
        if self.Y is None:
            raise RuntimeError('Please call parse() before accessing data')
        return self.Y

    def get_data(self):
        """Return data X and endpoints Y if parse() has been called"""
        if self.X is None:
            raise RuntimeError('Please call parse() before accessing data')
        return self.X, self.Y

    def get_val_data(self):
        """Return validation X and endpoints Y if parse() has been called"""
        if self.X_val is None:
            raise RuntimeError('Please call parse() before accessing data and make sure you have specified validation '
                               'data')
        return self.X_val, self.Y_val

    def get_feature_set(self):
        """Return information about feature set if already parsed ("pre", "intra", "post", "dyn")"""
        if self.feature_set is None:
            raise RuntimeError('Please call parse() before accessing `feature_set`')
        return self.feature_set

    def parse(self, drop_columns=True, feature_set=None, drop_missing_value=0, out_dir="", exploration=False, external_validation=False):
        """Parse dataframe according to parameters and fill X and Y class attributes

        Parameters
        ----------
        exploration : bool
            Whether to generate a data exploration report
        out_dir : basestring
            Output directory path
        drop_missing_value : int, optional
            Drop rows missing this percentage of columns
        drop_columns : bool, default=True
            Drop the columns/features determined as drop columns in data infos
        feature_set : list, optional
            List including any of "pre", "intra", "post", "dyn", defining the feature set to parse
        """
        self.feature_set = feature_set
        original_data, external_validation_data = self.read_csv()
        # print(validation_data.head())
        # print(df.head())

        # Assert the length of the intersection
        assert len(
            set(original_data.columns).intersection(set(self.get_all_features() + self.get_all_endpoints()))) == len(
            set(original_data.columns)), "Column set doesn't match: " + \
            str([col for col in original_data.columns if col not in set(original_data.columns).intersection(
                set(self.get_all_features() + self.get_all_endpoints()))])

        drop_columns_list = []
        # Drop the columns that are not in the validation data
        if external_validation_data is not None:
            drop_columns_list.extend(set(original_data.columns).difference(set(external_validation_data.columns)))

        print(drop_columns_list)

        if feature_set is not None:
            drop_columns_list.extend(list(self.data_infos.loc[
                                              ~self.data_infos['endpoint'] & ~self.data_infos['input_time'].isin(
                                                  feature_set), 'column_name']))

        if drop_columns:
            drop_columns_list.extend(list(self.data_infos.loc[self.data_infos['drop'], 'column_name']))
            # Remove the updated values from esophagus_info_updated
            print(list(set(self.data_infos.column_name.values).difference(set(original_data.columns.values))))
            difference = list(set(self.data_infos.column_name.values).difference(set(original_data.columns.values)))
            self.data_infos = self.data_infos[~self.data_infos["column_name"].isin(difference)]
        original_data.drop(columns=drop_columns_list, inplace=True, errors='ignore')

        if drop_missing_value > 0:
            # Calculate the minimum amount of columns that have to contain a value
            min_count = int(((100 - drop_missing_value) / 100) * original_data.shape[1] + 1)

            # Drop rows not meeting threshold
            original_data = original_data.dropna(axis=0, thresh=min_count)

            # Do the same for external validation data
            if external_validation:
                external_validation_data = external_validation_data.dropna(axis=0, thresh=min_count)

        # Extract features and endpoints
        self.X = original_data[self.data_infos.loc[
            ~(self.data_infos['endpoint'] | self.data_infos['column_name'].isin(drop_columns_list)), 'column_name']]
        self.Y = original_data[self.data_infos.loc[
            self.data_infos['endpoint'] & ~self.data_infos['column_name'].isin(drop_columns_list), 'column_name']]

        if exploration:
            # Generate data exploration comparison report
            if external_validation:
                for endpoint in self.Y:
                    data_exploration_comparison(original_data, external_validation_data, endpoint, out_dir)
            else:
                for endpoint in self.Y:
                    data_exploration(original_data, endpoint, out_dir)

        if external_validation_data is not None:
            self.X_val = external_validation_data[self.data_infos.loc[
                ~(self.data_infos['endpoint'] | self.data_infos['column_name'].isin(drop_columns_list)), 'column_name']]
            self.Y_val = external_validation_data[self.data_infos.loc[
                self.data_infos['endpoint'] & ~self.data_infos['column_name'].isin(drop_columns_list), 'column_name']]
        else:
            self.X_val = None
            self.Y_val = None

    def read_csv(self):
        """Subclass-specific implementation of reading in data CSV file"""
        raise NotImplementedError

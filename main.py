import argparse

from complete_evaluation import evaluation


def get_parser():
    parser = argparse.ArgumentParser('Evaluate classical ML models on post-operative complications dataset.\n' +
                                     'Test metrics correspond to the results of a classification threshold optimised ' +
                                     'based on the optimal F1-score.')

    parser.add_argument('dataset', type=str, choices=['esophagus', 'pancreas', 'stomach', 'cass_lab', 'cass_preop_elect', 'cass_preop_emerg'],
                        help='The dataset to process.')
    parser.add_argument('--feature_set', '-f', nargs='*', type=str, choices=['pre', 'intra', 'post', 'dyn'],
                        help='if given, processes only features from all provided feature sets')
    parser.add_argument('--external_testset', '-e', action='store_true',
                        help='if specified, external validation dataset will be used as test data')
    parser.add_argument('--out_dir', '-o', type=str,
                        help='output directory')
    parser.add_argument('--no_features_dropped', '-nfd', action='store_false', dest='drop_features',
                        help='deactivates dropping predefined features in dataframe')
    parser.add_argument('--no_feature_selection', '-nfs', action='store_false', dest='select_features',
                        help='deactivates feature selection in pipeline')
    parser.add_argument('--cv_splits', '-cv', type=int, default=8,
                        help='number of cross_validation splits; 1 denotes LOO-CV')
    parser.add_argument('--shap_eval', '-sh', type=bool, default=False,
                        help='if true, shap values will be evaluated. Disabled by default, since it increases runtime a lot.')
    parser.add_argument('--test_fraction', '-t', type=float, default=0.2,
                        help='size of the test set in fraction of total samples')
    parser.add_argument('--balancing_option', '-b', type=str, default='class_weight',
                        choices=['class_weight', 'random_oversampling', 'SMOTE', 'ADASYN', 'none'],
                        help='technique to deal with imbalanced data')
    parser.add_argument('--drop_missing_value', '-dr', type=float, default=0,
                        help='Drop rows with x% of columns having missing values')
    parser.add_argument('--missing_threshold', '-mt', type=float, default=0.5,
                        help='Threshold for dropping columns with missing values')
    parser.add_argument('--correlation_threshold', '-ct', type=float, default=0.95,
                        help='Threshold for dropping columns with high correlation')
    parser.add_argument('--data_exploration', '-ex', nargs='*', type=bool, default=False,
                        help='If true, an html file will be generated showing statistics of the parsed dataset')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='If true, a seed will be set for reproducibility')

    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    args = arg_parser.parse_args()
    evaluation(args)

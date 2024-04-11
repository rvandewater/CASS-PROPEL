import argparse
import logging
import warnings
from src.complete_evaluation import evaluation


def get_parser():
    parser = argparse.ArgumentParser('Evaluate classical ML models on post-operative complications dataset.)')
    parser.add_argument('dataset', type=str,
                        choices=['esophagus', 'pancreas', 'stomach', 'cass_prop', 'cass_preop_elect', 'cass_preop_emerg'],
                        help='The dataset to process.')
    parser.add_argument('--feature_set', '-f', nargs='*', type=str, choices=["wearable", "ishmed", "copra", "pre"],
                        help='if given, processes only features from all provided feature sets')
    parser.add_argument('--external_test_data', '-e', action='store_true',
                        help='If specified, external validation dataset will be used as test data')
    parser.add_argument('--imputer', '-i', choices=['iterative', 'knn', 'mean'], nargs='?', const='knn', default="knn",
                        help='Which imputer to use for missing values')
    parser.add_argument('--normaliser', '-n', choices=['standard', 'minmax'], nargs='?', const='standard', default="standard",
                        help='Which normaliser to use to scale numerical values')
    parser.add_argument('--out_dir', '-o', type=str,
                        help='output directory')
    parser.add_argument('--no_features_dropped', '-nfd', action='store_false', dest='drop_features',
                        help='deactivates dropping predefined features in dataframe')
    parser.add_argument('--feature_selection', '-nfs', type=bool, dest='select_features', default=False,
                        help='deactivates feature selection in pipeline')
    parser.add_argument('--cv_splits', '-cv', type=int, default=5,
                        help='number of cross_validation splits; 1 denotes LOO-CV')
    parser.add_argument('--shap_eval', '-sh', type=bool, default=True,
                        help='if true, shap values will be evaluated. Disabled by default, since it increases runtime a lot.')
    parser.add_argument('--test_fraction', '-t', type=float, default=0.2,
                        help='size of the test set in fraction of total samples')
    parser.add_argument('--artificial_balancing_option', '-b', type=str, default='class_weight',
                        choices=['class_weight', 'random_oversampling', 'SMOTE', 'ADASYN', 'none'],
                        help='Artificial oversampling option.')
    parser.add_argument('--drop_missing_value', '-dr', type=float, default=0,
                        help='Drop rows with x% of columns having missing values')
    parser.add_argument('--missing_threshold', '-mt', type=float, default=0,
                        help='Threshold for dropping columns with missing values')
    parser.add_argument('--correlation_threshold', '-ct', type=float, default=0,
                        help='Threshold for dropping columns with high correlation')
    parser.add_argument('--data_exploration', '-ex', nargs='*', type=bool, default=False,
                        help='If true, an html file will be generated showing statistics of the parsed dataset')
    parser.add_argument('--seeds', '-s', type=int, default=[000, 111, 222, 333, 444], nargs='*',
                        help='List of seeds for reproducibility, also determines individual repetitions')
    parser.add_argument('--cores', '-c', type=int, default=8,
                        help='Number of cores to use for parallel processing')
    parser.add_argument('--offset', '-of', type=int, default=None,
                        help='Time offset')
    parser.add_argument('--endpoints', '-ep', type=str, nargs='*', default=None,
                        help='Endpoints to evaluate, if not all')
    return parser


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    log_format = "%(asctime)s - %(levelname)s - %(name)s : %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format, force=True)
    logging.info('Starting evaluation...')
    arg_parser = get_parser()
    args = arg_parser.parse_args()
    logging.debug(args)
    evaluation(**vars(args))

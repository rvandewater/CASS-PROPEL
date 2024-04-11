import catboost
from hyperopt import hp
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from pyod.models.xgbod import XGBOD
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
common_hyperparameters = {
    "estimators":[50, 100, 250, 500, 750],
    "scale_pos_weight":[1, 5, 10, 25, 50, 75, 99, 100, 1000],
    'learning_rate': [0.005, 0.01, 0.1, 0.3, 0.5, 0.7, 1],
    'max_depth': [3, 5, 10, 15]
}
def get_classification_model_grid(class_weighting=None, seed=42):
    return [  # (DummyClassifier(strategy='constant', constant=1, random_state=seed), {}),
        # (DecisionTreeClassifier(class_weight=class_weighting, random_state=seed),
        #  {'max_depth': [5, 10, None],
        #   'max_features': ['sqrt', 'log2', None],
        #   'criterion': ['gini', 'entropy'],
        #   'splitter': ['best', 'random']}),
        # (RandomForestClassifier(class_weight=class_weighting, random_state=seed),
        #  {'n_estimators': [10, 100],
        #   'criterion': ['gini', 'entropy'],
        #   'max_depth': [5, 10, None],
        #   'max_features': ['sqrt', 'log2', None]}),
        # (BalancedRandomForestClassifier(random_state=seed, n_jobs=-1),
        #     {'n_estimators': common_hyperparameters['estimators'],
        #     'max_depth': common_hyperparameters['max_depth'],
        #     'max_features': ['sqrt', 'log2', None]}),
        # (GradientBoostingClassifier(random_state=seed),
        #  {'loss': ['log_loss', 'exponential'],
        #   'learning_rate': [0.01, 0.1, 0.3],
        #   'n_estimators': [100, 1000],
        #   'max_depth': [3, 5, 10],
        #   'max_features': ['sqrt', 'log2', None]}),
        (LogisticRegression(max_iter=2500, class_weight='balanced',
                            random_state=seed, n_jobs=-1),
         {  'solver': ['lbfgs', 'saga'],
            'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
            'C': [0.1, 0.5, 1.0, 5.0, 10, 100, 500,1000]})
            #'class_weight': common_hyperparameters['scale_pos_weight']}),
        # (CatBoostClassifier(random_state=seed, verbose=False), {
        #     'learning_rate': common_hyperparameters['learning_rate'],
        #     'num_trees': common_hyperparameters['estimators'],
        #     'depth': common_hyperparameters['max_depth'],
        #     'l2_leaf_reg': [1, 3, 5, 7, 9],
        #     'border_count': [32, 5, 10, 20, 50, 100, 200],
        #     'scale_pos_weight': common_hyperparameters['scale_pos_weight']
        # }),
        # (LGBMClassifier(random_state=seed,n_jobs=-1, ),{
        #  # 'boosting_type':['gbdt', 'dart', 'goss'],
        #  'num_leaves':[10,25,50,75,100],
        #  # 'n_estimators':[100, 200, 300],
        #  'subsample':[0.3,0.5, 0.7, 1.0],
        #  'feature_fraction':[0.3, 0.5, 1.0],
        #  # 'reg_alpha':[0.0, 0.1, 0.5],
        #  # 'reg_lambda':[0.0, 0.1, 0.5],
        #  'max_depth':[3,5,10,15,20],
        #  'bagging_fraction':[0.3,0.5,0.7,1.0],
        #  'min_data_in_leaf':[100,300,500,750,1000,2000],
        #  'scale_pos_weight': [1, 10, 25, 50, 75, 99, 100, 1000]
        # })
        # (XGBClassifier(random_state=seed, importance_type='gain'),
        #  {'learning_rate': common_hyperparameters['learning_rate'],
        #   'colsample_bytree': [0.1, 0.25, 0.5, 0.75, 1.0],
        #   'n_estimators': common_hyperparameters['estimators'],
        #   'min_child_weight': [1, 0.5],
        #   'max_depth': common_hyperparameters['max_depth'],
        #   'scale_pos_weight': common_hyperparameters['scale_pos_weight'],
        #   'max_delta_step': [0, 1, 2, 3, 4, 5,10]})
        # (XGBOD(random_state=seed, importance_type='gain'),{})
        # {'learning_rate': [0.005, 0.01, 0.1, 0.3, 0.5, 0.7, 1],
        #  'colsample_bytree': [0.1, 0.25, 0.5, 0.75, 1.0],
        #  'n_estimators': [50, 100, 250, 500, 750],
        #  'min_child_weight': [1, 0.5],
        #  'max_depth': [3, 5, 10, 15],
        #  'scale_pos_weight': [1, 10, 25, 50, 75, 99, 100, 1000]})
        # (SVC(class_weight=class_weighting, probability=True, random_state=seed),
        # 'max_features': ['sqrt', 'log2', None]}),
        # (SVC(class_weight=class_weighting, probability=True, random_state=seed),
        #  {'C': [0.1, 0.5, 1.0, 10.0, 100.0],
        #   'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}),
        # (LinearSVC(max_iter=10000, class_weight=class_weighting, random_state=seed),
        #  {'penalty': ['l1', 'l2'],
        #   'dual': [False, True],
        #   'loss': ['hinge', 'squared_hinge'],
        #   'C': [0.1, 0.5, 1.0, 10.0, 100.0],})
        # (MLPClassifier(max_iter=500, solver='sgd', random_state=seed),
        #  {'hidden_layer_sizes': [(128, 64), (64, 32), (32, 16), (32, 32), (64,), (32,), (16,)],
        #   'activation': ['relu', 'logistic', 'identity'],
        #   'momentum': [0.0, 0.9],
        #   'batch_size': [8, 32, 'auto'],
        #   'learning_rate_init': [0.1, 0.05, 0.01, 0.005, 0.001]})
        # (XGBClassifier(random_state=seed),
        #  {'loss': ['log_loss', 'exponential'],
        #   'learning_rate': [0.01, 0.1, 0.3],
        #   'n_estimators': [100, 1000],
        #   'max_depth': [3, 5, 10],
        #   'max_features': ['sqrt', 'log2', None]}),
    ]

tree_models = ['DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier',
               'BalancedRandomForestClassifier', 'CatBoostClassifier', 'LGBMClassifier']
svc_models = ['SVC', 'LinearSVC']
sklearn_models= ['LogisticRegression', 'MLPClassifier']
all_models = tree_models + svc_models + sklearn_models
def positive_class_probability(model, X):
    model_name = str(model.__class__.__name__)
    if model_name == 'Pipeline':
        model_name = str(model.named_steps['model'].__class__.__name__)
    if model_name in tree_models+sklearn_models:
        return model.predict_proba(X)[:, 1]
    if model_name in svc_models:
        return model.decision_function(X)
    raise TypeError(f'Model name {model_name} not found')


def get_feature_importance(trained_model):
    model_name = str(trained_model.__class__.__name__)
    if model_name == 'Pipeline':
        trained_model = trained_model.named_steps['model']
        model_name = str(trained_model.__class__.__name__)
    if model_name in tree_models:
        return trained_model.feature_importances_.flatten()
    # if model_name in ['XGBClassifier']:
    #     return trained_model.get_score(importance_type='gain').flatten()
    #     return
    if model_name in ['LogisticRegression', 'LinearSVC'] or (model_name == 'SVC' and trained_model.kernel == 'linear'):
        return trained_model.coef_.flatten()
    return None

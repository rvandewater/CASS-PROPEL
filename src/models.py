from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_classification_model_grid(class_weighting=None, seed=42):
    return [(DummyClassifier(strategy='constant', constant=1, random_state=seed), {}),
            (DecisionTreeClassifier(class_weight=class_weighting, random_state=seed),
             {'max_depth': [5, 10, None],
              'max_features': ['sqrt', 'log2', None],
              'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random']}),
            (RandomForestClassifier(class_weight=class_weighting, random_state=seed),
             {'n_estimators': [10, 100],
              'criterion': ['gini', 'entropy'],
              'max_depth': [5, 10, None],
              'max_features': ['sqrt', 'log2', None]}),
            (GradientBoostingClassifier(random_state=seed),
             {'loss': ['log_loss', 'exponential'],
              'learning_rate': [0.01, 0.1, 0.3],
              'n_estimators': [100, 1000],
              'max_depth': [3, 5, 10],
              'max_features': ['sqrt', 'log2', None]}),
            (LogisticRegression(solver='saga', max_iter=5000, class_weight=class_weighting, penalty='elasticnet',
                                random_state=seed),
             {'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
              'C': [0.1, 0.5, 1.0, 5.0]}),
            # (XGBClassifier(random_state=seed),
            #  {'loss': ['log_loss', 'exponential'],
            #   'learning_rate': [0.01, 0.1, 0.3],
            #   'n_estimators': [100, 1000],
            #   'max_depth': [3, 5, 10],
            #   'max_features': ['sqrt', 'log2', None]}),
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


def positive_class_probability(model, X):
    model_name = str(model.__class__.__name__)
    if model_name == 'Pipeline':
        model_name = str(model.named_steps['model'].__class__.__name__)
    if model_name in ['DummyClassifier', 'LogisticRegression', 'MLPClassifier',
                      'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']:
        return model.predict_proba(X)[:, 1]
    if model_name in ['SVC', 'LinearSVC']:
        return model.decision_function(X)
    raise TypeError('Model name not found')


def get_feature_importance(trained_model):
    model_name = str(trained_model.__class__.__name__)
    if model_name == 'Pipeline':
        trained_model = trained_model.named_steps['model']
        model_name = str(trained_model.__class__.__name__)
    if model_name in ['DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']:
        return trained_model.feature_importances_.flatten()
    if model_name in ['LogisticRegression', 'LinearSVC'] or (model_name == 'SVC' and trained_model.kernel == 'linear'):
        return trained_model.coef_.flatten()
    return None

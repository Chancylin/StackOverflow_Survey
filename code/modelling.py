import numpy as np
import pandas as pd
import category_encoders as ce

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, roc_auc_score, precision_score, recall_score, confusion_matrix

import xgboost as xgb


random_seed = 123

NUMERIC_VARS_WITH_NA = ['Age', 'Age1stCode', 'ConvertedComp',
                        'WorkWeekHrs', 'YearsCode', 'YearsCodePro']

CATEGORICAL_VARS_WITH_NA = ['NEWOnboardGood', 'UndergradMajor',
                            'NEWEdImpt',  'EdLevel', 'OrgSize',
                            'SOPartFreq', 'NEWOvertime']

CATEGORICAL_VARS_ORDER = ['NEWEdImpt',  'EdLevel', 'OrgSize',  'SOPartFreq', 'NEWOvertime']

CATEGORICAL_VARS_OHENC = ['NEWOnboardGood', 'UndergradMajor']

NEWEdImpt_map = {'Not at all important/not necessary': 0,
                 'Somewhat important': 1,
                 'Fairly important': 2,
                 'Very important': 3,
                 'Critically important': 4,
                 'Missing': -1}

EdLevel_map = {'I never completed any formal education': 0,
               'Primary/elementary school': 1,
                'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 2,
                'Some college/university study without earning a degree': 3,
               'Associate degree (A.A., A.S., etc.)': 4,
               'Bachelor’s degree (B.A., B.S., B.Eng., etc.)': 5,
               'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)': 6,
               'Professional degree (JD, MD, etc.)': 7,
               'Other doctoral degree (Ph.D., Ed.D., etc.)': 8,
               'Missing': -1}

OrgSize_map = {'Just me - I am a freelancer, sole proprietor, etc.': 1,
               '2 to 9 employees': 2,
               '10 to 19 employees': 3,
               '20 to 99 employees': 4,
               '100 to 499 employees': 5,
               '500 to 999 employees': 6,
               '1,000 to 4,999 employees': 7,
               '5,000 to 9,999 employees': 8,
               '10,000 or more employees': 9,
               'Missing': -1
               }

SOPartFreq_map = {'I have never participated in Q&A on Stack Overflow': 0,
                  'Less than once per month or monthly': 2,
                  'A few times per month or weekly': 3,
                  'A few times per week': 4,
                  'Daily or almost daily': 5,
                  'Multiple times per day': 6,
                  'Missing': -1
                  }

NEWOvertime_map = {'Never': 0,
                   'Rarely: 1-2 days per year or less': 1,
                   'Occasionally: 1-2 days per quarter but less than monthly': 2,
                   'Sometimes: 1-2 days per month but less than weekly': 3,
                   'Often: 1-2 days per week or more': 4,
                   'Missing': -1}


features_order_map = [{'col': 'NEWEdImpt', 'mapping': NEWEdImpt_map},
                      {'col': 'EdLevel', 'mapping': EdLevel_map},
                      {'col': 'OrgSize', 'mapping': OrgSize_map},
                      {'col': 'SOPartFreq', 'mapping': SOPartFreq_map},
                      {'col': 'NEWOvertime', 'mapping': NEWOvertime_map}]


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer to drop unnecessary features"""
    def __init__(self, variables_to_drop=None):
        
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical value imputer"""
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}

        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical missing value imputer"""
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X


def train_valid_stratify_split(X, y, random_seed=123):
    """stratified train_valid split, and return the test fold (0 or -1) for PredefinedSplit"""
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_seed)
    train_valid = pd.DataFrame(index=y.index)
    train_valid['train_valid'] = -1
    train_valid.loc[train_valid.index.isin(y_valid.index), 'train_valid'] = 0
    return train_valid


def get_performance(model, X, ground_y):
    """Calculate some importance metrics for model evaluation: roc_auc_ovr, accuracy, precision_macro, recall_macro,
    confusion matrix"""
    ground_y = np.squeeze(ground_y)

    predict_y = model.predict(X)
    predict_y_proba = model.predict_proba(X)

    roc_auc_score_perf = roc_auc_score(ground_y, predict_y_proba, average='macro', multi_class='ovr')  # ROC-AUC
    logLoss_perf = log_loss(ground_y, predict_y_proba)

    accuracy_perf = (predict_y == ground_y).sum() / len(predict_y)
    precision_score_perf = precision_score(ground_y, predict_y, average='macro')
    recall_score_perf = recall_score(ground_y, predict_y, average='macro')

    # Confusion matrix:
    # print("Confusion matrix [[TN, FP]\n[FN, TP]]:\n", confusion_matrix(ground_y, predict_y))
    conf_m = confusion_matrix(ground_y, predict_y)

    return roc_auc_score_perf, logLoss_perf, accuracy_perf, precision_score_perf, recall_score_perf, conf_m


# create xgboost classifier instance
xgb_clf = xgb.XGBClassifier(objective='multi:softmax',
                            seed=random_seed)

# pipeline for data process and feature engineer
clf_pipe = Pipeline([
    ('numeric_imputer', NumericalImputer(variables=NUMERIC_VARS_WITH_NA)),
    ('categorical_imputer', CategoricalImputer(variables=CATEGORICAL_VARS_WITH_NA)),
    ('categorical_orderEncoder', ce.OrdinalEncoder(cols=CATEGORICAL_VARS_ORDER,
                                                   mapping=features_order_map,
                                                   handle_unknown='value',
                                                   drop_invariant=False)),
    ('categorical_OHEncoder', ce.OneHotEncoder(cols=CATEGORICAL_VARS_OHENC,
                                               handle_unknown='value',
                                               drop_invariant=False)),
    ('xgb', xgb_clf)
     ])

# create GaussianNB instance
gnb_clf = GaussianNB()
# pipeline for data process and feature engineer
gnb_clf_pipe = Pipeline([
    ('numeric_imputer', NumericalImputer(variables=NUMERIC_VARS_WITH_NA)),
    ('categorical_imputer', CategoricalImputer(variables=CATEGORICAL_VARS_WITH_NA)),
    ('categorical_orderEncoder', ce.OrdinalEncoder(cols=CATEGORICAL_VARS_ORDER,
                                                   mapping=features_order_map,
                                                   handle_unknown='value',
                                                   drop_invariant=False)),
    ('categorical_OHEncoder', ce.OneHotEncoder(cols=CATEGORICAL_VARS_OHENC,
                                               handle_unknown='value',
                                               drop_invariant=False)),
    ('gnb', gnb_clf)
     ])



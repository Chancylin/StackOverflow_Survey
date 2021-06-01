import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif

def extract_data_related(df_survey):
    """Prepare the rows that have a data-related roles from the survey data"""
    df_survey['data_related'] = df_survey['DevType'].\
    str.contains('data(?![a-z])', case=False, regex=True)

    df_data = df_survey[df_survey['data_related']==True].copy()

    # replace ',' by '_' so we can split the string of DevType into a list later
    df_data['DevType_list'] = df_data['DevType'].str.replace(', ', '_').str.split(';')
    # convert list to dictionary for efficiency
    df_data['DevType_list'] = df_data['DevType_list'].apply(lambda x: set(x)) 

    # identify what data role for the surveyees:
    # 'Data scientist or machine learning specialist'
    # 'Data or business analyst' 
    # 'Engineer_data'
    df_data['DS'] = df_data['DevType_list'].apply(lambda x: 1 if 'Data scientist or machine learning specialist' in x else 0)
    df_data['DA'] = df_data['DevType_list'].apply(lambda x: 1 if 'Data or business analyst' in x else 0)
    df_data['DE'] = df_data['DevType_list'].apply(lambda x: 1 if 'Engineer_data' in x else 0)

    df_data['multiple_roles'] = df_data['DS']+df_data['DA']+df_data['DE']
    df_data['multiple_roles'] = df_data['multiple_roles'].apply(lambda x: True if x > 1 else False)

    df_data_ds = df_data[df_data['DS'] == 1].copy()
    df_data_ds['data_role'] = 'DS'

    df_data_da = df_data[df_data['DA'] == 1].copy()
    df_data_da['data_role'] = 'DA'

    df_data_de = df_data[df_data['DE'] == 1].copy()
    df_data_de['data_role'] = 'DE'

    df_data = pd.concat([df_data_ds, df_data_da, df_data_de], axis=0).drop(columns=['DS', 'DA', 'DE'])

    return df_data

def cal_basic_info(df):
    """Calculate the basic information for columns"""
    df_desc_info = df.describe().T[['mean', 'std', 'min', 'max']]
    df_nullCount_info = df.isnull().sum(axis=0).to_frame(name='null counts')
    df_dtype_info = df.dtypes.to_frame(name='dtypes')
    df_info = df_dtype_info.merge(df_nullCount_info, left_index=True, right_index=True)
    df_info['null percent'] = df_info['null counts'] / len(df) * 100
    df_info = df_info.merge(df_desc_info, how='left', left_index=True, right_index=True)
    # count the number of unique values for the non-numerical columns
    df_info['nunique'] = df_info.apply(
        lambda x: np.nan if np.issubdtype(x['dtypes'], np.number) else int(df[x.name].nunique(dropna=False)), axis=1)
    return df_info

def parse_age_year(df, variables=['Age1stCode', 'YearsCode', 'YearsCodePro']):
    """Convert age/year columns to numeric"""
    # replace_map = {'Younger than 5 years': '4', 'Older than 85': '86'}
    # replace_map = {'Less than 1 year': '0', 'More than 50 years': '51'}
    df = df.copy()
    for var in variables:
        if var == 'Age1stCode':
            df[var] = pd.to_numeric(df[var].str.\
                replace('Younger than 5 years', '4').\
                str.replace('Older than 85', '86'), downcast='integer')
        elif var == 'YearsCode' or var == 'YearsCodePro':
            df[var] = pd.to_numeric(df[var].str.\
                replace('Less than 1 year', '0').\
                str.replace('More than 50 years', '51'), downcast='integer')
        else:
            raise ValueError(f'only process {* variables,}')
    return df

def cal_mutual_info(df, target_var='loan_status', disc_features_only=True):
    df = df.copy()

    df_f_type = df.dtypes
    df_f_type = df_f_type.loc[~df_f_type.index.isin([target_var])].copy()
    cols_if_num = df_f_type.apply(lambda x: np.issubdtype(x, np.number))
    discrete_f = ~cols_if_num
    # get all categorical features
    cols_num = cols_if_num[cols_if_num].index.tolist()
    cols_cat = cols_if_num[~cols_if_num].index.tolist()

    for col_cat in cols_cat:
        df[col_cat] = df[col_cat].fillna('Missing')

    for col_num in cols_num:
        df[col_num] = df[col_num].fillna(df[col_num].mean())

    enc = OrdinalEncoder()
    df[cols_cat] = enc.fit_transform(df[cols_cat])
    enc = OrdinalEncoder()
    df.loc[:, target_var] = enc.fit_transform(df[[target_var]])


    if not disc_features_only:
        all_features = df_f_type.index.tolist()
        mutual_info = mutual_info_classif(df[all_features], df[target_var].values,
                                          discrete_features=discrete_f,
                                          n_neighbors=20,
                                          random_state=123)
        df_mutual_info = pd.DataFrame(data=zip(all_features, mutual_info), columns=['columns', 'mutual_info'])
        return df_mutual_info
    else:

        mutual_info = mutual_info_classif(df[cols_cat], df[target_var].values,
                                          discrete_features=True)
        df_mutual_info = pd.DataFrame(data=zip(cols_cat, mutual_info), columns=['columns', 'mutual_info'])
        return df_mutual_info


class StringtoListTranformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            # if nan, create an empty set instead of imputation
            X[feature] = X[feature].str.split(';').apply(lambda x: {} if x is np.nan else set(x))

        return X


class ListColumnsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.encoder_dict_ = {}
        
        for feature in self.variables:
            _ = MultiLabelBinarizer()
            _.fit(X[feature])
            self.encoder_dict_[feature] = _

        return self

    def transform(self, X):

        X = X.copy()
        for feature in self.variables:
            f_encoded = pd.DataFrame(
                self.encoder_dict_[feature].transform(X[feature]),
                columns=self.encoder_dict_[feature].classes_,
                index=X.index)

            
            X = pd.concat([X, f_encoded], axis=1).drop(columns=[feature])
        return X


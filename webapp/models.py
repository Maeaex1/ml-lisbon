import xgboost as xgb

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet

import pandas as pd

def encodeLabels(data):
    data = data.drop(['sqm_price_usage', 'sqm_price_total', 'area', 'ref', 'altitude', 'longitude'], axis=1)

    categorical_cols = ['condition', 'district', 'estate_type', 'energy_efficiency', 'no_rooms', 'no_bathroom']

    for c in categorical_cols:
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))

    data = pd.get_dummies(data, columns=categorical_cols)

    return data


def crete_data_set_ML(data):
    y = data['price']
    X = data.drop(columns=['price'], axis=1)

    X_train, y_train, X_test, y_test= train_test_split(X, y, test_size=0.1, shuffle=False)

    return X_train, y_train, X_test, y_test

def check_missing_values(all_data):
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
    return missing_data

def init_XGB(X_train, X_test, y_train, y_test):
    model = xgb.XGBRegressor()
    steps = [('scaler', RobustScaler()), ('XGB', model)]

    pipeline = Pipeline(steps)
    parameters = {'XGB__gamma':[0, 0.5], 'XGB__max_depth':[6, 8]}
    grid= GridSearchCV(pipeline, param_grid= parameters, cv=5, scoring='neg_median_absolute_error')

    grid.fit(X_train, y_train)
    score = grid.score(X_test, y_test)
    pred = grid.best_estimator_.predict(X_test)

    return score, pred, grid

def init_Ridge(X_train, X_test, y_train, y_test):
    model = Ridge()
    steps = [('scaler', RobustScaler()), ('Ridge', model)]
    pipeline = Pipeline(steps)
    parameters = {'Ridge__alpha':[0.5, 1, 1.5], 'Ridge__normalize':[False]} # Set to false as we standardize
    grid= GridSearchCV(pipeline, param_grid= parameters, cv=5, scoring='neg_median_absolute_error')

    grid.fit(X_train, y_train)
    score = grid.score(X_test, y_test)
    pred = grid.best_estimator_.predict(X_test)

    return score, pred, grid

def init_ElNet(X_train, X_test, y_train, y_test):
    model = ElasticNet()
    steps = [('scaler', RobustScaler()), ('ElNet', model)]
    pipeline = Pipeline(steps)
    parameters = {'ElNet__alpha':[0.5, 1, 1.5], 'ElNet__normalize':[False]} # Set to false as we standardize
    grid= GridSearchCV(pipeline, param_grid= parameters, cv=5, scoring='neg_median_absolute_error')

    grid.fit(X_train, y_train)
    score = grid.score(X_test, y_test)
    pred = grid.best_estimator_.predict(X_test)

    return score, pred, grid



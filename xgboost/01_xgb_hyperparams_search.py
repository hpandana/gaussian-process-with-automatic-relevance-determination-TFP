import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


X_train= np.load(r'..\data\sklearn_toy_diabetes_Xtrain.npy')
y_train= np.load(r'..\data\sklearn_toy_diabetes_ytrain.npy')

feature_names= np.load(r'..\data\sklearn_toy_diabetes_feature_names.npy')

reg= xgb.XGBRegressor()
gscv= GridSearchCV(
  reg,
  param_grid={
    'objective':['reg:squarederror'],
    'n_estimators': [50, 100, 500],
    'max_depth': [1, 2, 3, 6, 10],
    'learning_rate': [0.02, 0.05, 0.1, 0.3], #eta
    'subsample': [0.1, 0.3, 0.5, 0.8],
    'colsample_bytree': [0.4, 0.7, 1.0],                         
  },
  scoring='neg_root_mean_squared_error', 
  cv=5,
)
gscv.fit(X_train, y_train)
print(sorted(gscv.cv_results_.keys()) )
# ['mean_fit_time', 'mean_score_time', 'mean_test_score', 'param_colsample_bytree', 'param_learning_rate', 'param_max_depth', 'param_n_estimators', 'param_objective', 'param_subsample', 'params', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'std_fit_time', 'std_score_time', 'std_test_score']
print(gscv.best_params_)
# {'colsample_bytree': 1.0, 'learning_rate': 0.05, 'max_depth': 1, 'n_estimators': 100, 'objective': 'reg:squarederror', 'subsample': 0.3}
print(gscv.best_score_)
# -0.758618150006727

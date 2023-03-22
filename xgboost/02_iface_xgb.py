import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics as met

from functools import reduce
from operator import concat


X_train= np.load(r'..\data\sklearn_toy_diabetes_Xtrain.npy')
y_train_sc= np.load(r'..\data\sklearn_toy_diabetes_ytrain.npy')
X_test= np.load(r'..\data\sklearn_toy_diabetes_Xtest.npy')
y_test_sc= np.load(r'..\data\sklearn_toy_diabetes_ytest.npy')

y_train_mean= np.load(r'..\data\y_train_mean.npy')
y_train_std= np.load(r'..\data\y_train_std.npy')

feature_names= np.load(r'..\data\sklearn_toy_diabetes_feature_names.npy')

reg= xgb.XGBRegressor(
  objective='reg:squarederror',
  n_estimators= 100,
  max_depth= 1,
  learning_rate= 0.05,
  subsample= 0.3,
  colsample_bytree= 1.0,
)

reg.fit(X_train, y_train_sc)

yhat_train_sc= reg.predict(X_train)
yhat_train= yhat_train_sc * y_train_std + y_train_mean 

yhat_test_sc= reg.predict(X_test)
yhat_test= yhat_test_sc * y_train_std + y_train_mean

y_train= y_train_sc * y_train_std + y_train_mean
y_test= y_test_sc * y_train_std + y_train_mean

X = np.concatenate((X_train, X_test), axis=0)
df= pd.DataFrame(X, columns=feature_names)
df['target']= np.concatenate((y_train, y_test), axis=0)
df['traintest'] = reduce(concat, [['train']*X_train.shape[0], ['test']*X_test.shape[0] ])
df['yhat']= np.concatenate((yhat_train, yhat_test), axis=0)
df.to_csv('yhat_xgb.csv')

print('\nTrain set:')
print('\tRMSE: ',np.sqrt(met.mean_squared_error(y_train, yhat_train) ) )
print('\nTest set:')
print('\tRMSE: ',np.sqrt(met.mean_squared_error(y_test, yhat_test) ) )
# Train set:
#         RMSE:  52.56664457445779
# 
# Test set:
#         RMSE:  52.49413221118875
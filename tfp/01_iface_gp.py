import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf

from functools import reduce
from operator import concat


X_train= np.load(r'..\data\sklearn_toy_diabetes_Xtrain.npy')
y_train= np.load(r'..\data\sklearn_toy_diabetes_ytrain.npy')
X_test= np.load(r'..\data\sklearn_toy_diabetes_Xtest.npy')
y_test= np.load(r'..\data\sklearn_toy_diabetes_ytest.npy')

y_train_mean= np.load(r'..\data\y_train_mean.npy')
y_train_std= np.load(r'..\data\y_train_std.npy')

feature_names= np.load(r'..\data\sklearn_toy_diabetes_feature_names.npy')

from src.GPR import GPR
gp= GPR(X_train, y_train,
  num_results = 4000,
  num_burnin_steps = 5000,
)
ls_samples= gp.fit()
np.save('length_scale_samples.npy', ls_samples)

X = np.concatenate((X_train, X_test), axis=0)
y_samples_sc= gp.predict_samples(X, num_samples=1000)
y_samples= y_samples_sc * y_train_std + y_train_mean
np.save('y_samples.npy', y_samples)

y_q50= np.quantile(y_samples, 0.5,axis=0)
y_q1= np.quantile(y_samples, 0.1573 ,axis=0)
y_q3= np.quantile(y_samples, 0.8427 ,axis=0)
lower_err= y_q50 - y_q1
upper_err= y_q3 - y_q50

traintest= reduce(concat, [['train']*X_train.shape[0], ['test']*X_test.shape[0] ])
df= pd.DataFrame(X, columns=feature_names)
y_sc = np.concatenate((y_train, y_test ), axis=0)
y= y_sc * y_train_std + y_train_mean
df['target']= y
df['traintest'] = traintest
df['yhat_q50']= y_q50
df['yhat_lower_err']= lower_err
df['yhat_upper_err']= upper_err
df['yhat_mean']= np.mean(y_samples, axis=0)
df['yhat_std']= np.std(y_samples, axis=0)
df.to_csv('yhat_tfp.csv', index=False)
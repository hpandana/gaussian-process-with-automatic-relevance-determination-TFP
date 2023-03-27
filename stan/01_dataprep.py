import numpy as np
from cmdstanpy import write_stan_json


X_train= np.load(r'..\data\sklearn_toy_diabetes_Xtrain.npy')
y_train_sc= np.load(r'..\data\sklearn_toy_diabetes_ytrain.npy')
X_test= np.load(r'..\data\sklearn_toy_diabetes_Xtest.npy')
# y_test_sc= np.load(r'..\data\sklearn_toy_diabetes_ytest.npy')

X = np.concatenate((X_train, X_test), axis=0)

(N1, D) = X_train.shape
N2 = X.shape[0]

data= {
  'D': D,
  'N1': N1,
  'x1': X_train,
  'y1': y_train_sc,
  'N2': N2,
  'x2': X,
}
write_stan_json('./gp_ard.data.json', data)
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


dt= datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(dt.data, dt.target, test_size=0.3)

y_train_mean= np.mean(y_train)
y_train_std= np.std(y_train)

y_train_sc= (y_train - y_train_mean)/y_train_std
y_test_sc= (y_test - y_train_mean)/y_train_std

np.save("sklearn_toy_diabetes_feature_names.npy", dt.feature_names)
np.save("sklearn_toy_diabetes_Xtrain.npy", X_train)
np.save("sklearn_toy_diabetes_ytrain.npy", y_train_sc)
np.save("sklearn_toy_diabetes_Xtest.npy", X_test)
np.save("sklearn_toy_diabetes_ytest.npy", y_test_sc)
np.save("y_train_mean.npy", y_train_mean)
np.save("y_train_std.npy", y_train_std)
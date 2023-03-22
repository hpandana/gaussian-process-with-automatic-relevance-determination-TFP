import numpy as np
import pandas as pd
from sklearn import metrics as met


df= pd.read_csv('yhat_tfp.csv')
df_train= df.loc[df['traintest'] == 'train']
df_test= df.loc[df['traintest'] == 'test']

print('\nTrain set:')
print('\tRMSE: ',np.sqrt(met.mean_squared_error(df_train['target'], df_train['yhat_mean']) ) )
print('\nTest set:')
print('\tRMSE: ',np.sqrt(met.mean_squared_error(df_test['target'], df_test['yhat_mean']) ) )
# Train set:
#         RMSE:  54.82915792880878

# Test set:
#         RMSE:  51.05445473559019

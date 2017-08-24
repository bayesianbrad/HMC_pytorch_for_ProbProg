import pandas as pd
import numpy as np

### Load and clean the SECOM data.
secom = pd.read_table('../data_and_posterior/secom_features.txt', sep='\s+', header=None)
y = pd.read_table('../data_and_posterior/secom_outcome.txt', sep='\s+', header=None)[0]

# Remove predictors with too many na's
max_na_pred = 20
index_many_na = np.where(secom.isnull().sum(axis=0) > max_na_pred)[0]
secom = secom.drop(index_many_na, axis=1) 
print('{:d} features were dropped due to a large number of NA\'s.'.format(index_many_na.size))

# Remove incomplete cases
index_drop = np.where(secom.isnull().any(axis = 1))[0]
secom = secom.drop(index_drop, axis=0)
y = y.drop(index_drop)

X = secom.as_matrix()
print('Removing additional {:d} features for identifiability.'.format(np.sum(np.var(X, 0) == 0)))
X = X[:, np.var(X, 0) > 0]
X = (X - np.mean(X, 0)) / np.std(X, 0)
X = np.hstack((np.ones((X.shape[0], 1)), X)) # Intercept
y = y.as_matrix().astype('float')

n_param = X.shape[1]
n_disc = n_param # None of the conditional densities is smooth.



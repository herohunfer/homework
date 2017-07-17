from sklearn.preprocessing import Normalizer
from pandas import read_csv
from numpy import set_printoptions
import numpy as np

# -- Normalize using sklearn--L1 norm -- #
print('\nNormalize using sklearn--L1 norm')
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
array = data_frame.values

# Separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]

# L1 norm
scaler = Normalizer(norm='l1').fit(X)
normalizedX = scaler.transform(X)

# Summarize transformed data
set_printoptions(precision=3)
print(normalizedX[0:5, :])


# -- Normalize from scratch--L1 norm -- #
# L1 norm
print('\nNormalize from scratch--L1 norm')
norms = np.abs(X).sum(axis=1)

X_normalized = X / norms[:, np.newaxis]

print(X_normalized[0:5, :])

# -- Normalize using sklearn--L2 norm -- #
# L2 norm
print('\nNormalize using sklearn--L2 norm')
scaler = Normalizer(norm='l2').fit(X)
normalizedX = scaler.transform(X)

# Summarize transformed data
set_printoptions(precision=3)
print(normalizedX[0:5, :])

# -- Normalize from scratch--L2 norm -- #
# L2 norm
print('\nNormalize from scratch--L2 norm')
norms = np.einsum('ij,ij->i', X, X)
np.sqrt(norms, norms)
X_normalized = X / norms[:, np.newaxis]
print(X_normalized[0:5, :])




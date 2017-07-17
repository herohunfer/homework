from pandas import read_csv
import numpy as np
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

print('\nRescale using sklearn')
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
array = data_frame.values

print(array)
# Separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

# Summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5, :])


# -- Rescale from scratch -- #
print('\nRescale from scratch')
# -- Rescale from scratch -- #
# Define scale range min and max
MIN = 0
MAX = 1

# Calculate X min and max for each attribute
# axis=0 calculates along each column
X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)

# Calculate X_std and X_scaled
X_std = (X - X_min) / (X_max - X_min)
X_scaled = X_std * (MAX - MIN) + MIN

print(X_scaled[0:5, :])
print('\n Simpler Way')

# -- Simpler way to do it -- #
X_std_2 = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled_2 = X_std_2 * (MAX - MIN) + MIN

print(X_scaled_2[0:5, :])


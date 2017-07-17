from sklearn.preprocessing import Binarizer
from pandas import read_csv
from numpy import set_printoptions
import numpy as np

# -- Binarize using sklearn -- #
print('\nBinarize using sklearn')
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
array = data_frame.values

# Separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]

binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)

# Summarize transformed data
set_printoptions(precision=3)
print(binaryX[0:5, :])

# -- Binarize from scratch -- #
print('\nBinarize from scratch')
threshold = 0.0
cond = X > threshold
not_cond = np.logical_not(cond)
X_binarized = X
X_binarized[cond] = 1
X_binarized[not_cond] = 0

print(X_binarized[0:5, :])


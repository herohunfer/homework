from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import set_printoptions

print('\nStandardize using sklearn')
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
array = data_frame.values

# Separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]
scaler = StandardScaler().fit(X)
standardizedX = scaler.transform(X)

# Summarize transformed data
set_printoptions(precision=3)
print(standardizedX[0:5, :])


# -- Standardize from scratch -- #
print('\nStandardize from scratch')
# -- Standardize from scratch -- #
# Calculate data mean and standard deviation
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

# The values for each attribute now have a mean of 0 and standard deviation of 1
X_scaled = (X - X_mean) / X_std

print(X_scaled[0:5, :])




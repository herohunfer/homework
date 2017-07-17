# coding=utf-8
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

# 构造训练数据
'''
x = np.arange(0., 10., 0.2)
m = len(x)  # 训练数据点数目
x0 = np.full(m, 1.0)
#print(x0)
input_data = np.vstack([x0, x]).T  # 将偏置b作为权向量的第一个分量
#print(input_data)
target_data = 2 * x + 5 + np.random.randn(m)
#print(target_data)
'''
print('\nRescale using sklearn')
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_frame = read_csv(filename, names=names)
#print(data_frame)
array = data_frame.values

# Separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

input_data = rescaledX
print(input_data.shape)
target_data = Y
print(target_data)
m = input_data.shape[0]

x = np.arange(768)
# 两种终止条件
loop_max = 10000  # 最大迭代次数(防止死循环)
epsilon = 1e-3

# 初始化权值
np.random.seed(0)
w = np.random.randn(input_data.shape[1])

alpha = 0.001  # 步长(注意取值过大会导致振荡,过小收敛速度变慢)
diff = 0.
error = np.zeros(input_data.shape[1])
count = 0  # 循环次数
finish = 0  # 终止标志
error_list = []
batch_size = 128

# -------------------------------------------随机梯度下降算法----------------------------------------------------------
while count < loop_max:
    count += 1

    for i in range(0, m, batch_size):
        sum_batch = np.zeros(input_data.shape[1])
        for j in range(i, i+batch_size):
            diff = (np.dot(w, input_data[j]) - target_data[j]) * input_data[j]
            sum_batch = sum_batch + diff
        w = w - alpha * sum_batch / batch_size
        error_list.append(np.sum(sum_batch/batch_size)**2)
        # ------------------------------终止条件判断-----------------------------------------
        # 若没终止，则继续读取样本进行处理，如果所有样本都读取完毕了,则循环重新从头开始读取样本进行处理。
        if np.linalg.norm(w - error) < epsilon:  # 终止条件：前后两次计算出的权向量的绝对误差充分小
            finish = 1
            break
        else:
            error = w
    # ----------------------------------终止条件判断-----------------------------------------
    # 注意：有多种迭代终止条件，和判断语句的位置。终止判断可以放在权值向量更新一次后,也可以放在更新m次后。

print('loop count = %d' % count,  '\tw:[%f, %f]' % (w[0], w[1]))

# ----------------------------------------------------------------------------------------------------------------------


# check with scipy linear regression
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, target_data)
print('intercept = %s slope = %s' % (intercept, slope))

plt.plot(range(len(error_list[0:10000])), error_list[0:10000])
plt.show()

'''
plt.plot(x, target_data, 'k+')
plt.plot(x, w*input_data, 'r')
plt.show()
'''
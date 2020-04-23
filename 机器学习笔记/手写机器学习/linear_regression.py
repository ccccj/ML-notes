import numpy as np
import pandas as pd

NUM = 784

def read():
    df = pd.read_csv('./MNIST_data/train.csv')
    sample = pd.read_csv('./MNIST_data/test.csv')
    arr1 = np.array(df)
    arr2 = np.array(sample)
    line = int(arr1.shape[0] * 0.8)
    return arr1[0 : line], arr1[line : ], arr2

def slove_data(data):
    length = data.shape[0]
    data = data.astype(np.float64)
    label = data[:, 0]
    data = data[:, 1 : NUM + 1] / 255
    x1 = np.ones((length, 1))
    data = np.concatenate([x1, data], axis = 1)
    return data, label

def check(test, label, Q):
    length = test.shape[0]
    Y = np.matmul(test, Q)
    sum = 0.0
    for i in range(length):
        if abs(Y[i] - label[i]) < 0.5 or abs(Y[i] - label[i]) == 0.5:
            sum = sum + 1
    print (sum / length)

def gradient_descent(data, label, test, label_test, a):
    Q = np.zeros((NUM + 1, 1))
    length = data.shape[0]

    # 目前有784(NUM)个特征，需要拟合的函数有785个参数：θ0、θ1...θ784,默认全部初始为0
    # fun = θ0 + θ1·x1 + θ2·x2 + ... + θ784·x784
    # cost = 1/2m * (X·θ - Y)^2
    # 使用梯度下降法使得 cost 减小
    # θ = θ - α·dcost / dθ (偏导) = θ - α/m·∑(cost - Y)·x

    for i in range(10000):
        cost = np.matmul(data, Q) - np.transpose([label])
        ####################################################
        if i % 100 == 0:
            print('*************** i = ' + str(i))
            check(test, label_test, Q)
            print(np.matmul(np.transpose(cost), cost))
        ####################################################
        cost = np.true_divide(cost, length)
        Q = Q - np.matmul(np.transpose(data), cost) * a
    return Q

def main():
    train, test, sample = read()
    train, label_train = slove_data(train)
    test, label_test = slove_data(test)
    Q = gradient_descent(train, label_train, test, label_test, 0.001)

main()







#

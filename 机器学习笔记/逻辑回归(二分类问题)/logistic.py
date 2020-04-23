import numpy as np
import pandas as pd
import math

NUM = 784

def read():
    df = pd.read_csv('./MNIST_data/train.csv')
    sample = pd.read_csv('./MNIST_data/test.csv')
    arr1 = np.array(df)
    arr2 = np.array(sample)
    line = int(arr1.shape[0] * 1)
    return arr1[0 : line], arr1[line : ], arr2

def slove_data(data):
    length = data.shape[0]
    label = np.zeros([length, 10])
    for i in range(length):
        label[i][data[i][0]] = 1
    data = data.astype(np.float64)
    label = label.astype(np.float64)
    data = data[:, 1 : NUM + 1] / 255
    x1 = np.ones((length, 1))
    data = np.concatenate([x1, data], axis = 1)
    return data, label

def h(data, Q):
    return np.matmul(data, Q)

def sigmoid(data, Q):
    # fun = g(QX) = g(θ0 + θ1·x1 + θ2·x2 + ... + θ784·x784)
    # g = 1 / [1 + e ^ (-QX)]
    return 1 / (1 + np.exp(-h(data, Q)))

def check(test, label, Q):
    length = test.shape[0]
    Y = np.argmax(sigmoid(test, Q), axis = 1)
    sum = 0.0
    #print(Y.shape)
    #print(label.shape)
    for i in range(length):
        if Y[i] == np.argmax(label[i]):
            sum = sum + 1
    print (sum / length)

def logistic(data, label, test, label_test, a, λ):
    Q = np.zeros((NUM + 1, 10))
    length = data.shape[0]

    # 目前有784(NUM)个特征，需要拟合的函数有785个参数：θ0、θ1...θ784,默认全部初始为0
    # fun = g(QX) = g(θ0 + θ1·x1 + θ2·x2 + ... + θ784·x784)
    # g = 1 / [1 + e ^ (-QX)]
    # cost = 1/m · ∑ [y·(-logfun) - (1-y)·log(1-fun)]
    # cost = -1/m · ∑ [y·logfun + (1-y)·log(1-fun)]
    # 使用梯度下降法使得 cost 减小
    # θ = θ - α·dcost / dθ (偏导) = θ - α/m·∑(fun - Y)·x

    for i in range(3000):
        fun = sigmoid(data, Q)
        Q = Q - (np.matmul(np.transpose(data), fun - label) + Q * λ) * a / length
        ####################################################
        if i % 100 == 0:
            print('*************** i = ' + str(i))
            #check(test, label_test, Q)
            sample = read_sample()
            write(sample, Q)
            #print(np.matmul(np.transpose(cost), cost))
        ####################################################
    return Q

def read_sample():
    df = pd.read_csv('./MNIST_data/test.csv')
    data = np.array(df)
    data = data.astype(np.float64) / 255
    length = data.shape[0]
    x0 = np.ones([length, 1])
    data = np.concatenate([x0, data], axis = 1)
    return data

def write(data, Q):
    Y = np.argmax(sigmoid(data, Q), axis = 1)
    df = pd.read_csv('./MNIST_data/sample_submission.csv')
    df['Label'] = Y
    df.to_csv('./MNIST_data/sampleSubmission1.csv', index = 0, header = 1)

def main():
    train, test, sample = read()
    train, label_train = slove_data(train)
    test, label_test = slove_data(test)

    Q = logistic(train, label_train, test, label_test, 1, 1)
main()







#

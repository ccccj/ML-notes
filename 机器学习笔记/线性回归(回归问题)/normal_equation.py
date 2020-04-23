import numpy as np
import pandas as pd

NUM = 8

def slove(data, label):
    length = data.shape[0]
    x0 = np.ones([length, 1])
    data = np.concatenate([x0, data], axis = 1)
    length = int(length * 1)
    return data[0:length], data[length:], label[0:length], label[length:]

def read(col):
    df = pd.read_csv('./data/train.csv')
    data = np.array(df)
    label = data[:, col]
    data = data[:, 1 : NUM + 1]
    data = data.astype(np.float64)
    label = label.astype(np.float64)
    MAX = np.max(data, axis = 0)
    for j in range(data.shape[1]):
        data[:, j] = data[:, j] / MAX[j]
    return slove(data, label)

def check(data, label, Q):
    length = data.shape[0]
    Y = np.matmul(data, Q)
    label = np.transpose([label])
    cost = Y - label
    for i in range(length):
        print('==========' + str(Y[i]) + ' / ' + str(label[i]))
    print('cost = ' + str(np.matmul(np.transpose(cost), cost) / (2 * length)))


def linear_regression(train, label_train):
    Q = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(train), train)), np.transpose(train)), label_train)
    return Q

def read_sample():
    df = pd.read_csv('./data/test.csv')
    data = np.array(df)
    data = data[:, 1 : NUM + 1]
    data = data.astype(np.float64)
    MAX = np.max(data, axis = 0)
    for j in range(data.shape[1]):
        data[:, j] = data[:, j] / MAX[j]
    length = data.shape[0]
    x0 = np.ones([length, 1])
    data = np.concatenate([x0, data], axis = 1)
    return data

def write(data, Q1, Q2):
    #yyyy-mm-dd hh：mm：ss
    Y1 = np.matmul(data, Q1)
    Y2 = np.matmul(data, Q2)
    for i in range(Y1.shape[0]):
        #Y[i] = int(Y[i])
        if Y1[i] < 0:
            Y1[i] = 0
        if Y2[i] < 0:
            Y2[i] = 0
        Y1[i] = Y1[i] + Y2[i]
    df = pd.read_csv('./data/sampleSubmission.csv')
    df['count'] = Y1
    df.to_csv('./data/sampleSubmission1.csv', index = 0, header = 1)

def main():
    train, test, label_train, label_test = read(9)
    Q1 = linear_regression(train, label_train)
    train, test, label_train, label_test = read(10)
    Q2 = linear_regression(train, label_train)
    data = read_sample()
    write(data, Q1, Q2)

main()

#

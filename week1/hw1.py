'''
preprocessing
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
data = pd.read_csv('./train.csv', encoding = 'big5')
data[data=='NR'] = 0
data.iloc[:,3:]=data.iloc[:,3:].astype('float32')
# calculate mean and std
def mean_and_std(key):
    # calculate mean and std via ‘測項’
    # print(key)
    # print(data[data['測項']==key])
    t = data[data['測項']==key].iloc[:,3:].values
    mean = np.mean(t)
    std = np.std(t)
    # print(mean,std)
    return mean,std
m_a_s = [mean_and_std(x) for x in data['測項'][0:18]]
data = data.iloc[:,3:]
data.iloc[:18,:]
'''
extract feature and normalize
'''
raw_data=[]
idx = 0
for i in range(12):
    t = np.empty([18,24*20])
    for j in range(20):
        # print(idx)
        t[:,j*24:(j+1)*24] = data.iloc[idx:idx+18,:]
        idx+=18
    raw_data.append(t)
# nomalize
for i in range(12):
    for j in range(18):
        raw_data[i][j,:]=(raw_data[i][j,:]-m_a_s[j][0])/m_a_s[j][1]
# sampling and reshape

x = np.empty([12*471,18*9])
y = np.empty([12*471,1])
idx = 0
for i in range(12):
    for j in range(471):
        # print(idx)
        x[idx,:] = np.reshape(raw_data[i][:,j:j+9],-1)
        # print(x[idx:idx+18])
        y[idx] = raw_data[i][9,j+9]
        idx+=1
        # print(raw_data[i][:,j:j+9].shape)
        # print(np.reshape(raw_data[i][:,j:j+9],-1).shape)

y = y * m_a_s[9][1]+m_a_s[9][0]

# random split data
sr = 0.8
n = len(y)
bound = int(np.floor(sr*n))
indexes = np.arange(n)
np.random.shuffle(indexes)
train_x = x[indexes[:bound],:]
train_y = y[indexes[:bound]]
val_x = x[indexes[bound:],:]
val_y = y[indexes[bound:]]

'''
model and training
'''
# x = train_x # x -> train_x, y -> train_y
# y = train_y
dim = 18 * 9 + 1
x = np.concatenate((np.ones([x.shape[0], 1]), x), axis = 1).astype(float)
val_x = np.concatenate((np.ones([val_x.shape[0], 1]), val_x), axis = 1).astype(float)
# w = numpy.random([dim, 1])*0.1
w = np.zeros([dim,1])
learning_rate = 0.01
iter_time = 2000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
# visualize
train_log = []
val_log=[]
# training
for t in range(iter_time):
    # t = 1
    loss = np.sum(np.power(np.dot(x, w) - y, 2))/y.shape[0]#mse
    val_loss = np.sum(np.power(np.dot(val_x, w) - val_y, 2))/val_y.shape[0]
    train_log.append(loss)
    val_log.append(val_loss)
    if t%100==0:
        print(f"Epoch {t}:  train_loss: {loss}  val_loss: {val_loss}")
    # if t in [54,55]:
    #     print(w)
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)/y.shape[0] #dim*1

    adagrad += gradient ** 2
    w = w - learning_rate * gradient
    # w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
# plt.ylim([5.5,10])
plt.plot(np.sqrt(train_log),label='loss')
plt.plot(np.sqrt(val_log),label='val')
plt.legend()
plt.show()

'''
tuning
'''
# tuning learning rate
from matplotlib import pyplot as plt
train_logs = []
val_logs=[]
lambdaa = 1
learning_rates = [0.1,0.3,1,3,10,30,90]
iter_time = 4000
for learning_rate in learning_rates:
    adagrad = np.zeros([dim, 1])
    w = np.zeros([dim,1])
    # visualize
    train_log = []
    val_log=[]
    # training
    for t in range(iter_time):
        # regularization
        wt = w[1:]
        reg = (np.dot(np.transpose(wt),wt) * lambdaa /2/wt.shape[0])[0,0]
        grad_reg = lambdaa * wt / wt.shape[0]
        grad_reg = np.concatenate((np.zeros([1,1]),grad_reg),axis=0)
        loss = np.sum(np.power(np.dot(x, w) - y, 2))/y.shape[0] + reg # mse
        val_loss = np.sum(np.power(np.dot(val_x, w) - val_y, 2))/val_y.shape[0] + reg
        train_log.append(loss)
        val_log.append(val_loss)
        if t%100==0:
            print(f"Epoch {t}:  train_loss: {loss}  val_loss: {val_loss}")
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)/y.shape[0] + grad_reg   #dim*1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    train_logs.append(train_log)
    val_logs.append(val_log)
    plt.ylim([0,100])
    plt.plot(train_log,label=f'{learning_rate}')
    plt.legend()
    plt.show()

plt.ylim([5.5,10])
plt.xlim([0,2000])
for train_log,learning_rate in zip(train_logs,learning_rates):
    # print(learning_rate)
    plt.plot(np.sqrt(train_log),label=f'{learning_rate}')

plt.title("Tuning learning rate")
plt.legend()
plt.show()

# L2 regularization
w = np.zeros([dim,1])
learning_rate = 10
lambdaa = 1 # rate of regularization
iter_time = 2000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
# visualize
train_log = []
val_log=[]
# training
for t in range(iter_time):
    # regularization
    wt = w[1:]
    reg = (np.dot(np.transpose(wt),wt) * lambdaa /2/wt.shape[0])[0,0]
    grad_reg = lambdaa * wt / wt.shape[0]
    grad_reg = np.concatenate((np.zeros([1,1]),grad_reg),axis=0)
    loss = np.sum(np.power(np.dot(x, w) - y, 2))/y.shape[0] + reg # mse
    val_loss = np.sum(np.power(np.dot(val_x, w) - val_y, 2))/val_y.shape[0] + reg
    train_log.append(loss)
    val_log.append(val_loss)
    if t%100==0:
        print(f"Epoch {t}:  train_loss: {loss}  val_loss: {val_loss}")
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)/y.shape[0] + grad_reg   #dim*1
    adagrad += gradient ** 2
    
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
plt.ylim([5.5,10])
plt.plot(np.sqrt(train_log),label='loss')
plt.plot(np.sqrt(val_log),label='val')
plt.legend()
plt.show()
'''
test
'''
test_data = pd.read_csv('./test.csv', header=None,encoding = 'big5')
test_data[test_data=='NR'] = 0
test_data.iloc[:,2:]=test_data.iloc[:,2:].astype('float32')
test_data
# normalize

for i,key in enumerate(test_data[1]):
    test_data.iloc[i,2:] = (test_data.iloc[i,2:] - m_a_s[i%18][0])/(m_a_s[i%18][1]+eps)
# reshape
test_x = np.empty([240,18*9])
idx = 0
for i in range(0,4320,18):
    test_x[idx,:] = np.reshape(test_data.iloc[i:i+18,2:].values,-1)
    idx += 1
# test
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
ans = np.dot(test_x,w)
ans

import csv
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans[i][0]]
        csv_writer.writerow(row)
        print(row)
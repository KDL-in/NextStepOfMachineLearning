import numpy as np

np.random.seed(0)
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './output_{}.csv'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
with open(Y_train_fpath) as f:
    next(f)
    Y = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:]
                       for line in f], dtype=float)


def normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)

    return X, X_mean, X_std


def train_dev_split(X, Y, dev_ratio=0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


# Normalize training and testing data
X, X_mean, X_std = normalize(X, train=True)
X_test, _, _ = normalize(
    X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

# Split data into training set and development set
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = train_dev_split(X, Y, dev_ratio=dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))


def shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


def sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


def f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return sigmoid(np.matmul(X, w) + b)


def predict(X, w, b):
    # This function returns a truth value prediction for each row of X
    # by rounding the result of logistic regression function.
    return np.round(f(X, w, b)).astype(np.int)


def accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc


def cross_entropy_loss(y_pred, Y_label,w=None,lambda_ = 0):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    N = y_pred.shape[0]
    cross_entropy = -(np.dot(Y_label, np.log(y_pred)) \
            + np.dot((1 - Y_label), np.log(1 - y_pred)))/N
    if lambda_:
        cross_entropy += lambda_/2/N * np.dot(w,w)
    return cross_entropy


def gradient(X, Y_label, w, b,lambda_ = 0):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = f(X, w, b)
    pred_error = y_pred - Y_label 
    N = Y_label.shape[0]
    w_grad = (np.sum(pred_error * X.T, 1) + lambda_ * w)/N
    b_grad = np.sum(pred_error)/N
    return w_grad, b_grad

# 使用全部数据
# X_train = X
# Y_train = Y


learning_rate = 0.03
lambda_ = 0.3
epoch = 100
batch_size = 8



w = np.zeros([X_train.shape[1],])
b = np.zeros([1, ])
iter = np.ceil(X_train.shape[0]/batch_size).astype(int)
step = 1
train_loss =[]
val_loss = []
train_acc = []
val_acc = []

for e in range(epoch):
    X_train, Y_train = shuffle(X_train, Y_train)
    for i in range(iter):
        BX = X_train[i*batch_size:(i+1)*batch_size]
        BY = Y_train[i*batch_size:(i+1)*batch_size]
        grad_w, grad_b = gradient(BX, BY, w, b,lambda_ = lambda_)
    
        w = w - learning_rate/np.sqrt(step)*grad_w
        b = b - learning_rate/np.sqrt(step)*grad_b

        step += 1
    Y_pred = f(X_train, w,b)
    Y_dev_pred = f(X_dev,w,b)
    train_loss.append(cross_entropy_loss(Y_pred,Y_train,w,lambda_))
    val_loss.append(cross_entropy_loss(Y_dev_pred,Y_dev,w,lambda_))
    Y_pred = np.round(Y_pred)
    Y_dev_pred = np.round(Y_dev_pred)
    train_acc.append(accuracy(Y_pred,Y_train))
    val_acc.append(accuracy(Y_dev_pred,Y_dev))

    print("t_loss: {:.4}, v_loss: {:.4},t_acc: {:.4},v_acc: {:.4}".format(train_loss[-1],val_loss[-1],train_acc[-1],val_acc[-1]))

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(val_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(val_acc[-1]))


# plot loss
from matplotlib import pyplot as plt
plt.plot(train_loss, label = 'train_loss')
plt.plot(val_loss,label = 'dev_loss')
plt.legend()
plt.show()
plt.plot(train_acc,label = 'train_acc')
plt.plot(val_acc,label = 'dev_acc')
plt.legend()
plt.show()

# Predict testing labels
# predictions = predict(X_test, w, b)
# with open(output_fpath.format('logistic'), 'w') as f:
#     f.write('id,label\n')
#     for i, label in  enumerate(predictions):
#         f.write('{},{}\n'.format(i, label))

# Print out the most significant weights

# with open(output_fpath.format('w_abs'),'w') as f:
#     f.write('id,weight\n')
#     for i,x in enumerate(abs(w)):
#         f.write('{},{}\n'.format(i,x))


# show weight distribution
plt.bar(range(len(w)),abs(w))
# show weight distribution(sorted)
plt.bar(range(len(w)),np.sort(abs(w)))

ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)[1:]
print("{:3}, {:30}, {}".format('id','features', 'weight'))
for i in ind[0:10]:
    print("{:3}, {:30}, {}".format(i,features[i].strip(), w[i]))

# featrues selection
# first 200th, w_200 = 0.014117181
w = np.load('./w.npy')
ind = np.argsort(np.abs(w))[::-1]
sind = np.sort(ind[:200])
X_train_s = X_train[:,sind]
X_dev_s = X_dev[:,sind]

learning_rate = 0.01
lambda_ = 0
epoch = 100
batch_size = 8
w_s = np.zeros([X_train_s.shape[1],])
b = np.zeros([1, ])
iter = np.ceil(X_train.shape[0]/batch_size).astype(int)
step = 1
train_loss =[]
val_loss = []
train_acc = []
val_acc = []

for e in range(epoch):
    X_train_s, Y_train = shuffle(X_train_s, Y_train)
    for i in range(iter):
        BX = X_train_s[i*batch_size:(i+1)*batch_size]
        BY = Y_train[i*batch_size:(i+1)*batch_size]
        grad_w, grad_b = gradient(BX, BY, w_s, b,lambda_ = lambda_)
    
        w_s = w_s - learning_rate/np.sqrt(step)*grad_w
        b = b - learning_rate/np.sqrt(step)*grad_b

        step += 1
    Y_pred = f(X_train_s, w_s,b)
    Y_dev_pred = f(X_dev_s,w_s,b)
    train_loss.append(cross_entropy_loss(Y_pred,Y_train,w_s,lambda_))
    val_loss.append(cross_entropy_loss(Y_dev_pred,Y_dev,w_s,lambda_))
    Y_pred = np.round(Y_pred)
    Y_dev_pred = np.round(Y_dev_pred)
    train_acc.append(accuracy(Y_pred,Y_train))
    val_acc.append(accuracy(Y_dev_pred,Y_dev))

    print("t_loss: {:.4}, v_loss: {:.4},t_acc: {:.4},v_acc: {:.4}".format(train_loss[-1],val_loss[-1],train_acc[-1],val_acc[-1]))

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(val_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(val_acc[-1]))


# plot loss
from matplotlib import pyplot as plt
plt.plot(train_loss, label = 'train_loss')
plt.plot(val_loss,label = 'dev_loss')
plt.legend()
plt.show()
plt.plot(train_acc,label = 'train_acc')
plt.plot(val_acc,label = 'dev_acc')
plt.legend()
plt.show()

# Predict testing labels
predictions = predict(X_test, w, b)
with open(output_fpath.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

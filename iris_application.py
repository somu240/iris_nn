from load_data import load_dataset
import pdb
import numpy as np


train_data_x, test_data_x, train_output_y, test_output_y = load_dataset()
test_output_y[test_output_y == 'Iris-setosa'] = 1
test_output_y[test_output_y != 1] = 0
train_output_y[train_output_y == 'Iris-setosa'] = 1
train_output_y[train_output_y != 1] = 0
print("Number of training examples: m_train = " + str(train_data_x.shape[0]))
print("Number of testing examples: m_test = " + str(test_data_x.shape[0]))
print("train_set_x shape: " + str(train_data_x.shape))
print("train_set_y shape: " + str(train_output_y.shape))
print("test_set_x shape: " + str(test_data_x.shape))
print("test_set_y shape: " + str(test_output_y.shape))

train_data_x = train_data_x.astype(float)
test_data_x = test_data_x.astype(float)
train_output_y = train_output_y.astype(float)
test_output_y = test_output_y.astype(float)

#initialize parameters

#np.random.seed(1)

w1 = np.random.randn(5,4)*np.sqrt(2/5)
b1 = np.zeros(shape=(5,1))

w2 = np.random.randn(4,5)*np.sqrt(2/4)
b2 = np.zeros(shape=(4,1))

w3 = np.random.randn(3,4)*np.sqrt(2/3)
b3 = np.zeros(shape=(3,1))

w4 = np.random.randn(2,3)*np.sqrt(2/2)
b4 = np.zeros(shape=(2,1))

w5 = np.random.randn(1,2)*np.sqrt(2/1)
b5 = np.zeros(shape=(1,1))


for i in range(0, 3000):

 for index in range(0, train_data_x.shape[0]):
  x1 = train_data_x[index].reshape((4, 1))
  Y = train_output_y[index].reshape((1, 1))

  z1 = np.dot(w1, x1) + b1
  a1 = np.maximum(0, z1)

  z2 = np.dot(w2, a1) + b2
  a2 = np.maximum(0, z2)

  z3 = np.dot(w3, a2) + b3
  a3 = np.maximum(0, z3)

  z4 = np.dot(w4, a3) + b4
  a4 = np.maximum(0, z4)

  z5 = np.dot(w5, a4) + b5
  a5 = 1 / (1 + np.exp(-z5))





  #cost = (-np.dot(Y, np.log(a5).T) - np.dot(1 - Y, np.log(1 - a5).T))
  cost = Y-a5

  print("cost is " + str(cost))

  # backward propagation

  dA5 = - (np.divide(Y, a5) - np.divide(1 - Y, 1 - a5))
  s = 1 / (1 + np.exp(-z5))

  dZ5 = dA5 * s * (1 - s)
  dW5 = np.dot(dZ5, a4.T)
  db5 = np.sum(dZ5, axis=1, keepdims=True)
  dA4 = np.dot(w5.T, dZ5)

  dZ4 = np.array(dA4, copy=True)
  dW4 = np.dot(dZ4, a3.T)
  db4 = np.sum(dZ4, axis=1, keepdims=True)
  dA3 = np.dot(w4.T, dZ4)

  dZ3 = np.array(dA3, copy=True)
  dW3 = np.dot(dZ3, a2.T)
  db3 = np.sum(dZ3, axis=1, keepdims=True)
  dA2 = np.dot(w3.T, dZ3)

  dZ2 = np.array(dA2, copy=True)
  dW2 = np.dot(dZ2, a1.T)
  db2 = np.sum(dZ2, axis=1, keepdims=True)
  dA1 = np.dot(w2.T, dZ2)

  dZ1 = np.array(dA1, copy=True)
  dW1 = np.dot(dZ1, x1.T)
  db1 = np.sum(dZ1, axis=1, keepdims=True)

  # update parameters

  learning_rate = 0.0001
  w5 = w5 - learning_rate * dW5
  b5 = b5 - learning_rate * db5

  w4 = w4 - learning_rate * dW4
  b4 = b4 - learning_rate * db4

  w3 = w3 - learning_rate * dW3
  b3 = b3 - learning_rate * db3

  w2 = w2 - learning_rate * dW2
  b2 = b2 - learning_rate * db2

  w1 = w1 - learning_rate * dW1
  b1 = b1 - learning_rate * db1


for index in range(0, test_data_x.shape[0]):
 x1 = test_data_x[index].reshape((4, 1))
 Y = test_output_y[index].reshape((1, 1))


 print("This item should be iris" + str(Y))



 z1 = np.dot(w1, x1) + b1
 a1 = np.maximum(0, z1)

 z2 = np.dot(w2, a1) + b2
 a2 = np.maximum(0, z2)

 z3 = np.dot(w3, a2) + b3
 a3 = np.maximum(0, z3)

 z4 = np.dot(w4, a3) + b4
 a4 = np.maximum(0, z4)

 z5 = np.dot(w5, a4) + b5
 a5 = 1 / (1 + np.exp(-z5))

 print(a5)






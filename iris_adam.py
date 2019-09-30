from load_data import load_dataset
import pdb
import numpy as np
import matplotlib.pyplot as plt


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

train_data_x = train_data_x.astype(float).reshape(train_data_x.shape[1],train_data_x.shape[0])
test_data_x = test_data_x.astype(float).reshape(test_data_x.shape[1],test_data_x.shape[0])
train_output_y = train_output_y.astype(float).reshape(train_output_y.shape[1],train_output_y.shape[0])
test_output_y = test_output_y.astype(float).reshape(test_output_y.shape[1],test_output_y.shape[0])

#initialize parameters



layer_dims = [4,5,4,3,2,1]
def initialize_parameters(layer_dims):
 parameters = {}
 for i in range(1,len(layer_dims)):

  parameters["w" + str(i)] = (np.random.randn(layer_dims[i],layer_dims[i-1]))*(np.sqrt(2/layer_dims[i-1]))
  parameters["b" + str(i)] = np.zeros(shape=(layer_dims[i],1))

 return parameters


def forward_propagation_relu(w,b,a_prev):
 z = np.dot(w,a_prev)+b
 a = np.maximum(0, z)
 linear_cache = w,a_prev,b,z
 activation_cache = z
 cache = (linear_cache,activation_cache)

 return a,cache

def forward_propagation_sigmoid(w,b,a_prev):
 z = np.dot(w,a_prev)+b
 a = 1 / (1 + np.exp(-z))
 linear_cache = w,a_prev,b,z
 activation_cache = z
 cache = (linear_cache,activation_cache)

 return a,cache

def back_propagation_sigmoid(da_prev,cache):
 linear_cache,activation_cache = cache
 w,a_prev,b,z = linear_cache
 m = a_prev.shape[1]
 s = 1 / (1 + np.exp(-activation_cache))
 dZ = da_prev * s * (1 - s)
 dw = (1./m)*np.dot(dZ, a_prev.T)
 db = (1./m)*np.sum(dZ, axis=1, keepdims=True)
 da = np.dot(w.T, dZ)

 return dw,db,da

def back_propagation_relu(da_prev,cache):
 linear_cache,activation_cache = cache
 w,a_prev,b,z = linear_cache
 m = a_prev.shape[1]
 dZ = np.array(da_prev, copy=True)
 dZ[z <= 0] = 0
 dw = (1./m)*np.dot(dZ, a_prev.T)
 db = (1./m)*np.sum(dZ, axis=1, keepdims=True)
 da = np.dot(w.T, dZ)

 return dw,db,da

def update_parameters(parameters,grads,learning_rate):
 for i in range(1,len(layer_dims)):
  parameters["w" + str(i)] = parameters["w" + str(i)] - learning_rate*grads['dw'+str(i)]
  parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate*grads['db'+str(i)]

 return parameters

def predict(x,parameters):
 m = x.shape[1]
 p = np.zeros((1, m))
 probas,cache = model_forward(x,parameters)
 for i in range(0, probas.shape[1]):
  if probas[0, i] > 0.8:
    p[0, i] = 1
  else:
    p[0, i] = 0

 return p

def model_forward(train_data,parameters):
 caches = []
 a_prev = train_data
 for i in range(1,5):
  w,b = parameters['w'+str(i)],parameters['b'+str(i)]
  a_temp,cache = forward_propagation_relu(w,b,a_prev)
  caches.append(cache)
  a_prev = a_temp
 al,cache = forward_propagation_sigmoid(parameters['w5'],parameters['b5'],a_prev)
 caches.append(cache)

 return al,caches

def model_backward(al,y,caches):
 grads = {}
 dal = - (np.divide(y, al) - np.divide(1 - y, 1 - al))
 grads['dw5'],grads['db5'],grads['da4'] = back_propagation_sigmoid(dal,caches[4])
 for l in reversed(range(len(layer_dims)-2)):
  grads['dw'+str(l+1)], grads['db'+str(l+1)], grads['da'+str(l)] = back_propagation_relu(grads['da'+str(l+1)],caches[l])

 return grads

def compute_cost(al,y):
 m = y.shape[1]
 cost = (1. / m) * (-np.dot(y, np.log(al).T) - np.dot(1 - y, np.log(1 - al).T))
 cost = np.squeeze(cost)

 return cost


def model(train_data_x,train_output_y,learning_rate,num_iterations,layer_dims,print_cost=False):
 a_prev = train_data_x
 y = train_output_y
 costs = []
 parameters = initialize_parameters(layer_dims)
 for i in range(0, num_iterations):
  al,caches = model_forward(train_data_x,parameters)
  cost = compute_cost(al, y)
  grads = model_backward(al,y, caches)
  update_parameters(parameters,grads,learning_rate)
  if print_cost and i % 100 == 0:
      print("Cost after iteration %i: %f" % (i, cost))
  if print_cost and i % 100 == 0:
      costs.append(cost)
 plt.plot(np.squeeze(costs))
 plt.ylabel('cost')
 plt.xlabel('iterations (per tens)')
 plt.title("Learning rate =" + str(learning_rate))
# plt.show()

 return parameters,grads


parameters,grads = model(train_data_x,train_output_y,0.00075,30000,layer_dims,True)
pred_test = predict(test_data_x, parameters)
print(pred_test)
print(test_output_y)

from load_data import load_dataset
import pdb
import numpy as np
import matplotlib.pyplot as plt
import math


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

def minibatch_data(train_data_x,train_output_y,batch_size):
 mini_batches = []
 no_of_batches = math.floor((train_data_x.shape[1])/batch_size)
 for i in range(no_of_batches):
  train_data = train_data_x[:, i*batch_size : (i+1)*batch_size]
  test_data = train_output_y[:, i*batch_size : (i+1)*batch_size]
  mini_batch = (train_data,test_data)
  mini_batches.append(mini_batch)

 if (train_data_x.shape[1]) % batch_size != 0:
  train_data = train_data_x[:,no_of_batches * batch_size:]
  test_data = train_output_y[:, no_of_batches * batch_size:]
  mini_batch = (train_data,test_data)
  mini_batches.append(mini_batch)


 return mini_batches



layer_dims = [4,5,4,3,2,1]
def initialize_parameters(layer_dims):
 parameters = {}
 for i in range(1,len(layer_dims)):

  parameters["w" + str(i)] = (np.random.randn(layer_dims[i],layer_dims[i-1]))*(np.sqrt(2/layer_dims[i-1]))
  parameters["b" + str(i)] = np.zeros(shape=(layer_dims[i],1))

 return parameters

def initialize_adam(parameters):
 v = {}
 s = {}
 for l in range(1,6):
  v["dw"+str(l)] = np.zeros( (parameters["w"+str(l)].shape[0] , parameters["w"+str(l)].shape[1]) )
  v["db"+str(l)] = np.zeros( (parameters["b"+str(l)].shape[0] , parameters["b"+str(l)].shape[1]) )
  s["dw"+str(l)] = np.zeros( (parameters["w"+str(l)].shape[0] , parameters["w"+str(l)].shape[1]) )
  s["db"+str(l)] = np.zeros( (parameters["b"+str(l)].shape[0] , parameters["b"+str(l)].shape[1]) )

 return v,s


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

def update_params_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon=1e-8):
 v_corrected = {}
 s_corrected = {}
 for i in range(1,6):
   v["dw"+str(i)] =  beta1*(v['dw'+str(i)]) + (1-beta1)*grads['dw'+str(i)]
   v["db" + str(i)] = beta1 * (v['db' + str(i)]) + (1 - beta1) * grads['db' + str(i)]
   v_corrected["dw"+str(i)] = v["dw"+str(i)]/(1-beta1**t)
   v_corrected["db" + str(i)] = v["db" + str(i)] / (1 - beta1 ** t)
   s["dw"+str(i)] =  beta2*(s['dw'+str(i)]) + (1-beta2)*grads['dw'+str(i)]*grads['dw'+str(i)]
   s["db" + str(i)] = beta2 * (s['db' + str(i)]) + (1 - beta2) * grads['db' + str(i)]*grads['db' + str(i)]
   s_corrected["dw"+str(i)] = s["dw"+str(i)]/(1-beta2**t)
   s_corrected["db" + str(i)] = s["db" + str(i)] / (1 - beta2 ** t)
   parameters["w" + str(i)] = parameters["w" + str(i)] - learning_rate*v_corrected["dw"+str(i)]/(np.sqrt(s_corrected["dw"+str(i)])+epsilon)
   parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate*v_corrected["db"+str(i)]/(np.sqrt(s_corrected["db"+str(i)])+epsilon)

 return parameters,v,s


def model(mini_batches,learning_rate,num_iterations,layer_dims,print_cost=False,beta1=0.9,beta2=0.99):
 costs = []
 parameters = initialize_parameters(layer_dims)
 t=0
 v,s = initialize_adam(parameters)
 for i in range(0, num_iterations):
  for mini_batch in mini_batches:
   a_prev,y = mini_batch
   al,caches = model_forward(a_prev,parameters)
   cost = compute_cost(al, y)
   grads = model_backward(al,y, caches)
   t=t+1
   parameters,v,s = update_params_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2)
   if print_cost and i % 100 == 0:
       print("Cost after iteration %i: %f" % (i, cost))
   if print_cost and i % 100 == 0:
       costs.append(cost)
 plt.plot(np.squeeze(costs))
 plt.ylabel('cost')
 plt.xlabel('iterations (per tens)')
 plt.title("Learning rate =" + str(learning_rate))
 plt.show()

 return parameters,grads

mini_batches = minibatch_data(train_data_x,train_output_y,10)
parameters,grads = model(mini_batches,0.0001,15000,layer_dims,True)
pred_test = predict(test_data_x, parameters)
print(pred_test)
print(test_output_y)

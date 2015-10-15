import numpy as np
import random
import math

random.seed(0)
class model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inp):
        o = inp
        for l in self.layers:
            o = l.forward(o)
        return o

    def backward(self, err, grad):
        for l in reversed(self.layers):
            err,grad = l.backward(err, grad)

# input both are numpy array
def MSE(expected, out):
    return 0.5 * np.square(np.subtract(expected, out)), np.subtract(expected, out)

class linear_layer:
    MAX = 1
    MIN = -1
    def __init__(self, num_input, num_output):
        num_input = num_input+1
        self.weights = np.array([random.uniform(linear_layer.MIN, linear_layer.MAX) for a in xrange(num_input*num_output)]).reshape(num_input,num_output)
        #print self.weights

    def forward(self, inp):
        #inp is 1d input array of size = num input of the nn
        inp = np.insert(inp, 0, 1) # add the bias term
        self.inp = inp
        return np.dot(inp, self.weights)

    def backward(self, error, error_grad):
        #print "err" , error, "\n input ",self.inp
        delta = np.multiply(error, self.inp)
        #print "Delta is", delta, "weights", self.weights
        a = np.dot(self.weights, error)
        self.weights = self.weights + 1 * delta.T
        #print "weights after", self.weights
        return 0, a[1:]

class sigmoid_layer():
    def cal(self, val):
        return 1/(1 + math.exp(-val))

    #inp should be a one dimentional array
    def forward(self, inp):
        self.out = np.array([self.cal(val) for val in inp])
        return self.out

    def backward(self, error, error_grad):
        self.out = self.out[np.newaxis].T
        #print "out, Error grad is ", self.out, error_grad
        return np.multiply(np.multiply(self.out, 1-self.out), error_grad), error_grad

"""
test = np.array([[0,0,1],
        [0,1,1],
        [1,0,1],
        [1,1,1]])
y = np.array([[0], [1], [1], [0]])
"""
test = np.array([[0,0],
        [0,1],
        [1,0],
        [1,1]])
y = np.array([[0], [0], [1], [1]])

h = model()
h.add(linear_layer(2,2))
h.add(sigmoid_layer())
h.add(linear_layer(2,2))
h.add(sigmoid_layer())
h.add(linear_layer(2,1))
h.add(sigmoid_layer())
for iter in xrange(10000):
    for v in xrange(len(test)):
        val = h.forward(test[v][np.newaxis])
        err,grad = MSE(y[v], val)
        if iter%1000 == 0:
            print err
        h.backward(err, grad)
for v in xrange(len(test)):
    print h.forward(test[v][np.newaxis])

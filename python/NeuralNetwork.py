import theano
from theano import tensor as T
import numpy as np
import collections

Y = T.dmatrix('output')
X = T.dmatrix('input')
W = T.dmatrix('weights')

def J(p, y, weights=None, lam=0):
	mse = np.square(p - y)
	l = 0
	if weights != None and lam != 0:
		l = lam*np.sum(np.square(weights))
	return mse + l

def sigmoid(x):
	return (1/(1+np.exp(x)))

class StandardDataset:
	def __init__(self, insize, outsize):
		self.data=[]
		self.insize = insize
		self.outsize = outsize
	
	def addData(x, y):
		if x.shape[1] != self.insize:
			print 'Bad input size'
			return
		if y.shape[1] != self.outsize:
			print 'Bad output size'
			return
		self.data.append((x, y))
		return

class FFNetwork:
	def __init__(self, *dimensions):
		self.weights = []
		rotatedd = collections.deque(dimensions)
		rotatedd.rotate(-1)
		for i, j in zip(dimensions[:-1],rotatedd):
			self.weights.append(np.random.rand(i+1,j))
		
		self.weights[-1] = np.random.rand(dimensions[-2]+1, dimensions[-1])
		return
	
	def learn(self, X, Y, alpha=0.01):
		grad = []
		z = []
		a = []
		print X.shape
		a.append(np.append(np.ones((X.shape[0], 1)) ,X, axis=1))
		for w in self.weights:
			z.append(a[-1].dot(w))
			a.append(np.append(np.ones((z[-1].shape[0], 1)), sigmoid(z[-1]), axis=1))
		print a


ffnet = FFNetwork(2, 4, 1)

ffnet.learn(np.array([[-1,1],[-2, 0]]), np.array([[1],[0]]))

print ffnet.weights
class RNNNetwork:
	pass

class LSTMNetwork:
	pass

from __future__ import print_function
import math
from collections import OrderedDict
import theano.tensor as T
import theano
import numpy as np
import numpy
import matplotlib.pyplot as plt
#import urllib.request

ARRAYMODE=True

def createWebpage(lstm, filename, maxChars=10000, array=False):
	print('Writing to file...')
	f = open(filename, 'w')
	status = 0
	curChar = ord('<')
	f.write(chr(curChar))
	chars = 0
	while status == 0 and chars < maxChars:
		if array:
			dinput = [0] * 256
			dinput[curChar] = 1
		else:
			dinput = curChar
		output = lstm.activate(dinput)
		if output[-1] < 0.5: status = 0
		else: status = 1

		if array:
			m = max(output)
			char = [i for i, j in enumerate(output) if j == m]
			try:
				curChar = char[0]
				f.write(chr(curChar))
			except ValueError:
				pass
		else:
			m = int(output[0])
			f.write(chr(m))
		chars = chars + 1
	print('Finished a file')
	return

#get Wiki pages
def getPage():
	response = urllib.request.urlopen('http://en.wikipedia.org/wiki/Special:Random')
	return response.read()

def convertPageToArrays(html):
	convertedData=[]
	for char in html:
		charList = [0]*256
		charList[char] = 1
		convertedData.append(charList)
	x = np.array(convertedData)
	y = np.append(x, np.zeros((x.shape[0], 1)), axis=1)
	y.itemset((y.shape[0]-1, y.shape[1]-1), 1)
	return (x, y)




#WIKI PAGE PART 2

class LSTMLayer:
	'''
		This assumes that in this recurrent net, there is a corresponding output to each input
	'''
	def __init__(self, in_size, out_size, cell_size=None, alpha=0.01, init_size=0.1,
			out_type='sigmoid', momentum=0):
		if cell_size == None:
			cell_size = in_size * 10
		self.alpha = alpha
		self.in_size = in_size
		self.out_size = out_size
		self.cell_size = cell_size
		self.C = theano.shared(value=np.zeros((1, cell_size)), name='LongTerm')
		self.h = theano.shared(value=np.zeros((1, out_size)), name='Previous Prediction')
		self.momentum = momentum
		x = T.dmatrix(name='input example')

		#Forget gate
		self.W_xf = theano.shared(value=init_size*np.random.rand(in_size, cell_size), name='x to forget gate')
		self.W_hf = theano.shared(value=init_size*np.random.rand(out_size, cell_size), name='h to forget gate')
		self.W_cf = theano.shared(value=init_size*np.random.rand(cell_size, cell_size), name='cell to forget gate')
		self.b_f = theano.shared(init_size*np.random.rand(1, cell_size), name='forget bias')

#		forget = T.nnet.sigmoid(T.dot(self.h, self.W_hf) + T.dot(self.C, self.W_cf) + T.dot(self.x, self.W_xf) + self.b_f)

		#Memories
		self.W_hm = theano.shared(value=init_size*np.random.rand(out_size, cell_size), name='h to memories')
		self.W_xm = theano.shared(value=init_size*np.random.rand(in_size, cell_size), name='x to memories')
		self.b_m = theano.shared(init_size*np.random.rand(1, cell_size), name='memory bias')

#		memories = T.tanh(T.dot(self.h, self.W_hm) + T.dot(self.x, self.W_xm) + self.b_m)

		#Remember Gate
		self.W_hr = theano.shared(value=init_size*np.random.rand(out_size, cell_size), name='h to remember')
		self.W_cr = theano.shared(value=init_size*np.random.rand(cell_size, cell_size), name='cell to \
		remember')
		self.W_xr = theano.shared(value=init_size*np.random.rand(in_size, cell_size), name='x to remember')
		self.b_r = theano.shared(value=init_size*np.random.rand(1, cell_size), name='remember bias')

#		remember = T.nnet.sigmoid(T.dot(self.h, self.W_hr) + T.dot(self.C, self.W_cr) + T.dot(self.x, self.W_xr) + self.b_r)

		#Output
		self.W_co = theano.shared(value=init_size*np.random.rand(cell_size, out_size), name='cell to out')
		self.W_ho = theano.shared(value=init_size*np.random.rand(out_size, out_size), name='hidden to out')
		self.W_xo = theano.shared(value=init_size*np.random.rand(in_size, out_size), name='x to out')
		self.b_o = theano.shared(value=init_size*np.random.rand(1, out_size), name='out bias')

		self.params = [self.W_xf, self.W_hf, self.W_cf, self.b_f, self.W_hm, self.W_xm, self.b_m,
				self.W_hr, self.W_cr, self.W_xr, self.b_r, self.W_co, self.W_ho,
				self.W_xo, self.b_o]

		def recurrence(x, h_tm1, c_tm1):
			rem = T.nnet.sigmoid(T.dot( h_tm1, self.W_hr) + T.dot( c_tm1 , self.W_cr) + T.dot( x, self.W_xr) + self.b_r)
			mem = T.tanh(T.dot( h_tm1, self.W_hm) + T.dot( x, self.W_xm) + self.b_m)
			forget = T.nnet.sigmoid(T.dot( h_tm1, self.W_hf) + T.dot( c_tm1, self.W_cf) + T.dot( x, self.W_xf) + self.b_f)

			z = T.dot( c_tm1 , self.W_co) + T.dot( h_tm1 , self.W_ho) + T.dot(x, self.W_xo) + self.b_o
			if out_type=='sigmoid':
				h_t = T.nnet.sigmoid(z)
			elif out_type=='linear':
				h_t = z

			c_t = self.C * forget + rem * mem
			return [h_t, c_t]

		([hidden, cell_state], _) = theano.scan(fn=recurrence, 
				sequences=x, 
				outputs_info=[self.h, self.C],
				n_steps=x.shape[0])

		output = hidden.reshape((hidden.shape[0], hidden.shape[2]))
		self.predict = theano.function([x], output, name='predict', updates=[(self.C, cell_state[-1]),
			(self.h, hidden[-1])])

		y = T.dmatrix(name='output')

		if out_type=='sigmoid':
			self.error = -T.mean((y)*T.log(output) + (1-y)*T.log(1-output))
		elif out_type=='linear':
			self.error = T.mean(T.sqr(y - output))
		self.J = theano.function([x, y], self.error)
		self.gradients = []
		self.mparams = []
		for p in self.params:
			self.gradients.append(T.grad(self.error, p))
			self.mparams.append(theano.shared(np.zeros(p.get_value().shape), name='momentum bs'))
		gradUpdates = OrderedDict((p, p - g) for p, g in zip(self.params, self.mparams))
		gradUpdates.update(OrderedDict((m, self.momentum * m + self.alpha * g) for m, g in
			zip(self.mparams, self.gradients)))
		self.learn = theano.function([x, y], outputs=self.error, updates=gradUpdates)

	def defineGradients(self):
		self.gradients = T.grad(cost=self.error, wrt=self.params)
		gradUpdates = OrderedDict((p, p - self.alpha * g) for p, g in zip(self.params, gradients))
		self.learn = theano.function([x, y], outputs=self.error, updates=gradUpdates)

	def setError(errorExpression):
		self.error = errorExpression
		self.J = theano.function([x, y], self.error)
		self.defineGradients()

	def reset(self):
		self.C.set_value(np.zeros((1, self.cell_size)))
		self.h.set_value(np.zeros((1, self.out_size)))
	
def testLSTM():
	#Generate sinx dataset
	x = []
	y = []
	for i in np.linspace(0, 10, 100):
		x.append(i)
		y.append(math.sin(i))
	plt.plot(x, y)
	x = np.array(x).reshape(len(x), 1)
	y = np.array(y).reshape(len(y), 1)

	lstm_test = LSTMLayer(1, 1, out_type='linear', momentum=0.5, alpha=0.01)
	print('Testing its prediction function')
	lstm_test.predict(x)
	print('Testing learning function')
	i = 1
	iterations = 0
	train_error = []
	while i > 1e-2:
		print(lstm_test.learn(x, y))
		i = lstm_test.learn(x, y)
		train_error.append(lstm_test.learn(x, y))
		iterations += 1
	print('Trained predictions:')
	plt.plot(np.linspace(0, 10, 100), lstm_test.predict(x))
	plt.figure()
	plt.plot(np.arange(iterations), train_error)
	plt.show()

testLSTM()

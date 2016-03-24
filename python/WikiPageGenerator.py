from __future__ import print_function
import math
from collections import OrderedDict
import theano.tensor as T
import theano
import numpy as np
import numpy
import matplotlib.pyplot as plt
from theano import config
import urllib.request

ARRAYMODE=True
config.floatX='float64'

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

def convertPageToArrays(html=getPage()):
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
			out_type='sigmoid', momentum=0, rprop=False, no_compile=False, in_var=None,
			out_var=None, verbose=False):
		if cell_size == None:
			cell_size = max(in_size, out_size)
		self.alpha = alpha
		self.in_size = in_size
		self.out_size = out_size
		self.cell_size = cell_size
		self.C = theano.shared(value=np.zeros((1, cell_size)), name='LongTerm')
		self.h = theano.shared(value=np.zeros((1, out_size)), name='Previous Prediction')
		self.momentum = momentum
		if in_var == None:
			x = T.matrix(name='input example')
		else:
			x = in_var
		if verbose:
			print('Constants have been initalized')

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

		#Hidden
		self.W_ch = theano.shared(value=init_size*np.random.rand(cell_size, out_size), name='cell to hidden')
		self.W_hh = theano.shared(value=init_size*np.random.rand(out_size, out_size), 
			name='hidden to hidden')
		self.W_xh =theano.shared(value=init_size*np.random.rand(in_size, out_size), name='x to hidden')
		self.b_h =theano.shared(value=init_size*np.random.rand(1, out_size), name='bias for hidden')


		self.params = [self.W_xf, self.W_hf, self.W_cf, self.b_f, self.W_hm, self.W_xm, self.b_m,
				self.W_hr, self.W_cr, self.W_xr, self.b_r, self.W_co, self.W_ho,
				self.W_xo, self.b_o, self.W_ch, self.W_hh, self.W_xh, self.b_h]
		if verbose:
			print('Weights have been initalized')

		def recurrence(x, h_tm1, c_tm1):
			rem = T.nnet.sigmoid(T.dot( h_tm1, self.W_hr) + T.dot( c_tm1 , self.W_cr) + T.dot( x, self.W_xr) + self.b_r)
			mem = T.tanh(T.dot( h_tm1, self.W_hm) + T.dot( x, self.W_xm) + self.b_m)
			forget = T.nnet.sigmoid(T.dot( h_tm1, self.W_hf) + T.dot( c_tm1, self.W_cf) + T.dot( x, self.W_xf) + self.b_f)

			z = T.dot(c_tm1 , self.W_co) + T.dot( h_tm1 , self.W_ho) + T.dot(x, self.W_xo) + self.b_o
			h_t = T.nnet.sigmoid(T.dot(c_tm1, self.W_ch) + T.dot(h_tm1, self.W_hh) + T.dot(x,
				self.W_xh) + self.b_h)
			out = z
			if out_type=='sigmoid':
				out = T.nnet.sigmoid(z)
			elif out_type=='linear':
				out = z

			c_t = self.C * forget + rem * mem
			return [z, h_t, c_t]

		([actualOutput, hidden, cell_state], _) = theano.scan(fn=recurrence, 
				sequences=x, 
				outputs_info=[None, self.h, self.C],
				n_steps=x.shape[0])
		if verbose:
			print('Recurrence has been set up')

		self.hidden = hidden
		self.cell_state = cell_state
		output = actualOutput.reshape((actualOutput.shape[0], actualOutput.shape[2]))
		self.out = output
		if not no_compile:
			self.predict = theano.function([x], output, name='predict', updates=[(self.C, cell_state[-1]),
				(self.h, hidden[-1])])

		if verbose:
			print('Output has been created')

		if out_var == None:
			y = T.dmatrix(name='output')
		else:
			y = out_var

		if not no_compile:
			if out_type=='sigmoid':
				self.error = -T.mean((y)*T.log(output) + (1-y)*T.log(1-output))
			elif out_type=='linear':
				self.error = T.mean(T.sqr(y - output))
			self.gradients = []
			self.mparams = []
			if rprop:
				self.prevw = []
				self.deltaw = []
				updates = []
				#initalize stuff
				for p in self.params:
					self.prevw.append(theano.shared(np.zeros(p.get_value().shape)).astype(config.floatX))
					self.deltaw.append(theano.shared(0.1 * np.ones(p.get_value().shape)).astype(config.floatX))
				for p, dw, pw in zip(self.params, self.deltaw, self.prevw):
					self.gradients.append(T.grad(self.error, p))
					#Array describing which values are when gradients are both positive or both negative
					simW = T.neq((T.eq((pw > 0), (self.gradients[-1] > 0))), (T.eq((pw < 0), (self.gradients[-1] <
						0))))
	
					#Array describing which values are when gradients are in opposite directions
					diffW = ((pw > 0) ^ (self.gradients[-1] > 0)) * (T.neq(pw, 0) * T.neq(self.gradients[-1], 0))
					updates.append((p, p - (T.sgn(self.gradients[-1]) * dw * (T.eq(diffW, 0)))))
					updates.append((dw, T.switch(diffW, dw *
						0.5, T.switch(simW, dw * 1.2, dw))))
					updates.append((pw, (T.sgn(self.gradients[-1]) * dw * (T.eq(diffW, 0)))))
				if not no_compile:
					self.learn = theano.function([x, y], outputs=self.error, updates=updates)
	
			else:
				for p in self.params:
					self.gradients.append(T.grad(self.error, p))
					self.mparams.append(theano.shared(np.zeros(p.get_value().shape), name='momentum bs'))
				gradUpdates = OrderedDict((p, p - g) for p, g in zip(self.params, self.mparams))
				gradUpdates.update(OrderedDict((m, self.momentum * m + self.alpha * g) for m, g in
					zip(self.mparams, self.gradients)))
				if not no_compile:
					self.learn = theano.function([x, y], outputs=self.error, updates=gradUpdates)
			if verbose:
				print('Done with gradient functions')

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

class LSTM():

	def __init__(self, *dim, **kwargs):
		self.alpha = kwargs.get('alpha', 0.01)
		self.momentum = kwargs.get('alpha', 0)
		rprop = kwargs.get('rprop', False)
		out_type = kwargs.get('out_type', 'sigmoid')
		self.layers = []
		verbose = kwargs.get('verbose', False)
		x = T.matrix('Input')
		y = T.matrix('Output')
		self.layers.append(LSTMLayer(dim[0], dim[1], no_compile=True, in_var=x, verbose=False))
		for i in range(1, len(dim) - 1):
			self.layers.append(LSTMLayer(dim[i], dim[i+1], no_compile=True, 
				in_var=self.layers[-1].out))
		

		if verbose: 
			print('Number of layers:' + str(len(self.layers)))
			print('Finished with initialization of layers -> Defining prediction')

		#Defining updates for all layers:
		layerUpdates = []
		for l in self.layers:
			layerUpdates.append((l.C, l.cell_state[-1]))
			layerUpdates.append((l.h, l.hidden[-1]))

		#Define prediction:
		prediction = self.layers[-1].out
		self.predict = theano.function([x], prediction, updates=layerUpdates)

		if verbose:
			print('defining error')
		#Define Error
		if out_type=='sigmoid':
			self.error = -T.mean((y)*T.log(prediction) + (1-y)*T.log(1-prediction))
		elif out_type=='linear':
			self.error = T.mean(T.sqr(y - prediction))
		
		if verbose:
			print('Wrapping paramaters')
		#Define paramater list
		self.params = []
		for i in self.layers:
			self.params = self.params + i.params

		if verbose:
			print('Defining Gradients')
		#Define gradients
		self.gradients = []
		for p in self.params:
			self.gradients.append(T.grad(cost=self.error, wrt=p))

		#Set training functions:
		if rprop:
			deltaw = []
			prevw = []
			updates = []

			#initalize stuff
			for p in self.params:
				prevw.append(theano.shared(np.zeros(p.get_value().shape)).astype(config.floatX))
				deltaw.append(theano.shared(0.1 * np.ones(p.get_value().shape)).astype(config.floatX))

			#Acutal algorithm
			for p, dw, pw in zip(self.params, deltaw, prevw):
				self.gradients.append(T.grad(self.error, p))
				#Array describing which values are when gradients are both positive or both negative
				simW = T.neq((T.eq((pw > 0), (self.gradients[-1] > 0))), (T.eq((pw < 0), (self.gradients[-1] <
					0))))

				#Array describing which values are when gradients are in opposite directions
				diffW = ((pw > 0) ^ (self.gradients[-1] > 0)) * (T.neq(pw, 0) * T.neq(self.gradients[-1], 0))
				updates.append((p, p - (T.sgn(self.gradients[-1]) * dw * (T.eq(diffW, 0)))))
				updates.append((dw, T.switch(diffW, dw *
					0.5, T.switch(simW, dw * 1.2, dw))))
				updates.append((pw, (T.sgn(self.gradients[-1]) * dw * (T.eq(diffW, 0)))))
			self.learn = theano.function([x, y], self.error, updates=updates)
		else:
			for p in self.params:
				self.gradients.append(T.grad(self.error, p))
				self.mparams.append(theano.shared(np.zeros(p.get_value().shape), name='momentum bs'))
			gradUpdates = OrderedDict((p, p - g) for p, g in zip(self.params, self.mparams))
			gradUpdates.update(OrderedDict((m, self.momentum * m + self.alpha * g) for m, g in
				zip(self.mparams, self.gradients)))
			self.learn = theano.function([x, y], self.error, updates=gradUpdates)
		
		if verbose:
			print('Finished initalization')
	
def testLSTM(graphs=True):
	#Generate sinx dataset
	x = []
	y = []
	for i in np.linspace(0, 10, 100):
		x.append(i)
		y.append(math.sin(i))
	if graphs: plt.plot(x, y)
	x = np.array(x).reshape(len(x), 1)
	y = np.array(y).reshape(len(y), 1)

	lstm_test = LSTM(1, 10, 1, out_type='linear', momentum=0.5, alpha=0.01, rprop=True,
			cell_size=100, verbose=True)
	print('Testing its prediction function')
	lstm_test.predict(x)
	print('Testing learning function')
	i = 1
	iterations = 0
	train_error = []
	while i > 1e-4:
		print(lstm_test.learn(x, y))
		i = lstm_test.learn(x, y)
		train_error.append(lstm_test.learn(x, y))
		iterations += 1
	print('Trained predictions:')
	if graphs:
		plt.plot(np.linspace(0, 10, 100), lstm_test.predict(x))
		plt.figure()
		plt.plot(np.arange(iterations), train_error)
		plt.show()

testLSTM(graphs=False)

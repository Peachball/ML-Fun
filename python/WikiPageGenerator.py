from collections import OrderedDict
import numpy
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
import matplotlib.pyplot as plt
import urllib.request

ARRAYMODE=False

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





#WIKI PAGE PART 2
import theano.tensor as T
import theano
import numpy as np

class LSTMLayer:
	'''
		This assumes that in this recurrent net, there is a corresponding output to each input
	'''
	def __init__(self, in_size, out_size, cell_size=10, alpha=0.01):
		self.alpha = alpha
		self.in_size = in_size
		self.out_size = out_size
		self.cell_size = cell_size
		self.C = theano.shared(value=np.zeros((1, cell_size)), name='LongTerm')
		self.h = theano.shared(value=np.zeros((1, out_size)), name='Previous Prediction')
		x = T.dmatrix(name='input')

		#Forget gate
		self.W_xf = theano.shared(value=np.random.rand(in_size, cell_size), name='x to forget gate')
		self.W_hf = theano.shared(value=np.random.rand(out_size, cell_size), name='h to forget gate')
		self.W_cf = theano.shared(value=np.random.rand(cell_size, cell_size), name='cell to forget gate')
		self.b_f = theano.shared(np.random.rand(1, cell_size), name='forget bias')

#		forget = T.nnet.sigmoid(T.dot(self.h, self.W_hf) + T.dot(self.C, self.W_cf) + T.dot(self.x, self.W_xf) + self.b_f)

		#Memories
		self.W_hm = theano.shared(value=np.random.rand(out_size, cell_size), name='h to memories')
		self.W_xm = theano.shared(value=np.random.rand(in_size, cell_size), name='x to memories')
		self.b_m = theano.shared(np.random.rand(1, cell_size), name='memory bias')

#		memories = T.tanh(T.dot(self.h, self.W_hm) + T.dot(self.x, self.W_xm) + self.b_m)

		#Remember Gate
		self.W_hr = theano.shared(value=np.random.rand(out_size, cell_size), name='h to remember')
		self.W_cr = theano.shared(value=np.random.rand(cell_size, cell_size), name='cell to \
		remember')
		self.W_xr = theano.shared(value=np.random.rand(in_size, cell_size), name='x to remember')
		self.b_r = theano.shared(value=np.random.rand(1, cell_size), name='remember bias')

#		remember = T.nnet.sigmoid(T.dot(self.h, self.W_hr) + T.dot(self.C, self.W_cr) + T.dot(self.x, self.W_xr) + self.b_r)

		#Output
		self.W_co = theano.shared(value=np.random.rand(cell_size, out_size), name='cell to out')
		self.W_ho = theano.shared(value=np.random.rand(out_size, out_size), name='hidden to out')
		self.W_xo = theano.shared(value=np.random.rand(in_size, out_size), name='x to out')
		self.b_o = theano.shared(value=np.random.rand(1, out_size), name='out bias')

		self.params = [self.W_xf, self.W_hf, self.W_cf, self.b_f, self.W_hm, self.W_xm, self.b_m,
				self.W_hr, self.W_cr, self.W_xr, self.b_r, self.W_co, self.W_ho,
				self.W_xo, self.b_o]

		def recurrence(x, h_tm1, c_tm1):
			rem = T.nnet.sigmoid(T.dot( h_tm1, self.W_hr) + T.dot( c_tm1 , self.W_cr) + T.dot( x, self.W_xr) + self.b_r)
			mem = T.tanh(T.dot( h_tm1, self.W_hm) + T.dot( x, self.W_xm) + self.b_m)
			forget = T.nnet.sigmoid(T.dot( h_tm1, self.W_hf) + T.dot( c_tm1, self.W_cf) + T.dot( x, self.W_xf) + self.b_f)

			h_t = T.nnet.sigmoid(T.dot( c_tm1 , self.W_co) + T.dot( h_tm1 , self.W_ho) + T.dot(x, self.W_xo) + self.b_o)
			c_t = self.C * forget + rem * mem
			return [h_t, c_t]

		[hidden, cell_state], _ = theano.scan(fn=recurrence, 
				sequences=x, 
				outputs_info=[self.h, self.C],
				n_steps=x.shape[0])

		output = hidden
		self.predict = theano.function([x], output, name='predict', updates=[(self.C, cell_state[-1]),
			(self.h, hidden[-1])])

		y = T.dmatrix()
		self.error = -T.mean((y)*T.log(output) + (1-y)*T.log(1-output))
		self.J = theano.function([x, y], self.error)
		gradients = T.grad(cost=self.error, wrt=self.params)
		gradUpdates = OrderedDict((p, p - self.alpha * g) for p, g in zip(self.params, gradients))
		self.learn = theano.function([x, y], outputs=self.error, updates=gradUpdates)

	def reset(self):
		self.C.set_value(np.zeros((1, self.cell_size)))
		self.h.set_value(np.zeros((1, self.out_size)))
	
	def batchLearning(self, x, y, iterations=100):
		self.reset()
		for i in range(iterations):
			print (self.learn(x,y))

lstm = LSTMLayer(2, 2)


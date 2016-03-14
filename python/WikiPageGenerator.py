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
	def __init__(self, in_size, out_size, cell_size=10):
		self.C = theano.shared(value=np.zeros((1, cell_size)))
		self.h = theano.shared(value=np.zeros((1, out_size)))
		self.x = T.dmatrix()
		self.y = T.dmatrix()

		#Forget gate
		self.W_xf = theano.shared(value=np.random.rand(in_size, cell_size))
		self.W_hf = theano.shared(value=np.random.rand(out_size, cell_size))
		self.W_cf = theano.shared(value=np.random.rand(cell_size, cell_size))
		self.b_f = theano.shared(np.random.rand(1, cell_size))

		forget = T.nnet.sigmoid(T.dot(self.h, self.W_cf) + T.dot(self.x, self.W_xf) + self.b_f)

		#Memories
		self.W_hm = theano.shared(value=np.random.rand(out_size, cell_size))
		self.W_xm = theano.shared(value=np.random.rand(in_size, cell_size))
		self.b_m = theano.shared(np.random.rand(1, cell_size))

		memories = T.tanh(T.dot(self.h, self.W_hm) + T.dot(self.x, self.W_xm) + self.b_m)

		#Remember Gate
		self.W_hr = theano.shared(value=np.random.rand(out_size, cell_size))
		self.W_cr = theano.shared(value=np.random.rand(cell_size, cell_size))
		self.W_xr = theano.shared(value=np.random.rand(in_size, cell_size))
		self.b_r = theano.shared(value=np.random.rand(1, cell_size))

		remember = T.nnet.sigmoid(T.dot(self.h, self.W_hr) + T.dot(self.x, self.W_xr) + self.b_r)

		#Output
		self.W_co = theano.shared(value=np.random.rand(cell_size, out_size))
		self.b_co = theano.shared(value=np.random.rand(1, out_size))
		self.W_ho = theano.shared(value=np.random.rand(out_size, out_size))
		self.W_xo = theano.shared(value=np.random.rand(in_size, out_size))
		self.b_xo = theano.shared(value=np.random.rand(1, out_size))

		output = T.nnet.sigmoid(T.dot(self.C, self.W_co) + self.b_co) * \
				T.nnet.sigmoid(T.dot(self.h, self.W_ho) + T.dot(self.x, self.W_xo) + self.b_xo)
		self.predict = theano.function([self.x], output, updates=[(self.h, output),
			(self.C, self.C * forget + remember * memories)])

		error = -T.mean((self.y)*T.log(output) + (1-self.y)*T.log(1-output))
		self.J = theano.function([self.x, self.y], error)

	def reset(self):
		self.C.set_value(np.zeros((1, cell_size)))
		self.h.set_value(np.zeros((1, out_size)))

	def learn(self, x, y):
		self.reset()
		for a, b in zip(x, y):
			if b == None:
				predict(a)
		pass

lstm = LSTMLayer(4, 10)

for i in range(10):
	print(lstm.predict(np.array([[1,2,3,4]])))

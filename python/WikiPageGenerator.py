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

from pybrain.datasets import SequentialDataSet
from itertools import cycle

def fillDataSet(data, num=1, array=False):
	print('Getting Wiki articles...')
	for i in range(num):
		data.newSequence()
		html = getPage()
		for char1, char2 in zip(html, html[1:]):
			if array:
				dinput = [0] * 256
				doutput = [0]  * 257
				dinput[char1] = 1
				doutput[char2] = 1
			else:
				dinput = [char1]
				doutput = [char2, 0]
			data.addSample(dinput, doutput)
		char1 = html[-1]

		if array:
			dinput = [0] * 256
			doutput = [0] * 257
			dinput[char1] = 1
			doutput[-1] = 1
		else:
			dinput = char1
			douput = [0, 0]
		data.addSample(dinput, doutput)
		print('\r Article # {}/{}'.format(i+1, num), end="")
	print('\r\n Finished getting Articles')
	return data

ds = SequentialDataSet(1, 2)

fillDataSet(ds, 1)

#Set up lstm stuff
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer

try:
	lstm = NetworkReader.readFrom('Wiki.xml')
	print('Loaded lstm from file')
except:
	lstm = buildNetwork(1, 5, 2, hiddenclass=LSTMLayer, outputbias=False,
		recurrent=True)
	print('Created new network')

from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.supervised.trainers import BackpropTrainer

createWebpage(lstm, 'test.html')

trainer = BackpropTrainer(lstm, dataset=ds)
train_errors = [] # save errors for plotting later
EPOCHS_PER_CYCLE = 1
CYCLES = 10
EPOCHS = EPOCHS_PER_CYCLE * CYCLES
for i in range(CYCLES):
	print('Started training')
	trainer.trainEpochs(EPOCHS_PER_CYCLE)
	train_errors.append(trainer.testOnData())
	epoch = (i+1) * EPOCHS_PER_CYCLE
	print("\r epoch {}/{}".format(epoch, EPOCHS), end="")

plt.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()

print('Wrote network down')
NetworkWriter.writeToFile(lstm, 'Wiki.xml')

createWebpage(lstm, 'test.html')


lstm.reset()

createWebpage(lstm, 'test.html')

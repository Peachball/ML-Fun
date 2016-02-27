from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
import struct

def readInput(filename, length=60000):
	data = []
	f = open(filename, 'rb')
	f.read(4)
	images = int.from_bytes( f.read(4), byteorder='big', signed=True)
	rows = int.from_bytes( f.read(4), byteorder='big', signed=True)
	cols = int.from_bytes( f.read(4), byteorder='big', signed=True)
	if length > images:
		length = images
	for i in range(length):
		image = []
		for j in range(rows*cols):
			image.append(struct.unpack('B',f.read(1))[0])
		data.append(image)
		print('\r Reading {}/{}'.format(i+1, images), end="")
	f.close()
	return data

def readLabels(filename, length=60000):
	labels=[]
	f = open(filename, 'rb')
	f.read(4)
	ls = int.from_bytes(f.read(4), byteorder='big', signed=True)
	if length > ls:
		length = ls
	for i in range(length):
		testLabel = [0] * 10
		testLabel[struct.unpack('B', f.read(1))[0]] = 1
		labels.append(testLabel)
		print('\r Reading {}/{}'.format(i+1, ls), end="")
	f.close()
	return labels

INPUT_FILE_NAME = '../train-images.idx3-ubyte'
INPUT_LABELS_NAME = '../train-labels.idx1-ubyte'

images = readInput(INPUT_FILE_NAME)
labels = readLabels(INPUT_LABELS_NAME)

data = ClassificationDataSet(28*28, 10)
for i in range(len(images)):
	data.addSample(images[i], labels[i])

print('\n\r Creating network')
NN = buildNetwork(28*28, 1000, 1000, 10, outclass=SoftmaxLayer)

trainer = BackpropTrainer(NN, dataset=data)

print('Started training')
for i in range(100):
	print('Train errors:' + str(trainer.train()))

NetworkWriter.writeToFile(NN, 'handwriting.xml')

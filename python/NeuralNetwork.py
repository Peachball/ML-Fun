import struct
import theano
from theano import tensor as T
import numpy as np
import collections

class FFNet:
    def __init__(self, alpha=0.01, *dim):
        self.in_size = dim[0]
        self.out_size = dim[-1]
        self.W = []
        self.b = []
        for i in range(len(dim)-1):
            self.W.append(theano.shared(value=np.random.rand(dim[i], dim[i+1])))
            self.b.append(theano.shared(value=np.random.random()))

        X = T.dmatrix('input')
        y = T.dmatrix('output')
        a = []
        a.append(X)
        for w, bias in zip(self.W, self.b):
            a.append(T.nnet.sigmoid(T.dot(a[-1], w) + bias))
        prediction = a[-1]
        self.predict = theano.function([X], prediction)
        self.error = -T.mean((y)*T.log(prediction) + (1-y)*T.log(1-prediction))

        self.J = theano.function([X, y], self.error)

        self.alpha = theano.shared(alpha)
        g_b = []
        g_w = []
        updates = []
        i = 0
        for w, bias in zip(self.W, self.b):
            g_w.append(T.grad(self.error, w))
            g_b.append(T.grad(self.error, bias))
            updates.append((w, w - self.alpha * g_w[i]))
            updates.append((bias, bias - self.alpha * g_b[i]))
            i = i + 1

        self.learn = theano.function([X, y], self.error, updates=updates)

    def batchLearning(self, X, y, iterations=100):
        for i in range(iterations):
            print (self.learn(X, y))

    def getWeightValues(self):
        return self.W

def readMNISTData(length=10000):
    images = open('../train-images.idx3-ubyte', 'rb')
    labels = open('../train-labels.idx1-ubyte', 'rb')
    images.read(8)
    labels.read(8)
    def readInt(isn=True):
        if isn:
            return int.from_bytes(images.read(4), byteorder='big', signed=True)
        else:
            return int.from_bytes(labels.read(4), byteorder='big', signed=True)
    xsize = readInt()
    ysize = readInt()
    def readImage():
        img = []
        for i in range(xsize):
            for j in range(ysize):
                img.append(struct.unpack('>B', images.read(1))[0])
        return img

    def readLabel():
        testLabel = [0]*10
        testLabel[struct.unpack('B', labels.read(1))[0]] = 1
        return testLabel

    imgs = []
    lbls = []
    for i in range(length):
        imgs.append(readImage())
        lbls.append(readLabel())
        print('\r Read {}/{}'.format(i, length), end="")
    return (np.array(imgs), np.array(lbls))

x, y = readMNISTData(60000)

nn = FFNet(0.01, 28*28, 1000, 10)


nn.batchLearning(x, y, iterations=100000)

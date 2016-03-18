from collections import OrderedDict
import struct
import theano
from theano import tensor as T
from theano import pp
import numpy as np
import collections

class FFNet:
    def __init__(self, alpha=0.01, init_size=0.1, *dim):
        self.in_size = dim[0]
        self.out_size = dim[-1]
        self.W = []
        self.b = []
        for i in range(len(dim)-1):
            self.W.append(theano.shared(value=init_size*(0.5 * np.random.rand(dim[i], dim[i+1]) - 1), name=('Weight' + str(i))))
            self.b.append(theano.shared(value=init_size*(0.5 * np.random.rand(dim[i+1]) - 1), name=('Bias' + str(i))))

        X = T.dmatrix('input')
        y = T.dmatrix('output')
        a = []
        a.append(X)
        for w, bias in zip(self.W, self.b):
            a.append(T.nnet.sigmoid(T.dot(a[-1], w) + bias))
        prediction = a[-1]
        self.predict = theano.function([X], prediction)
        self.error = -T.mean((y)*T.log(prediction) + (1-y)*T.log(1-prediction))
        
        theano.printing.debugprint(self.predict)

        self.J = theano.function([X, y], self.error)

        self.alpha = alpha
        g_b = []
        g_w = []
        self.params = [self.W, self.b]
        updates = []
        i = 0
        for w, bias in zip(self.W, self.b):
            g_w.append(T.grad(self.error, w))
            g_b.append(T.grad(self.error, bias))
            updates.append((w, w - self.alpha * g_w[i]))
            updates.append((bias, bias - self.alpha * g_b[i]))
            i = i + 1

        self.learn = theano.function([X, y], self.error, updates=updates)

    def batchLearning(self, X, y, iterations=100, verbose=False):
        for i in range(iterations):
            if not verbose:
                self.learn(X, y)
            else:
                print(self.learn(X,y))

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
    print('Done reading')
    return (np.array(imgs), np.array(lbls))
    
def readcv():
    images = open('../t10k-images.idx3-ubyte', 'rb')
    labels = open('../t10k-labels.idx1-ubyte', 'rb')
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
    for i in range(10):
        imgs.append(readImage())
        lbls.append(readLabel())
        print('\r Read {}/{}'.format(i, 10), end="")
        
    print ('Done Reading')
    return (np.array(imgs), np.array(lbls))
    
    
def percentError(net, x, y):
    p = np.argmax(net.predict(x), axis=1)
    ans = np.argmax(y, axis=1)
    accuracy = np.sum(np.equal(ans, p))
    print(ans, net.predict(x), p, np.equal(ans, p))
    return accuracy

x, y = readMNISTData(1)

xcv, ycv = readcv()

nn = FFNet(1, 1, 28*28, 100, 10)


print(percentError(nn, x, y))
nn.batchLearning(x, y, iterations=1000)
print(percentError(nn, x, y))

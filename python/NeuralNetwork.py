from __future__ import print_function
import matplotlib.pyplot as plt
import struct
from collections import OrderedDict
import struct
import theano
from theano import tensor as T
from theano import pp
import numpy as np
import collections

class FFNet:
    def __init__(self, *dim, **kwargs):
        self.in_size = dim[0]
        self.out_size = dim[-1]
        self.W = []
        self.b = []
        self.momentumb = []
        self.momentumw = []

        init_size = kwargs.get('init_size', 0.01)
        for i in range(len(dim)-1):
            weight = init_size * ( np.random.rand(dim[i], dim[i+1]) - 0.5)
            bias = init_size * (np.random.rand(dim[i+1]) - 0.5)

            mw = np.zeros((dim[i], dim[i+1]))
            mb = np.zeros((dim[i+1]))
            self.momentumb.append(theano.shared(value=mb, name='Bias \
                momentum'))
            self.momentumw.append(theano.shared(value=mw, name='Weight \
                momentum'))
            self.W.append(theano.shared(value=weight, name=('Weight' +
                str(i))))
            self.b.append(theano.shared(value=bias, name=('Bias' +
                str(i))))

        self.momentum = kwargs.get('momentum', 0)

        X = T.dmatrix('input')
        y = T.dmatrix('output')
        a = []
        a.append(X)
        for w, bias in zip(self.W, self.b):
            a.append(T.nnet.sigmoid(T.dot(a[-1], w) + bias))
#        hidden = theano.printing.Print('Activation of middle')(a[1])
        prediction = a[-1]
        self.predict = theano.function([X], prediction)
        self.error = -T.mean((y)*T.log(prediction) + (1-y)*T.log(1-prediction))
        self.J = theano.function([X, y], self.error)
#        self.showHidden = theano.function([X], hidden, mode='DebugMode')

        self.alpha = theano.shared(kwargs.get('alpha', 0.01))
        g_b = []
        g_w = []
        updates = []
        i = 0
        for (w, bias, mw, mb) in zip(self.W, self.b, self.momentumw, self.momentumb):
            g_w.append(T.grad(self.error, w))
            g_b.append(T.grad(self.error, bias))
            updates.append((mb, self.momentum * mb + self.alpha * g_b[-1]))
            updates.append((mw, self.momentum * mw + self.alpha * g_w[-1]))
            updates.append((w, w - mw))
            updates.append((bias, bias - mb))
            i = i + 1
        '''
        weight_grad = theano.printing.Print('Intermediate weight gradient')(g_w[0])
        bias_grad = theano.printing.Print('Intermediate bias gradient')(g_b[0])
        self.getGrad= theano.function([X, y], [weight_grad, bias_grad])
        '''
        self.learn = theano.function([X, y], self.error, updates=updates)

    def batchLearning(self, X, y, iterations=100, verbose=False, plot=False):
        it = []
        err = []
        for i in range(iterations):
            if verbose:
                err.append(self.learn(X,y))
                print(err[-1])
            else:
                err.append(self.learn(X,y))
            it.append(i)
        if plot:
            plt.plot(it, err)
            plt.show()


    def getWeightValues(self):
        return self.W
    
    def converge(self, X, y, plot=False, verbose=False, maxError=1e-2):
        it = []
        err = []
        prevError = self.learn(X, y)
        i=0
        alpha = self.alpha.get_value()
        while self.learn(X, y) > maxError:
            if verbose:
                print(self.learn(X,y))
            else:
                self.learn(X,y)
            it.append(i*3)
            err.append(self.learn(X,y))
            i += 1
        if plot:
            plt.plot(it, err)
            plt.show()

def readMNISTData(length=10000):
    images = open('../train-images.idx3-ubyte', 'rb')
    labels = open('../train-labels.idx1-ubyte', 'rb')
    images.read(8)
    labels.read(8)
    def readInt(isn=True):
        if isn:
            return struct.unpack('>i', images.read(4))[0]
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
        print('\r Read {}/{}'.format(i+1, length), end="")
    print('Done reading')
    return (np.array(imgs), np.array(lbls))
    
def readcv():
    images = open('../t10k-images.idx3-ubyte', 'rb')
    labels = open('../t10k-labels.idx1-ubyte', 'rb')
    images.read(8)
    labels.read(8)
    def readInt(isn=True):
        if isn:
            return struct.unpack('>i', images.read(4))[0]
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
    for i in range(10000):
        imgs.append(readImage())
        lbls.append(readLabel())
        print('\r Read {}/{}'.format(i+1, 10000), end="")
        
    print ('Done Reading')
    return (np.array(imgs), np.array(lbls))
    
def percentError(net, x, y):
    p = np.argmax(net.predict(x), axis=1)
    ans = np.argmax(y, axis=1)
    accuracy = np.sum(np.equal(ans, p))
    return (accuracy, y.shape[0])

x, y = readMNISTData(20000)

xcv, ycv = readcv()

xor = np.array([[0,0],[0,1],[1,0],[1,1]])
yor = np.array([[0],[1],[1],[0]])

handwriting = FFNet(28**2, 1000, 10, momentum=0.5, alpha=0.01)
xornetwork = FFNet(2, 2, 1, alpha=0.1, init_values=1)

handwriting.converge(x, y, verbose=True, plot=True)
print(percentError(handwriting, xcv, ycv))

i = 0
while raw_input('Guess') != 'q':
    image = xcv[i,:]
    image = image.reshape(28, 28)
    plt.imshow(image, cmap='Greys')
    plt.show()
    print('Computer predicts: ' + str(np.argmax(handwriting.predict(ycv[i,:]))))
    print('Acutal: ' + str(np.argmax(ycv[i,:])))
    i += 1

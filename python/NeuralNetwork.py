from __future__ import print_function
import matplotlib.pyplot as plt
import struct
from collections import OrderedDict
import struct
import theano
from theano import tensor as T
from theano import pp
from theano import config
import numpy as np
import collections

theano.config.floatX = 'float64'
np.random.seed(1233)

class FFNet:
    def __init__(self, *dim, **kwargs):
        self.in_size = dim[0]
        self.out_size = dim[-1]
        self.W = []
        self.b = []
        self.momentumb = []
        self.momentumw = []
        rprop = kwargs.get('rprop', False)
        init_size = kwargs.get('init_size', 0.01)
        if rprop:
            self.prevStepb = []
            self.prevStepw = []
            self.deltaw = []
            self.deltab = []
        for i in range(len(dim)-1):
            weight = init_size * ( np.random.rand(dim[i], dim[i+1]) - 0.5)
            bias = init_size * (np.random.rand(dim[i+1]) - 0.5)

            if rprop:
                mw = 0.1 * np.ones((dim[i], dim[i+1]), dtype=config.floatX)
                mb = 0.1 * np.ones((dim[i+1]), dtype=config.floatX)
                self.deltaw.append(theano.shared(value=mw, name='weight \
                    delta').astype(config.floatX))
                self.deltab.append(theano.shared(value=mb, name='bias delta').astype(config.floatX))
                self.prevStepw.append(theano.shared(value=np.zeros((dim[i], dim[i+1]),
                    dtype=config.floatX), name='weight prev').astype(config.floatX))
                self.prevStepb.append(theano.shared(value=np.zeros(dim[i+1], dtype=config.floatX),
                    name='bias prev').astype(config.floatX))
            else:
                mw = np.zeros((dim[i], dim[i+1]), dtype=config.floatX)
                mb = np.zeros((dim[i+1]), dtype=config.floatX)
                self.momentumb.append(theano.shared(value=mb, name='Bias \
                    momentum').astype(config.floatX))
                self.momentumw.append(theano.shared(value=mw, name='Weight \
                    momentum').astype(config.floatX))
            self.W.append(theano.shared(value=weight, name=('Weight' +
                str(i))).astype(config.floatX))
            self.b.append(theano.shared(value=bias, name=('Bias' +
                str(i))).astype(config.floatX))

        self.momentum = theano.shared(kwargs.get('momentum', 0)).astype(config.floatX)

        X = T.matrix('input')
        y = T.matrix('output')
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

        self.alpha = theano.shared(kwargs.get('alpha', 0.01)).astype(config.floatX)
        g_b = []
        g_w = []
        updates = []
        if rprop:
            for (w, bias, deltaw, deltab, prevw, prevb) in zip(self.W, self.b, self.deltaw,
                    self.deltab, self.prevStepw, self.prevStepb):
                g_w.append(T.grad(self.error, w))
                g_b.append(T.grad(self.error, bias))
                #Array describing which values are when gradients are both positive or both negative
                simW = T.neq((T.eq((prevw > 0), (g_w[-1] > 0))), (T.eq((prevw < 0), (g_w[-1] <
                    0))))

                #Array describing which values are when gradients are in opposite directions
                diffW = ((prevw > 0) ^ (g_w[-1] > 0)) * (T.neq(prevw, 0) * T.neq(g_w[-1], 0))

                updates.append((w, w - (T.sgn(g_w[-1]) * deltaw * (T.eq(diffW, 0)))))
                updates.append((deltaw, T.switch(diffW, deltaw *
                    0.5, T.switch(simW, deltaw * 1.2, deltaw))))

                #Array describing which values are when gradients are both positive or both negative
                simB = T.neq((T.eq((prevb > 0), (g_b[-1] > 0))), (T.eq((prevb < 0), (g_b[-1] <
                    0))))

                #Array describing which values are when gradients are in opposite directions
                diffB = ((prevb > 0) ^ (g_b[-1] > 0)) * (T.neq(prevb, 0) * T.neq(g_b[-1], 0))

                updates.append((bias, bias - (T.sgn(g_b[-1]) * deltab * (T.eq(diffB, 0)))))
                updates.append((deltab, T.switch(diffB, deltab *
                    0.5, T.switch(simB, deltab * 1.2, deltab))))

                updates.append((prevb, (T.sgn(g_b[-1]) * deltab * (T.eq(diffB, 0)))))
                updates.append((prevw, (T.sgn(g_w[-1]) * deltaw * (T.eq(diffW, 0)))))
        else:
            for (w, bias, mw, mb) in zip(self.W, self.b, self.momentumw, self.momentumb):
                g_w.append(T.grad(self.error, w))
                g_b.append(T.grad(self.error, bias))
                updates.append((mb, self.momentum * mb + self.alpha * g_b[-1]))
                updates.append((mw, self.momentum * mw + self.alpha * g_w[-1]))
                updates.append((w, w - mw))
                updates.append((bias, bias - mb))
        weight_grad = theano.printing.Print('Intermediate weight gradient')(g_w[-1])
        bias_grad = theano.printing.Print('Intermediate bias gradient')(g_b[-1])
        self.getGrad= theano.function([X, y], [weight_grad, bias_grad])
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

x, y = readMNISTData(1000)

xcv, ycv = readcv()

xor = np.array([[0,0],[0,1],[1,0],[1,1]])
yor = np.array([[0],[1],[1],[0]])

handwriting = FFNet(28**2, 300, 10, alpha=0.01, momentum=0.9, init_size=0.1, rprop=True)
#xornetwork = FFNet(2, 2, 1, alpha=0.1, init_values=1)

handwriting.converge(x, y, verbose=True, plot=True, maxError=1e-1)
print(percentError(handwriting, xcv, ycv))

i = 0
while True:
    image = xcv[i,:]
    image = image.reshape(28, 28)
    plt.imshow(image, cmap='Greys')
    plt.show()
    print('Computer predicts: ' + str(np.argmax(handwriting.predict(xcv[i,:].reshape(1, 28**2)))))
    print('Acutal: ' + str(np.argmax(ycv[i,:])))
    i += 1

import numpy as np
import matplotlib.pyplot as plt

YMIN = 0
YMAX = 1800
XMIN = 0
XMAX = 600
POINTS = 100

def plotFunction(eq):
    x = np.linspace(XMIN, XMAX, POINTS)
    y = eq(x)
    plt.plot(x, y)

def plotVectorField(derivative, slopeLength= (YMAX - YMIN)/POINTS):
    x = np.linspace(XMIN, XMAX, POINTS)
    y = np.linspace(YMIN, YMAX, POINTS)
    for i in x:
        for j in y:
            slope = derivative(i, j)
            def getLineLength():
                theta = np.arctan(slope)
                length = np.cos(theta) * slopeLength
                lineLength = np.linspace(i - length/2, i + length/2, 2)
                return lineLength
            lineLength = getLineLength()
            def fun(x1, y1):
                z = slope * (lineLength - x1) + y1
                return z
            plt.plot(lineLength, fun(i, j))

def EulersMethod(startX, startY, stepSize, derivative, endX):
    endY = startY
    Ys = [endY]
    for i in range(startX, endX + 1, stepSize):
        endY = endY + stepSize * derivative(i, endY)
        Ys.append(endY)
    return endY, Ys

def dxdy(x, y):
    return 0.0015 * (y) * (1 - (y/6000))

plotVectorField(dxdy)

plt.show()

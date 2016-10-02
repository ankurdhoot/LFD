import numpy as np
import random
import matplotlib.pyplot as plt
import time

class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y



def __classify(m, b, p):
    #classify Point b according to line y = mx + b
    pointOnLine = m * p.x + b
    if (pointOnLine > p.y):
        return -1
    else:
        return 1

def __reclassify(weights, featureMatrix):
    return np.sign(np.dot(weights, np.transpose(featureMatrix)))

def PLA(N, numRuns, debug=False):
    #TODO Generify to take arbitrary dimension, clean up plotting, get rid of Point class
    featureMatrix = np.zeros(shape=(N, 3))

    #generate random data
    for i in range(0, N):
        featureMatrix[i][0] = 1
        featureMatrix[i][1] = random.uniform(-1, 1)
        featureMatrix[i][2] = random.uniform(-1, 1)

    #generate random target function using two points
    p1 = Point(random.uniform(-1, 1), random.uniform(-1, 1))
    p2 = Point(random.uniform(-1, 1), random.uniform(-1, 1))

    m = float(p2.y - p1.y) / (p2.x - p1.x)
    b = p1.y - m * p1.x

    actualClass = np.zeros(N)
    predictedClass = np.zeros(N)
    w = np.zeros(3)
    for j in range(0, N):
        actualClass[j] = __classify(m, b, Point(featureMatrix[j][1], featureMatrix[j][2]))

    def f(x):
        #target function
        return m * x + b

    def h(x):
        #current hypothesis
        return -(w[0] + w[1]*x)/w[2]


    while not np.array_equal(actualClass, predictedClass):
        binVector = np.equal(actualClass, predictedClass)
        misclassifiedIDs = np.flatnonzero(binVector == False)
        randMisclassifiedPoint = random.choice(misclassifiedIDs)

        #update weight vector according to PLA rule
        w = w + actualClass[randMisclassifiedPoint] * featureMatrix[randMisclassifiedPoint]

        predictedClass = __reclassify(w, featureMatrix)

        if debug:
            #TODO show which point PLA was run on, color positive blue, negative red
            t1 = np.arange(-1, 1.05, .05)
            plt.axis([-1, 1, -1, 1])
            plt.plot(t1, f(t1), color='k')   #plot target in black
            plt.plot(t1, h(t1), color='c')   #plot hypothesis in cyan
            badPoints = featureMatrix[np.where(np.equal(actualClass, predictedClass) == False)]
            goodPoints = featureMatrix[np.where(np.equal(actualClass, predictedClass) == True)]
            plt.scatter(badPoints[:, 1], badPoints[:, 2], color='r')
            plt.scatter(goodPoints[:, 1], goodPoints[:, 2], color='b')
            plt.plot()
            plt.show()
            plt.close()







PLA(10, 1, True)
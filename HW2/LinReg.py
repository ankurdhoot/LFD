import numpy as np
import matplotlib.pyplot as plt
import random


class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

def classify(m, b, p):
    # classify Point b according to line y = mx + b
    return np.sign(p.y - (m * p.x + b))

def reclassify(weights, featureMatrix):
    return np.sign(np.dot(weights, np.transpose(featureMatrix)))

def LinReg(N, debug=False):
    #linear regression for classification
    featureMatrix = np.zeros(shape=(N, 3))

    for i in range(N):
        featureMatrix[i][0] = 1
        featureMatrix[i][1] = random.uniform(-1, 1)
        featureMatrix[i][2] = random.uniform(-1, 1)

    # generate random target function using two points
    p1 = Point(random.uniform(-1, 1), random.uniform(-1, 1))
    p2 = Point(random.uniform(-1, 1), random.uniform(-1, 1))

    m = float(p2.y - p1.y) / (p2.x - p1.x)
    b = p1.y - m * p1.x

    actualClass = np.zeros(N)
    predictedClass = np.zeros(N)
    w = np.zeros(3)
    for j in range(N):
        actualClass[j] = classify(m, b, Point(featureMatrix[j][1], featureMatrix[j][2]))

    w = np.dot(np.linalg.pinv(featureMatrix), actualClass)

    predictedClass = np.sign(np.dot(featureMatrix, w))
    Ein = float(sum(actualClass != predictedClass)) / N

    def f(x):
        #target function
        return m * x + b

    def h(x):
        #current hypothesis
        return -(w[0] + w[1]*x)/w[2]

    if debug:
        t1 = np.arange(-1, 1.05, .05)
        plt.axis([-1, 1, -1, 1])
        plt.plot(t1, f(t1), label='target', color='k')  # plot target in black
        plt.plot(t1, h(t1), label='hypothesis', color='c')  # plot hypothesis in cyan
        positivePoints = featureMatrix[np.where(actualClass == 1)]
        negativePoints = featureMatrix[np.where(actualClass == -1)]
        plt.scatter(positivePoints[:, 1], positivePoints[:, 2], color='b', s=50)
        plt.scatter(negativePoints[:, 1], negativePoints[:, 2], color='r', s=50)
        plt.legend(loc="lower left", numpoints=1)
        plt.show()
        plt.close()

    #calculate out of sample error
    testSize = 1000
    testMatrix = np.zeros(shape=(testSize, 3))
    testActualClass = np.zeros(testSize)
    testPredictedClass = np.zeros(testSize)
    for i in range(testSize):
        testMatrix[i][0] = 1
        testMatrix[i][1] = random.uniform(-1, 1)
        testMatrix[i][2] = random.uniform(-1, 1)

    for j in range(testSize):
        testActualClass[j] = classify(m, b, Point(testMatrix[j][1], testMatrix[j][2]))
        testPredictedClass[j] = np.sign(np.dot(testMatrix[j], w))

    Eout = float(sum(testActualClass != testPredictedClass)) / testSize

    #run PLA using initial linear regression weights
    iterations = 0  # track convergence of PLA on N data points
    while not np.array_equal(actualClass, predictedClass):
        binVector = np.equal(actualClass, predictedClass)
        misclassifiedIDs = np.flatnonzero(binVector == False)
        randMisclassifiedPoint = random.choice(misclassifiedIDs)

        # update weight vector according to PLA rule
        w = w + actualClass[randMisclassifiedPoint] * featureMatrix[randMisclassifiedPoint]

        predictedClass = reclassify(w, featureMatrix)

        iterations = iterations + 1

        if debug:
            t1 = np.arange(-1, 1.05, .05)
            plt.axis([-1, 1, -1, 1])
            plt.plot(t1, f(t1), label='target', color='k')  # plot target in black
            plt.plot(t1, h(t1), label='hypothesis', color='c')  # plot hypothesis in cyan
            positivePoints = featureMatrix[np.where(actualClass == 1)]
            negativePoints = featureMatrix[np.where(actualClass == -1)]
            plt.scatter(positivePoints[:, 1], positivePoints[:, 2], color='b', s=50)
            plt.scatter(negativePoints[:, 1], negativePoints[:, 2], color='r', s=50)
            plt.scatter(featureMatrix[randMisclassifiedPoint][1], featureMatrix[randMisclassifiedPoint][2],
                        label='PLA point', color='g', s=150)
            plt.legend(loc="lower left", numpoints=1)
            plt.show()
            plt.close()

    return Ein, Eout, iterations

def LinRegExperiment(N):
    runs = 1000
    EinTotal = 0
    EoutTotal = 0
    iterationsTotal = 0
    for i in range(runs):
        Ein, Eout, iterations = LinReg(N)
        EinTotal = EinTotal + Ein
        EoutTotal = EoutTotal + Eout
        iterationsTotal = iterationsTotal + iterations

    EinAvg = float(EinTotal) / runs
    EoutAvg = float(EoutTotal) / runs
    iterationsAvg = float(iterationsTotal) / runs
    return EinAvg, EoutAvg, iterationsAvg

#LinReg(10, debug=True)
print("Ein, Eout, PLA iterations=", LinRegExperiment(10))
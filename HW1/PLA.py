import numpy as np
import random
import matplotlib.pyplot as plt
import time

class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y



def classify(m, b, p):
    #classify Point b according to line y = mx + b
    return np.sign(p.y - (m * p.x + b))

def reclassify(weights, featureMatrix):
    return np.sign(np.dot(weights, np.transpose(featureMatrix)))

def PLA(N, debug=False):
    #TODO Generify to take arbitrary dimension, clean up plotting, get rid of Point class
    featureMatrix = np.zeros(shape=(N, 3))

    #generate random data
    for i in range(N):
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
    for j in range(N):
        actualClass[j] = classify(m, b, Point(featureMatrix[j][1], featureMatrix[j][2]))

    def f(x):
        #target function
        return m * x + b

    def h(x):
        #current hypothesis
        return -(w[0] + w[1]*x)/w[2]

    iterations = 0 #track convergence of PLA on N data points
    while not np.array_equal(actualClass, predictedClass):
        binVector = np.equal(actualClass, predictedClass)
        misclassifiedIDs = np.flatnonzero(binVector == False)
        randMisclassifiedPoint = random.choice(misclassifiedIDs)

        #update weight vector according to PLA rule
        w = w + actualClass[randMisclassifiedPoint] * featureMatrix[randMisclassifiedPoint]

        predictedClass = reclassify(w, featureMatrix)

        iterations = iterations + 1

        if debug:
            t1 = np.arange(-1, 1.05, .05)
            plt.axis([-1, 1, -1, 1])
            plt.plot(t1, f(t1), label='target', color='k')   #plot target in black
            plt.plot(t1, h(t1), label='hypothesis', color='c')   #plot hypothesis in cyan
            positivePoints = featureMatrix[np.where(actualClass == 1)]
            negativePoints = featureMatrix[np.where(actualClass == -1)]
            plt.scatter(positivePoints[:,1], positivePoints[:,2], color='b', s=50)
            plt.scatter(negativePoints[:,1], negativePoints[:,2], color='r', s=50)
            plt.scatter(featureMatrix[randMisclassifiedPoint][1], featureMatrix[randMisclassifiedPoint][2], label='PLA point', color='g', s=150)
            plt.legend(loc="lower left", numpoints=1)
            plt.show()
            plt.close()

    #calculate P(f(x) != g(x))
    numPoints = 1000
    numError = 0;
    for j in range(numPoints):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        trueClass = classify(m, b, Point(x, y))
        learnedClass = np.sign(w[0] + w[1]*x + w[2] * y)
        if trueClass != learnedClass:
            numError = numError + 1
    pError = float(numError)/numPoints

    return (iterations, pError)



def convergenceAndError(N, numRuns):
    totalIterations = 0
    totalError = 0
    for i in range(numRuns):
        (iterations, pError) = PLA(N)
        totalIterations = totalIterations + iterations
        totalError = totalError + pError
    return (float(totalIterations)/numRuns, float(totalError)/numRuns)


#PLA(27, debug=True)

(iterations, error) = convergenceAndError(10, 1000)
print("Average number of iterations taken to converge on 10 data points is", iterations)
print("Average error on 10 data points is", error)
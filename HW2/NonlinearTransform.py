import numpy as np
import random
from math import pow

def classify(x1, x2):
    return np.sign(pow(x1, 2) + pow(x2, 2) - .6)

def nonLinear(N, debug=False):
    # TODO add debugging plots, one using x1, x2; one using x1^2, x2^2
    # TODO refactor point generation to new method
    #target is sign(x1^2 + x2^2 - .6)

    featureMatrix = np.zeros(shape=(N, 3))

    for i in range(N):
        featureMatrix[i][0] = 1
        featureMatrix[i][1] = random.uniform(-1, 1)
        featureMatrix[i][2] = random.uniform(-1, 1)

    actualClass = np.zeros(N)
    predictedClass = np.zeros(N)
    for j in range(N):
        actualClass[j] = classify(featureMatrix[j][1], featureMatrix[j][2])

    #simulate noise by flipping sign of 10% of examples
    indicesToFlip = random.sample(range(N), int(N/10))

    for id in indicesToFlip:
        actualClass[id] = actualClass[id] * (-1)

    w = np.zeros(3)
    w = np.dot(np.linalg.pinv(featureMatrix), actualClass)

    predictedClass = np.sign(np.dot(featureMatrix, w))

    Ein = float(sum(predictedClass != actualClass)) / N

    #nonlinear transform
    featureMatrixNonlinear = np.zeros(shape=(N, 6))
    #(1, x1, x2, x1x2, x1^2, x2^2)
    for i in range(N):
        featureMatrixNonlinear[i][0] = featureMatrix[i][0]
        featureMatrixNonlinear[i][1] = featureMatrix[i][1]
        featureMatrixNonlinear[i][2] = featureMatrix[i][2]
        featureMatrixNonlinear[i][3] = featureMatrix[i][1] * featureMatrix[i][2]
        featureMatrixNonlinear[i][4] = featureMatrix[i][1] ** 2
        featureMatrixNonlinear[i][5] = featureMatrix[i][2] ** 2

    wTilde = np.zeros(6)
    wTilde = np.dot(np.linalg.pinv(featureMatrixNonlinear), actualClass)
    predictedClassNonlinear = np.zeros(N)
    predictedClassNonlinear = np.sign(np.dot(featureMatrixNonlinear, wTilde))

    EinNonlinear = float(sum(predictedClassNonlinear != actualClass)) / N

    #calculate Eout on wTilde hypothesis
    testSize = 1000
    testMatrix = np.zeros(shape=(testSize, 6))
    testActualClass = np.zeros(testSize)
    testPredictedClass = np.zeros(testSize)

    for i in range(testSize):
        testMatrix[i][0] = 1
        testMatrix[i][1] = random.uniform(-1, 1)
        testMatrix[i][2] = random.uniform(-1, 1)
        testMatrix[i][3] = testMatrix[i][1] * testMatrix[i][2]
        testMatrix[i][4] = testMatrix[i][1] ** 2
        testMatrix[i][5] = testMatrix[i][2] ** 2

    for j in range(testSize):
        testActualClass[j] = classify(testMatrix[j][1], testMatrix[j][2])

    # simulate noise by flipping sign of 10% of examples
    indicesToFlip = random.sample(range(testSize), int(testSize / 10))
    for id in indicesToFlip:
        testActualClass[id] = testActualClass[id] * (-1)

    testPredictedClass = np.sign(np.dot(testMatrix, wTilde))
    EoutNonlinear = float(sum(testActualClass != testPredictedClass)) / testSize

    return Ein, EinNonlinear, EoutNonlinear

def nonlinearExperiment(N):
    runs = 1000
    EinTotal, EinNonlinearTotal, EoutNonlinearTotal = 0, 0, 0
    for i in range(runs):
        Ein, EinNonlinear, EoutNonlinear = nonLinear(N)
        EinTotal += Ein
        EinNonlinearTotal += EinNonlinear
        EoutNonlinearTotal += EoutNonlinear

    EinAvg = float(EinTotal) / runs
    EinNonlinearAvg = float(EinNonlinearTotal) / runs
    EoutNonlinearAvg = float(EoutNonlinearTotal) / runs

    return EinAvg, EinNonlinearAvg, EoutNonlinearAvg


print("EinAvg, EinNonlinearAvg, EoutNonlinearAvg =", nonlinearExperiment(1000))
import random
import numpy as np

def flipCoins():
    #randomly flip 1000 coins 10 times each, report averages
    #use first coin, random coin, least frequent heads coin
    numCoins = 1000
    numFlips = 10
    array = np.zeros(numCoins)
    for i in range(numCoins):
        for j in range(numFlips):
            array[i] = array[i] + random.randint(0, 1)

    minFreqIndex = 0
    for i in range(numCoins):
        if array[i] < array[minFreqIndex]:
            minFreqIndex = i

    c1 = float(array[1]) / numFlips
    crand = float(array[random.randint(0, numCoins - 1)]) / numFlips
    cmin = float(array[minFreqIndex]) / numFlips

    return c1, crand, cmin

def HoeffdingExperiment():
    runs = 10000
    c1Total = 0
    crandTotal = 0
    cminTotal = 0
    for i in range(runs):
        (c1, crand, cmin) = flipCoins()
        c1Total = c1Total + c1
        crandTotal = crandTotal + crand
        cminTotal = cminTotal + cmin

    c1Avg = float(c1Total) / runs
    crandAvg = float(crandTotal) / runs
    cminAvg = float(cminTotal) / runs

    return c1Avg, crandAvg, cminAvg


print("c1, crand, cmin =", HoeffdingExperiment())

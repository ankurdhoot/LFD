import numpy as np
import random
import matplotlib.pyplot as plt
from math import sqrt, log, pow

def mH(N, dvc):
    # growth function which we'll approximate by N^(dvc)
    if (N > dvc):
        return pow(N, dvc)
    return pow(2, N)

def lgmH(N, dvc):
    # log growth function for computational feasibility
    if (N > dvc):
        return dvc * log(N)
    return N * log(2)
def VC(N, dvc, conf):
    # dvc is VC dimension
    # (1 - conf) is confidence required
    # N is number of training pts
    # epsilon is generalization error
    epsilon = sqrt(8/N * (log(4 / conf) + lgmH(2*N, dvc)))
    return epsilon

def Rademacher(N, dvc, conf):
    epsilon = sqrt(2/N * (log(2 * N) + lgmH(N, dvc))) + sqrt(2/N * log(1/conf)) + 1/N
    return epsilon

def Parrando(N, dvc, conf):
    epsilon = 1/N + sqrt(pow(1/N, 2) + 1/N * (log(6 / conf) + lgmH(2*N, dvc)))
    return epsilon

def Devroye(N, dvc, conf):
    epsilon = (1/N + sqrt(pow(1/N, 2) + 1/(2*N) * (log(4 / conf) + lgmH(pow(N,2), dvc)) * (1 - 2/N))) / (1 - 2/N)
    return epsilon

def getBounds(N, dvc, conf):
    vc = VC(N, dvc, conf)
    rademacher = Rademacher(N, dvc, conf)
    parrando = Parrando(N, dvc, conf)
    devroye = Devroye(N, dvc, conf)

    print(vc, rademacher, parrando, devroye)

getBounds(10000, 50, .05)
getBounds(5, 50, .05)

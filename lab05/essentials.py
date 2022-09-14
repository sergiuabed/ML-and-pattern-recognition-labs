import numpy
import math

def covarMat(X):
    mu = X.mean(axis=1)  # mean of the dataset
    mu = mu.reshape(mu.size, 1)
    # mu is suptracted from each column(sample) of X through broadcasting
    X_centered = X-mu

    covMat = numpy.dot(X_centered, X_centered.T) / X.shape[1]

    return covMat

def datasetMean(X):
    mu = X.mean(axis=1)
    mu = mu.reshape(mu.size, 1) # making it 2-dimensional so it can be used for broadcasting
    return mu

def logpdf_GAU_ND(X, mu, C):
    firstTerm = (-1)*numpy.log(2*math.pi)*0.5*C.shape[0]

    sign, detC = numpy.linalg.slogdet(C)

    secondTerm = (-1)*0.5*detC

    i = 0
    Y = 0  # just to define the scope of Y outside the loop
    while i < X.shape[1]:
        x = X[:, i].reshape(X.shape[0],1)
        # subtract the mean from the sample
        x_centered = x-mu.reshape(mu.size, 1)

        invC = numpy.linalg.inv(C)

        thirdTerm = numpy.dot(x_centered.T, invC)
        thirdTerm = numpy.dot(thirdTerm, x_centered)
        thirdTerm = (-1)*0.5*thirdTerm

        y = firstTerm + secondTerm + thirdTerm
        if i == 0:
            Y = y
        else:
            Y = numpy.hstack([Y, y])

        i += 1

    return Y

def pdf_GAU_ND(X, mu, C):
    Y = logpdf_GAU_ND(X, mu, C)
    return numpy.exp(Y)

def loglikelihood(X, mu, C):
    Y = logpdf_GAU_ND(X, mu, C)

    sum = Y.sum()

    return sum

def samples_of_class(X, L, l):
    Y = X[:, L == l]
    return Y

def accuracy(predicted, actual):
    check = predicted == actual
    return check.sum() / predicted.size
from more_itertools import first
import numpy
import matplotlib.pyplot as plt
import scipy.linalg
import math
import matplotlib.pyplot as plt


def load(datasetFile):
    list_vectors = []
    labels = []
    label_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    nr_lines = 0
    with open(datasetFile, 'r') as f:
        for line in f:
            sample = line.strip().split(",")
            measurements = numpy.array([float(i)
                                       for i in sample[0:-1]]).reshape(4, 1)
            label = sample[-1]
            label = label_dict[label]

            list_vectors.append(measurements)
            labels.append(label)
            nr_lines += 1

    array_lables = numpy.array(labels)
    data_matrix = numpy.hstack(list_vectors)

    return (data_matrix, array_lables)


def covarMat(X):
    mu = X.mean(axis=1)  # mean of the dataset
    mu = mu.reshape(mu.size, 1)
    # mu is suptracted from each column(sample) of X through broadcasting
    X_centered = X-mu

    covMat = numpy.dot(X_centered, X_centered.T) / X.shape[1]

    return covMat


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

def loglikelihood(X, mu, C):
    Y = logpdf_GAU_ND(X, mu, C)

    sum = Y.sum()

    return sum

if __name__ == "__main__":
    #dataMat, labelArr=load('iris.csv')

    #plt.figure()
    #XPlot = numpy.linspace(-8, 12, 1000)
    #m = numpy.ones((1, 1)) * 1.0
    #C = numpy.ones((1, 1)) * 2.0
    #plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    #plt.show()

    XND = numpy.load('Solution/XND.npy')
    mu = numpy.load('Solution/muND.npy')
    C = numpy.load('Solution/CND.npy')
    pdfSol = numpy.load('Solution/llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(numpy.abs(pdfSol - pdfGau).max())
    print()
    print(pdfGau)
    print(pdfSol)
    print()
    
    # dataset XND
    mu = XND.mean(axis=1)
    C = covarMat(XND)
    ll = loglikelihood(XND, mu, C)
    print(mu)
    print(C)
    print(ll)
    print()

    # dataset X1D
    X1D = numpy.load('Solution/X1D.npy')
    mu = X1D.mean(axis=1)
    C = covarMat(X1D)
    ll = loglikelihood(X1D, mu, C)
    print(mu)
    print(C)
    print(ll)

    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(XPlot.reshape((1, XPlot.size)), mu, C)).ravel())
    plt.show()

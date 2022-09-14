import sklearn.datasets
import numpy
import scipy
import essentials as ess

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L


def split_db_2to1(D, L, seed=0):

    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def MVG_parameters(DTR, LTR):
    labels = numpy.unique(LTR)
    means = {}
    covariances = {}

    for l in labels:
        X = ess.samples_of_class(DTR, LTR, l)
        mean = ess.datasetMean(X)   # "broadcasting ready"
        cov = ess.covarMat(X)

        means[l] = mean
        covariances[l] = cov
    return [means, covariances], labels

def MVG_classifier(means, covariances, labels, priors, D):    
    S = []
    p = []
    i = 0
    for l in labels:
        likelihoods = ess.pdf_GAU_ND(D, means[l], covariances[l])
        if i == 0:
            S = likelihoods
            p = priors[l]
            i = 1
        else:
            S = numpy.vstack([S, likelihoods])
            p = numpy.vstack([p, priors[l]])

    SJoint = S*p
    SMarginal = SJoint.sum(0).reshape(1, SJoint.shape[1])
    SPost = SJoint / SMarginal

    prediction = SPost.argmax(axis=0)
    return prediction

def MVG_classifier_logDomain(means, covariances, labels, priors, D):
    logS = []
    logp = []
    i = 0
    for l in labels:
        loglikelihoods = ess.logpdf_GAU_ND(D, means[l], covariances[l])
        if i == 0:
            logS = loglikelihoods
            logp = numpy.log(priors[l])
            i = 1
        else:
            logS = numpy.vstack([logS, loglikelihoods])
            logp = numpy.vstack([logp, numpy.log(priors[l])])

    logSJoint = logS + logp
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0).reshape(1, logSJoint.shape[1])
    logSPost = logSJoint - logSMarginal
    SPost = numpy.exp(logSPost)
    prediction = SPost.argmax(axis=0)
    return prediction

if __name__ == '__main__':
    [D, L] = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    [means, covariances], labels = MVG_parameters(DTR, LTR)

    prediction = MVG_classifier(means, covariances, labels, {0: 1/3, 1: 1/3, 2: 1/3}, DTE)
    SPost_MVG = numpy.load('./solutions/Posterior_MVG.npy')

    acc = ess.accuracy(prediction, LTE)
    print('Accuracy of non log-domain model ' + str(acc))

    prediction = MVG_classifier_logDomain(means, covariances, labels, {0: 1/3, 1: 1/3, 2: 1/3}, DTE)
    logPosterior_MVG= numpy.load('./solutions/logPosterior_MVG.npy')

    acc = ess.accuracy(prediction, LTE)
    print('Accuracy of log-domain model ' + str(acc))

    #print(SPost)
    #print()
    #print(SPost_fromLog)
    #print()
    #diff = SPost-SPost_fromLog
    ##print(diff)
    #print(abs(diff) < 0.00001)




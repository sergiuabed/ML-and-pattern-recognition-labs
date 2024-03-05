import numpy as np
from data.GMM_load import load_gmm
from utils import logpdf_GMM, logpdf_GAU_ND, gmm_estimation, lbg_algorithm, covarMat, datasetMean, gmm_classifier_train, gmm_classifier_wrap
import matplotlib.pyplot as plt
import sklearn.datasets

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L


def split_db_2to1(D, L, seed=0):

    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def visualize(X, gmm):
    sortedx = np.sort(X)
    #yvalues = np.zeros(len(sortedx))
    lls = logpdf_GMM(sortedx, gmm)
    yvalues = np.exp(lls).reshape((lls.size,))

    #for g in range(len(gmm)):
    #    yvalues += np.exp(lls[g])

    plt.title("Final GMM Results")
    plt.xlabel("X Value")
    plt.ylabel("Density")
    plt.hist(X.reshape((X.size,)), bins=100, density=True, color='grey',
             edgecolor='black', linewidth=1.5, alpha=0.5)
    plt.plot(sortedx.reshape((sortedx.size,)), yvalues, linewidth=2)
    plt.show()

def test_logpdf_GMM():
    X = np.load('data/GMM_data_4D.npy')
    correct_logdens = np.load('data/GMM_4D_3G_init_ll.npy')
    
    gmms = load_gmm('data/GMM_4D_3G_init.json')
    logdens = logpdf_GMM(X, gmms)

    #print("My logdens")
    #print(correct_logdens)
    #print()
    #print("Professor's logdens")
    #print(logdens)

    print(logdens.shape)
    print(logdens -correct_logdens)

def test_EM_algorithm():
    X = np.load('data/GMM_data_4D.npy')
    gmm = load_gmm('data/GMM_4D_3G_init.json')
    correct_final_gmm = load_gmm('data/GMM_4D_3G_EM.json')

    print(gmm[0][1].shape)

    new_gmm, avg_llr = gmm_estimation(X, gmm)

    print(f"Avg llr: {avg_llr}")

def test_lbg_algorithm():
    X = np.load('data/GMM_data_4D.npy')
    correct_gmm = load_gmm('data/GMM_4D_4G_EM_LBG.json')

    mean = datasetMean(X)
    covar = covarMat(X)

    num_components = 4
    gmm, avg_llh = lbg_algorithm(X, [(1, mean, covar)], 0.1, num_components, mode='diag')

    print(f"Obtained gmm weights: {[comp[0] for comp in gmm]}")
    print(f"Correct gmm weights: {[comp[0] for comp in correct_gmm]}")


    print(f"Obtained avg_llh: {avg_llh}")

    #visualize(X, gmm)

def accuracy(predicted, actual):
    check = predicted == actual
    return check.sum() / predicted.size

def test_classifier():
    data, labels = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(data, labels)

    gmm_params = gmm_classifier_train(DTR, LTR, 16, 'tied')

    unique_labels = np.unique(LTE)
    iris_priors = {l: 1/3 for l in unique_labels}

    gmm_classifier = gmm_classifier_wrap(gmm_params, unique_labels, iris_priors)

    print(gmm_params)
    print(f"Error rate: {1 - accuracy(gmm_classifier(DTE), LTE)}")
    print(unique_labels)

def something():
    data, labels = load_iris()
    X = np.load('data/GMM_data_4D.npy')

    print(data.shape)
    print(X.shape)

if __name__ == '__main__':
    #test_logpdf_GMM()
    #test_EM_algorithm()
    #test_lbg_algorithm()
    test_classifier()

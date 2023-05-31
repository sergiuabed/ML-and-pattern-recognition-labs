from sklearn.datasets import load_iris
import numpy as np

def load_iris_binary():
    #D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D, L = load_iris()['data'].T, load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
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

def svm_dual_obj_wrap(DTR, LTR, K):
    '''
    this is actually the negative of the svm dual objective. It's done this way because
    we need to look for the maximum and scipy.optimize.fmin_l_bfgs_b is a minimizer
    '''

    embedding = K*np.ones((1, DTR.shape[1]), dtype=float)
    D = np.vstack((DTR, embedding))
    zh = np.array(LTR).reshape(1,D.shape[1])
    zv = np.array(LTR).reshape(D.shape[1],1)

    G = np.dot(D.T, D)

    H = G*zh
    H = H*zv

    def svm_dual_obj(_alphas):
        L = 0.5*_alphas.T.dot(H).dot(_alphas) - _alphas.T.dot(np.ones((_alphas.shape)))
        grad_l = H.dot(_alphas) - 1
        grad_l = grad_l.reshape((grad_l.size,))

        return (L.item(), grad_l)
    
    return svm_dual_obj

def svm_primal_obj_wrap(DTR, LTR, K, C):
    embedding = K*np.ones((1, DTR.shape[1]), dtype=float)
    D = np.vstack((DTR, embedding))
    z = np.array(LTR).reshape(1, D.shape[1])

    def svm_primal_obj(w_b):
        w_b = w_b.reshape(w_b.size, 1)
        P = w_b.T.dot(D)
        P = 1 - z * P
        P = C*np.maximum(P, 0).sum()
        J = 0.5 * w_b.T.dot(w_b) + P

        return J.item()
    
    return svm_primal_obj

def svm_dual_classifier_wrap(_alphas, DTR, LTR, K):
    embedding = K*np.ones((1, DTR.shape[1]), dtype=float)
    D = np.vstack((DTR, embedding))
    
    z = np.array(LTR).reshape(D.shape[1],1)

    w_b = D.dot(z*_alphas)

    def svm_classifier(x):
        w = w_b[0:-1]
        b = w_b[-1]*K

        s = w.T.dot(x) + b

        preds = np.zeros(s.shape)
        preds[s > 0] = 1
        preds[s <= 0] = -1

        return preds.reshape(preds.size,)
    
    return svm_classifier, w_b

def svm_primal_classifier_wrap(w_b, K):
    def svm_classifier(x):
        w = w_b[0:-1]
        b = w_b[-1]*K

        s = w.T.dot(x) + b

        preds = np.zeros(s.shape)
        preds[s > 0] = 1
        preds[s <= 0] = -1

        return preds.reshape(preds.size,)
    
    return svm_classifier, w_b

def svm_accuracy(classifier, DTE, LTE):
    preds = classifier(DTE)

    correct_preds = np.zeros(preds.shape)
    correct_preds[preds == LTE] = 1

    return np.sum(correct_preds)/correct_preds.size

def duality_gap(primal_obj, dual_obj, w_b, _alphas):
    return primal_obj(w_b) + dual_obj(_alphas)[0]


import numpy as np
import scipy.optimize
import sklearn
from sklearn.datasets import load_iris

def func(x):
    if x.shape != (2,):
        raise Exception("Incompatible input shape! Input should have (2,)")
    
    val = (x[0] + 3)**2 + np.sin(x[0]) + (x[1] + 1)**2

    return val

def func_withGrad(x):
    if x.shape != (2,):
        raise Exception("Incompatible input shape! Input should have shape (2,)")
    
    val = (x[0] + 3)**2 + np.sin(x[0]) + (x[1] + 1)**2

    grad_y = 2*(x[0] + 3) + np.cos(x[0])
    grad_z = 2*(x[1] + 1)

    grad = np.array([grad_y, grad_z])

    return val, grad

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

def load_iris_binary():
    #D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D, L = load_iris()['data'].T, load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def logreg_obj_wrap(DTR, LTR, _lambda):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        w = w.reshape(w.size, 1)

        z = np.array(LTR)
        z[z==0] = -1
        z = z.reshape(1, LTR.size)

        exp = np.dot(w.T, DTR) + b
        exp = -exp * z

        terms = np.logaddexp(np.zeros(DTR.shape[1]), exp)
        sum = terms.sum()
        
        reg = float(_lambda)/2 * np.dot(w.T, w)

        result = reg + sum/DTR.shape[1]

        return result.reshape(result.size,)

    return logreg_obj

def logistic_regression_wrap(w, b):
    def logistic_regression_classifier(x):
        s = np.dot(w.T, x) + b

        preds = np.zeros(s.shape)
        preds[s > 0] = 1

        return preds.reshape(preds.size,)
    
    return logistic_regression_classifier

def logreg_accuracy(classifier, DTE, LTE):
    preds = classifier(DTE)

    correct_preds = np.zeros(preds.shape)
    correct_preds[preds == LTE] = 1

    return np.sum(correct_preds)/correct_preds.size

if __name__ == "__main__":
    inp = np.array([1,2])
    print(func(inp))

    
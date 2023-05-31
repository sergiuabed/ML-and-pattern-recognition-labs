import numpy as np
import scipy.optimize
from utils import load_iris_binary, split_db_2to1, svm_dual_obj_wrap, svm_accuracy, svm_dual_classifier_wrap, duality_gap, svm_primal_obj_wrap, svm_primal_classifier_wrap

def linear_svm_exercise():
    D, L = load_iris_binary()
    L[L==0] = -1
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    x0 = np.zeros((DTR.shape[1]), dtype=float)

    C = [0.1, 1.0, 10.0]
    K = [1, 10]

    for k in K:
        for c in C:
            # dual optimization
            svm_dual_obj = svm_dual_obj_wrap(DTR, LTR, k)
            solution = scipy.optimize.fmin_l_bfgs_b(svm_dual_obj, x0=x0, bounds=[(0, c) for _ in range(DTR.shape[1])], factr=1.0)

            _alphas = np.array(solution[0]).reshape(len(solution[0]), 1)
            loss_dual = solution[1]
            classifier_dual, w_b1 = svm_dual_classifier_wrap(_alphas, DTR, LTR, k)

            error_rate = 1-svm_accuracy(classifier_dual, DTE, LTE)

            #primal optimization
            svm_primal_obj = svm_primal_obj_wrap(DTR, LTR, k, c)
            solution = scipy.optimize.fmin_l_bfgs_b(svm_primal_obj, approx_grad=True, x0=np.zeros((DTR.shape[0]+1)), factr=1.0)

            w_b = np.array(solution[0]).reshape(len(solution[0]), 1)
            loss_primal = solution[1]

            gap = duality_gap(svm_primal_obj, svm_dual_obj, w_b, _alphas)

            print(f"K: {k}")
            print(f"    C: {c}")
            #print(f"        Value of the objective at the minimum: {solution[1]}")
            print(f"        Error rate: {error_rate*100}%\n")
            print(f"        Primal loss: {'{:e}'.format(loss_primal)}")
            print(f"        Dual loss: {'{:e}'.format(-loss_dual)}")
            print(f"        Duality gap: {'{:e}'.format(gap)}")

if __name__ == "__main__":
    linear_svm_exercise()
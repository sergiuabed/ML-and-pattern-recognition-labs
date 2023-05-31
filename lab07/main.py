import numpy as np
import scipy.optimize
from utils import func, func_withGrad, load_iris_binary, split_db_2to1, logreg_obj_wrap, logistic_regression_wrap, logreg_accuracy

def numerical_optimization_exercise():
    print("Test optimizer with numerical approximation of the gradient")
    res = scipy.optimize.fmin_l_bfgs_b(func,  x0=np.array([0, 0]), approx_grad=True, iprint=1)

    print(f"Estimated minimum: {res[0]}")
    print(f"Function value at estimated minimum: {res[1]}")
    print(f"Function calls executed: {res[2]['funcalls']}")

    print("\nTest optimizer given gradient expression")
    res = scipy.optimize.fmin_l_bfgs_b(func_withGrad,  x0=np.array([0, 0]), iprint=1)

    print(f"Estimated minimum: {res[0]}")
    print(f"Function value at estimated minimum: {res[1]}")
    print(f"Function calls executed: {res[2]['funcalls']}")

def binary_logistic_regression_exercise():
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    x0 = np.zeros(DTE.shape[0] + 1)

    lambdas = [1e-6, 1e-3, 1e-1, 1.0]

    for _lambda in lambdas:
        logreg_obj = logreg_obj_wrap(DTR, LTR, _lambda)
        solution = scipy.optimize.fmin_l_bfgs_b(logreg_obj,  x0=x0, approx_grad=True)

        w = np.array(solution[0][0:-1]).reshape(len(solution[0][0:-1]), 1)
        b = solution[0][-1]
        classifier = logistic_regression_wrap(w, b)

        error_rate = 1-logreg_accuracy(classifier, DTE, LTE)

        print(f"Lambda: {_lambda}")
        print(f"    Value of the objective at the minimum: {solution[1]}")
        print(f"    Error rate: {error_rate*100}%\n")

if __name__ == "__main__":
    #numerical_optimization_exercise()
    binary_logistic_regression_exercise()
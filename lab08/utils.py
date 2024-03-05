import numpy as np
import matplotlib.pyplot as plt

def get_confusion_matrix_old(loglikelihoods, priors, costs, labels):
    joint_loglikel = loglikelihoods * priors # class conditional log-likelihood
    prod = costs.dot(joint_loglikel)

    predicted_labels = prod.argmin(axis=0)  # argmin because we want to select the minimum cost within each column
                                            # argmax is used when selecting based on highest posterior probability

    confusion_matrix = np.zeros((3,3))
    for i in range(loglikelihoods.shape[0]):

        for j in range(loglikelihoods.shape[0]):
            match = predicted_labels[labels == j] # labels==j -> select items on indices corresponding to samples belonging to class j
            confusion_matrix[i, j] = sum((match == i).astype(int))

    return confusion_matrix

def get_confusion_matrix(predicted_labels, labels, nr_classes):
    confusion_matrix = np.zeros((nr_classes, nr_classes))
    for i in range(nr_classes):
        for j in range(nr_classes):
            match = predicted_labels[labels == j] # labels==j -> select items on indices corresponding to samples belonging to class j
            confusion_matrix[i, j] = sum((match == i).astype(int))

    return confusion_matrix

def binary_optimal_bayes_decision(llr, working_point):
    '''
    This assigns labels using the theoretical optimal threshold
    '''
    prior1, C_fn, C_fp = working_point

    prior0 = 1 - prior1
    th = -np.log(prior1 * C_fn / (prior0 * C_fp))

    predicted_labels = np.zeros(llr.shape)
    predicted_labels[llr > th] = 1

    return predicted_labels

def bayes_risk(cm, working_point):
    prior1, C_fn, C_fp = working_point
    prior0 = 1 - prior1

    FNR = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    FPR = cm[1, 0] / (cm[1, 0] + cm[0, 0])

    DCF_u = prior1 * C_fn * FNR + prior0 * C_fp * FPR
    DCF_n = DCF_u / min(prior1 * C_fn, prior0 * C_fp)

    return (DCF_u, DCF_n)

def minimum_bayes_risk(llr, labels, working_point):

    ths = np.linspace(start=min(llr), stop=max(llr), num=7000)
    minDCF = float("inf")
    for th in ths:
        predicted_labels = np.zeros(llr.shape)
        predicted_labels[llr > th] = 1

        cm = get_confusion_matrix(predicted_labels, labels, 2)

        DCF_u, DCF_n = bayes_risk(cm, working_point)
        #print(f"    {DCF_n}")
        
        if minDCF > DCF_n:
            minDCF = DCF_n
    
    return minDCF

def plot_roc(llr, labels):
    ths = np.linspace(start=min(llr), stop=max(llr), num=700)
    TPRs = []
    FPRs = []
    for th in ths:
        predicted_labels = np.zeros(llr.shape)
        predicted_labels[llr > th] = 1

        cm = get_confusion_matrix(predicted_labels, labels, 2)
        FNR = cm[0, 1] / (cm[0, 1] + cm[1, 1])
        FPR = cm[1, 0] / (cm[1, 0] + cm[0, 0])

        TPR = 1 - FNR

        TPRs.append(TPR)
        FPRs.append(FPR)

    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(FPRs, TPRs)
    plt.grid()
    plt.show()

def bayes_error_plot(llr, correct_labels):

    p_vals = np.linspace(-3, 3, 21)
    dcf = []
    min_dcf = []
    for p in p_vals:
        eff_prior = 1 / (1 + np.exp(-p))

        pl = binary_optimal_bayes_decision(llr, (eff_prior, 1, 1))
        cm = get_confusion_matrix(pl, correct_labels, 2)

        _, DCF_n = bayes_risk(cm, (eff_prior, 1, 1))
        minDCF = minimum_bayes_risk(llr, correct_labels, (eff_prior, 1, 1))

        dcf.append(DCF_n)
        min_dcf.append(minDCF)

    plt.plot(p_vals, dcf, label='DCF', color='r')
    plt.plot(p_vals, min_dcf, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()

#    priors = np.array([1-prior1, prior1])
#
#    C = np.zeros((2,2))
#    C[0, 1] = C_fn
#    C[1, 0] = C_fp


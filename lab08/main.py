import numpy as np
from utils import get_confusion_matrix, binary_optimal_bayes_decision, bayes_risk, minimum_bayes_risk, plot_roc, bayes_error_plot

def main():

    loglikelihoods = np.load('data/commedia_ll.npy')
    labels = np.load('data/commedia_labels.npy')
    priors = np.array([1/3, 1/3, 1/3]).reshape((3,1))
    costs = np.ones((3,3))
    np.fill_diagonal(costs, 0)

    print(f"loglikelihood shape{loglikelihoods.shape}")
    print(f"labels shape{labels.shape}")

    joint_loglikel = loglikelihoods + np.log(priors) # class conditional log-likelihood
    prod = costs.dot(joint_loglikel)

    predicted_labels = prod.argmin(axis=0)  # argmin because we want to select the minimum cost within each column
                                            # argmax is used when selecting based on highest posterior probability

    print(get_confusion_matrix(predicted_labels, labels, 3))

def exercise_binary_task_optimal_decisions():
    llr = np.load('data/commedia_llr_infpar.npy')
    labels = np.load('data/commedia_labels_infpar.npy')

    working_points = [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]

    for wp in working_points:
        pl = binary_optimal_bayes_decision(llr, labels, wp)
        
        print(get_confusion_matrix(pl, labels, 2))
        print()

def exercise_binary_task_evaluation():
    llr = np.load('data/commedia_llr_infpar.npy')
    labels = np.load('data/commedia_labels_infpar.npy')

    working_points = [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]

    print("prior1, Cfn, Cfp|    DCF_u, DCF_n")
    for wp in working_points:
        pl = binary_optimal_bayes_decision(llr, wp)
        
        cm = get_confusion_matrix(pl, labels, 2)
        DCF_u, DCF_n = bayes_risk(cm, wp)
        print(f"({wp[0]}, {wp[1]}, {wp[2]})|    {DCF_u}, {DCF_n}")

def exercise_binary_task_evaluation_minDCF():
    llr = np.load('data/commedia_llr_infpar.npy')
    labels = np.load('data/commedia_labels_infpar.npy')

    working_points = [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]

    print("prior1, Cfn, Cfp|    DCF_n")
    for wp in working_points:
        DCF_n = minimum_bayes_risk(llr, labels, wp)

        print(f"({wp[0]}, {wp[1]}, {wp[2]})|    {DCF_n}")

def exercise_plot_roc():
    llr = np.load('data/commedia_llr_infpar.npy')
    labels = np.load('data/commedia_labels_infpar.npy')

    plot_roc(llr, labels)

def exercise_bayes_plot():
    llr = np.load('data/commedia_llr_infpar.npy')
    labels = np.load('data/commedia_labels_infpar.npy')

    bayes_error_plot(llr, labels)


if __name__ == '__main__':
    #main()
    #exercise_binary_task_optimal_decisions()
    #exercise_binary_task_evaluation()
    #exercise_binary_task_evaluation_minDCF()
    #exercise_plot_roc()
    exercise_bayes_plot()
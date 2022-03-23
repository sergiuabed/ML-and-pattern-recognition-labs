import sys
import numpy
import matplotlib.pyplot as plt
import scipy.linalg

def load(datasetFile):
    list_vectors=[]
    labels=[]
    label_dict={'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    nr_lines=0
    with open(datasetFile, 'r') as f:
        for line in f:
            sample=line.strip().split(",")
            measurements=numpy.array([float(i) for i in sample[0:-1]]).reshape(4,1)
            label=sample[-1]
            label=label_dict[label]

            list_vectors.append(measurements)
            labels.append(label)
            nr_lines+=1

    array_lables=numpy.array(labels)
    data_matrix=numpy.hstack(list_vectors)

    return (data_matrix, array_lables)

def PCA_matrix(dataMat, m):
    mu=dataMat.mean(1)
    dataMatCentered=dataMat-mu.reshape(mu.size,1) #through broadcasting, mu is subracted from each column of dataMatCentered

    N=dataMatCentered.shape[1]
    covarianceMat=numpy.dot(dataMatCentered,dataMatCentered.T)/N
    s, U=numpy.linalg.eigh(covarianceMat)
    P=U[:, ::-1][:, 0:m]

    return P

def LDA_matrix(dataMat, labels, m):
    data_class=[]   #list of matrices, each matrix having samples of the same label as columns
    mean_class=[]

    mean=dataMat.mean(1)
    mean=mean.reshape(mean.size, 1)
    Sb=numpy.zeros((mean.shape[0], mean.shape[0]), dtype=numpy.float32)
    Sw=numpy.zeros((mean.shape[0], mean.shape[0]), dtype=numpy.float32)

    for i in range(0,3):
        data_labeled=dataMat[:, labels==i]
        
        mean_labeled=data_labeled.mean(1)
        mean_labeled=mean_labeled.reshape(mean_labeled.size, 1) #reshape the mean as a (2D) column vector

        nc=data_labeled.shape[1]    #nr of samples in the current class
        N=dataMat.shape[1]

        #compute between class covariance matrix Sb
        e=mean_labeled-mean
        term=numpy.dot(e, e.T)
        term=term*nc
        term=term/N
        Sb=Sb+term

        #compute within class covariance matrix Sw
        data_labeled_centered=data_labeled-mean_labeled     #recall: mean_labeled is already shaped as a column vector, so broadcasting occurs as wanted
        covariance_mat=numpy.dot(data_labeled_centered, data_labeled_centered.T)
        covariance_mat=covariance_mat/N
        Sw=Sw+covariance_mat

    s, U=scipy.linalg.eigh(Sb, Sw)
    W=U[:, ::-1][:, 0:m]

    return W

def plot_scatter(mat, lab, m):

    mat_setosa=mat[:, lab==0]
    mat_versicolor=mat[:, lab==1]
    mat_virginica= mat[:, lab==2]

    for i in range(m):
        for j in range(i+1, m):
            plt.figure()
            plt.xlabel("dimension "+str(i))
            plt.ylabel("dimension "+str(j))
            plt.scatter(mat_setosa[i, :], mat_setosa[j, :], label = 'Setosa')
            plt.scatter(mat_versicolor[i, :], mat_versicolor[j, :], label = 'Versicolor')
            plt.scatter(mat_virginica[i, :], mat_virginica[j, :], label = 'Virginica')

            plt.legend()
            #plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()

if __name__=='__main__':
    mat, lab=load('iris.csv')
    m=2

    P=PCA_matrix(mat, m)
    print(P)
    projectedDataMat=numpy.dot(P.T, mat)
    plot_scatter(projectedDataMat, lab, m)

    W=LDA_matrix(mat, lab, m)
    print(W)
    projectedDataMatLDA=numpy.dot(W.T, mat)
    plot_scatter(projectedDataMatLDA, lab, m)



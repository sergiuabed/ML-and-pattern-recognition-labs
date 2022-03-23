import sys
import numpy
import matplotlib.pyplot as plt

def load(datasetFile):
    list_vectors=[]
    labels=[]
    nr_lines=0
    with open(datasetFile, 'r') as f:
        for line in f:
            sample=line.strip().split(",")
            measurements=numpy.array([float(i) for i in sample[0:-1]]).reshape(4,1)
            label=sample[-1]

            list_vectors.append(measurements)
            labels.append(label)
            nr_lines+=1

    array_lables=numpy.array(labels)
    data_matrix=numpy.hstack(list_vectors)

    return (data_matrix, array_lables)


def plot_hist(mat, lab):
    mat_setosa=mat[:, lab=='Iris-setosa']
    mat_versicolor=mat[:, lab=='Iris-versicolor']
    mat_virginica= mat[:, lab=='Iris-virginica']

    attributeDict = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }

    for i in range(4):
        plt.figure()
        plt.xlabel(attributeDict[i])
        plt.hist(mat_setosa[i, :], bins = 10, density = True, alpha = 0.4, label = 'Setosa')
        plt.hist(mat_versicolor[i, :], bins = 10, density = True, alpha = 0.4, label = 'Versicolor')
        plt.hist(mat_virginica[i, :], bins = 10, density = True, alpha = 0.4, label = 'Virginica')
        
        plt.legend()
        #plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()


def plot_scatter(mat, lab):
    mat_setosa=mat[:, lab=='Iris-setosa']
    mat_versicolor=mat[:, lab=='Iris-versicolor']
    mat_virginica= mat[:, lab=='Iris-virginica']

    attributeDict = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }

    for i in range(4):
        for j in range(i+1, 4):
            plt.figure()
            plt.xlabel(attributeDict[i])
            plt.ylabel(attributeDict[j])
            plt.scatter(mat_setosa[i, :], mat_setosa[j, :], label = 'Setosa')
            plt.scatter(mat_versicolor[i, :], mat_versicolor[j, :], label = 'Versicolor')
            plt.scatter(mat_virginica[i, :], mat_virginica[j, :], label = 'Virginica')

            plt.legend()
            #plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()

if __name__=='__main__':
    filename="iris.csv"
    mat, lab=load(filename)
    plot_hist(mat, lab)
    plot_scatter(mat, lab)

#print(mat)

#Histograms
#sepal length

#setosa_index=(lab=="Iris-setosa")
#setosa_sample=mat[:, setosa_index]
#setosaSepalLength=setosa_sample[0, :]
#
#plt.figure()
#plt.hist(setosaSepalLength, bins=10, density=True, ec='black')
#plt.show()

#sepal width

#petal lenght

#petal width


            
import numpy
import scipy.special
from data.load import load_data
from data.load import split_data

# cantiche is a list of lists of tercets, each list corresponding to a cantica
def createDictionary(cantiche):     # dictionary has as keys the words and as value integers in the interval [0, M-1]; M = nr of words w/o repetitions
    dict = {}                   # this dictionary is useful when working with the matrix of occurrences
    nrWords = 0
    for text in cantiche:
        for t in text:
            words = t.split()
            for w in words:
                if w not in dict:
                    dict[w] = nrWords
                    nrWords += 1
    
    return dict

def computeOccurrences(text, pseudo, dict):   # computes occurrences within a class (cantica)
    
    occ = numpy.zeros([1, len(dict)])    # 2-dim array of occurrences
    nrWords = 0

    for t in text:
        words = t.split()
        for w in words:
            if w in dict:   # this if statement is relevant for computing the occurrences of words for test tercets
                occ[0][dict[w]] += 1
                nrWords += 1

    occ_aug = occ + pseudo
    nrWords_aug = nrWords + len(occ)*pseudo

    return occ_aug, nrWords_aug

def trainModel(trainingSet, classes, pseudo):   # trainingSet is a list with 3 lists, one for each cantica
    dict = createDictionary(trainingSet)
    
    occurrencesMat = None
    nrWordsArr = numpy.zeros(len(classes))
    
    for c in classes:   # loop for crearing the matrix of occurrences
        occ_aug, nrWords_aug = computeOccurrences(trainingSet[c], pseudo, dict)
        
        if occurrencesMat is None:
            occurrencesMat = occ_aug
        else:
            occurrencesMat = numpy.vstack([occurrencesMat, occ_aug])

        nrWordsArr[c] = nrWords_aug
    
    nrWordsArr = nrWordsArr.reshape(len(classes), 1)

    return dict, occurrencesMat, nrWordsArr

def logLikelihoods(tercet, occurrencesMat, totNrWords, dict, pseudo): 
    freqMat = occurrencesMat/totNrWords     # elementwise division through broadcasting (occurrencesMat -> 3xM  totNrWords -> 3x1)

    wordsOccur, notImportant = computeOccurrences([tercet], pseudo, dict)
    wordsOccur = wordsOccur.T

    logFreqMat = numpy.log(freqMat)
    log_likelihoods = numpy.dot(logFreqMat, wordsOccur)

    return log_likelihoods

def classPosterior(log_likelihoods, priors):
    # log-likelihoods: log(P(x | c));    log-priors: log(P(c));
    # log-joint: log(P(x, c)) = log(P(x | c)*P(c))  =>  log(P(x, c)) = log(P(x | c)) + log(P(c))
    # log-marginal: sum_over_c(P(x, c)) IMPORTANT: sum over 'c' of the probabilities, NOT of the log-probabilities
    # log-posterior: log-joint - log-marginal 

    log_priors = numpy.log(priors)
    log_jointProbs = log_likelihoods + log_priors
    log_marginal = scipy.special.logsumexp(log_jointProbs, axis = 0)
    log_posterior = log_jointProbs -log_marginal

    return numpy.exp(log_posterior)

def scoresMatrix(cantiche, priors, occurrencesMat, totNrWords, dict, pseudo):
    scores = None
    S = None

    for tercets in cantiche:
        for t in tercets:
            l=logLikelihoods(t, occurrencesMat, totNrWords, dict, pseudo)
            cp = classPosterior(l, priors)

            if scores is None:
                scores = cp
            else:
                scores = numpy.hstack([scores, cp])
            
            if S is None:
                S = l
            else:
                S = numpy.hstack([S, l])

    return scores

def accuracy(expectedClass, scores):
    prediction = scores.argmax(axis=0)
    nrCorrectPredicitions = (prediction == expectedClass).sum()
    totalPredictions = len(prediction)

    return nrCorrectPredicitions/totalPredictions
    


if __name__ == '__main__':
    lInf, lPur, lPar = load_data()
    
    lInfTrain, lInfTest = split_data(lInf, 4)
    lPurTrain, lPurTest = split_data(lPur, 4)
    lParTrain, lParTest = split_data(lPar, 4)
    
    #dict = createDictionary([lInfTrain, lPurTrain, lParTrain])

    classLabel = {0: 'inferno', 1: 'purgatorio', 2: 'paradiso'}

    dict, occurrencesMat, nrWordsArr = trainModel([lInfTrain, lPurTrain, lParTrain], classLabel.keys(), 0.001)

    print(occurrencesMat)
    
    #scores = scoresMatrix([lInfTest, lPurTest, lParTest], [1/3, 1/3, 1/3], occurrencesMat, nrWordsArr, dict, 0.001)
    scores = scoresMatrix([lInfTest], [1/3, 1/3, 1/3], occurrencesMat, nrWordsArr, dict, 0.001)
    print(scores.shape)
    acc = accuracy(0, scores)

    print("Accuracy:")
    print(acc)







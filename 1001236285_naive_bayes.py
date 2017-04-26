# References-
# http://blog.yhat.com/posts/naive-bayes-in-python.html
# http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
# http://guidetodatamining.com/chapter7/

import os
import math

vocab=dict()
prob=dict()
total_words=dict()


def count_words(path, docs):
    #Counting words and building vocab
    clas_counts=dict()
    clas_total = 0
    for doc in docs:
            with open(path+'/'+doc,'r') as d:
                for line in d:
                    words = line.lower().split()
                    for word in words:
                        word = word.strip('() \'".,?:-')
                        if word != '' and word not in vocab:
                            vocab[word] = 1
                            clas_counts[word] = 1
                            clas_total += 1
                        elif word != '':
                            clas_counts.setdefault(word, 0)
                            vocab[word] += 1
                            clas_counts[word] += 1
                            clas_total += 1
    return clas_counts, clas_total


def  naive_bayes_train():
    classes=os.listdir("20_newsgroups/training")  # read all folders in the directory
    print 'Calculating counts for all classes and building vocabulary...'
    for clas in classes:
        path, direc, docs = os.walk("20_newsgroups/training/"+clas).next() # seperating files from folders
        prob[clas], total_words[clas] = count_words(path, docs)

    # Deleting less frequent words with count less than 3
    mark_del = []
    for word in vocab:
        if vocab[word] < 3:
            mark_del.append(word)
    for word in mark_del:
        del vocab[word]
    print '\nVocabulary Created\n'

    #Calculating probabilities for each class and its words.
    for clas in classes:
        print 'Training class', clas
        total_count = total_words[clas]+len(vocab)
        for word in vocab:
            if word in prob[clas]:
                count = prob[clas][word]
            else:
                count = 1
            prob[clas][word] = float(count+1)/total_count
    print '===========Finished Training===========\n'


def calc_clas_prob(path, classes, doc):
    #Calculate probabilities for words in a document belonging to a class
    test_output = dict()
    for clas in classes:
        test_output[clas] = 0
    with open(path+'/'+doc, 'r') as doc:
        for line in doc:
            words = line.lower().split()
            for word in words:
                word = word.strip('() \'".,?:-')
                if word in vocab:
                    for clas in classes:
                        test_output[clas] += math.log(prob[clas][word])
    #take the class with the highest probability
    max_prob = -1000000000
    for clas in test_output:
        if test_output[clas] > max_prob:
            max_prob = test_output[clas]
            output_clas = clas
    return output_clas


def naive_bayes_test():
    classes=os.listdir("20_newsgroups/testing")
    total_accuracy = 0
    for clas in classes:
        print 'Testing class', clas
        correct = 0
        total = 0
        path, direc, docs = os.walk("20_newsgroups/testing/"+clas).next()
        for doc in docs:
            total += 1
            test_output = calc_clas_prob(path, classes, doc)
            if test_output == clas:
                correct += 1
        #calculated classifier's accuracy for the class
        accuracy = float(correct)/total*100
        print 'Accuracy is', accuracy, '%\n'
        total_accuracy += accuracy
    print '===========Finished Testing===========\n'
    print 'The accuracy of naive bayes classifier is', total_accuracy/20, '%'

naive_bayes_train()
naive_bayes_test()

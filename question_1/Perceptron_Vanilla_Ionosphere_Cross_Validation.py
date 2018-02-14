import numpy as np
import pandas as pd
from numpy import genfromtxt
import random
import copy as cp
import csv

def split_data(data):
	test_dataset_folds = []
	trainset_c = list(data)
	fold_size = int(len(data) / 10)
	for i in range(10):
		current_fold = []
		while (len(current_fold) < fold_size):
			i = random.randrange(0, len(trainset_c))
			current_fold.append(trainset_c.pop(i))
		test_dataset_folds.append(current_fold)
	return test_dataset_folds


def predict_labels(test_data, w):
    #print("length : %d",len(X[0]))
    score = 0
    correct_identified = 0
    for row in test_data:
        y_ionosphere = row[34]
        temp_row = cp.copy(row)
        temp_row[-1] = '1'
        temp_row = [float(i) for i in temp_row]
        x_ionosphere = temp_row
     #   print(x_breast_cancer)
     #   print(y_breast_cancer)

        prediction = (np.dot(x_ionosphere, w))
      #  print(prediction," and label ", y_ionosphere)

        if(prediction <= 0):
            prediction = -1
        else:
            prediction = 1

        if(y_ionosphere == 'g'):
            actual_label = -1;
        else:
            actual_label = 1;

        if(prediction == actual_label):
            score = score + 1
    return score

def perceptron_vanilla(X, Y,epochs):
    #print("length : %d",len(X[0]))
    w = np.zeros(len(X[0]))

    current_round = 0

    while (current_round < epochs):
        row_index = 0
        for row in X:

            if (np.isnan(row).any()):

                row_index += 1
                continue;

            if Y[row_index] == 'g':
                y = -1
            else:
                y = 1

            if np.dot(y, (np.dot(w, row))) <= 0:
                w = np.add(w, np.dot(y, row))

            row_index += 1

        current_round += 1
        row_index = 0

    return w


epoches =  [10, 15, 20, 25, 30, 35, 40, 45, 50]


for epoch in epoches:

    ionosphere_data = csv.reader(open('ionosphere.data'), delimiter=',')
    data = []
    for row in ionosphere_data:
        temp_row = cp.copy(row)
        data.append(temp_row)

    data = split_data(data)
    temp_data = data
    data = np.array(data)

    epoch_score = 0
    total_test_data = 0
    index = 0
    for fold in data:
        training_data = []
        training_data = [x for i,x in enumerate(data) if i!=index]
        x_ionosphere = []
        y_ionosphere = []
        i = 1
        for set in training_data:
            for row in set:
               # print(i, "row", row)
                y_ionosphere.append(row[34])
                temp_row = cp.copy(row)
                temp_row[-1] = '1'
                temp_row = [float(i) for i in temp_row]
                x_ionosphere.append(temp_row)
                i += 1

       # print(x_breast_cancer)
       # print(y_breast_cancer)


        w = perceptron_vanilla(x_ionosphere,y_ionosphere, epoch)
       # print("weight vector for epoch ", epoch, "is :", w)
        test_data = fold
        score = predict_labels(test_data, w)
        total_test_data += len(test_data)
        epoch_score += score
        index = index + 1
        #print("score for round no :", index, " is ", score, "out of", len(fold))

    print(" for epoch :", epoch, " score is ", epoch_score, " out of ", total_test_data, " success rate ", epoch_score/total_test_data )
import numpy as np
import pandas as pd
from numpy import genfromtxt
import random
import copy as cp

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


def predict_labels(test_data, weights):
    #print("length : %d",len(X[0]))
    score = 0
    correct_identified = 0
    for row in test_data:
        result = 0
        y_breast_cancer = row[9]
        temp_row = cp.copy(row)
        temp_row[-1] = 1
        x_breast_cancer = temp_row
     #   print(x_breast_cancer)
     #   print(y_breast_cancer)

        for w in weights:
            prediction = (np.dot(x_breast_cancer, w[0]))
            if(prediction <= 0):
                prediction = -1
            else:
                prediction = 1

            result += w[1] * prediction

        if(result <= 0):
            prediction  =-1
        else:
            prediction = 1
        if(y_breast_cancer == 4):
            actual_label = -1;
        else:
            actual_label = 1;

        if(prediction == actual_label):
            score = score + 1
    return score


def voted_perceptron(X, Y, epochs):
    # print("length : %d",len(X[0]))
    w = np.zeros(len(X[0]))

    current_round = 0

    weight_with_votes = []
    votes = 1
    while (current_round < epochs ):
        row_index = 0
        hasChanged = False
        for row in X:

            if (np.isnan(row).any()):
                row_index += 1
                continue;

            if Y[row_index] == 4:
                y = -1
            else:
                y = 1

            if np.dot(y, (np.dot(w, row))) <= 0:
                weight_with_votes.append([w, votes])
                votes = 1
                w = np.add(w, np.dot(y, row))
            else:
                votes += 1
            row_index += 1

        current_round += 1
        row_index = 0

    return weight_with_votes


epoches =  [10, 15, 20, 25, 30, 35, 40, 45, 50]


for epoch in epoches:

    breast_cancer_data = genfromtxt('breast-cancer-wisconsin.data', delimiter=',')
    data = split_data(breast_cancer_data)
    temp_data = data
    data = np.array(data)
    print(data.__len__())

    epoch_score = 0
    total_test_data = 0
    index = 0
    for fold in data:
        training_data = [x for i,x in enumerate(data) if i!=index]
      #  print(training_data)
        x_breast_cancer = []
        y_breast_cancer = []
        for set in training_data:
            for row in set:
                y_breast_cancer.append(row[9])
                temp_row = cp.copy(row)
                temp_row[-1] = 1
                x_breast_cancer.append(temp_row)

        w = voted_perceptron(x_breast_cancer,y_breast_cancer, epoch)
        test_data = fold
        score = predict_labels(test_data, w)
        total_test_data += len(test_data)
        epoch_score += score
        index = index + 1
       # print("score for round no :", index, " is ", score, "out of", len(fold))

    print(" for epoch :", epoch, " score is ", epoch_score, " out of ", total_test_data, " success rate ", epoch_score/total_test_data )
import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt

breast_cancer_data = genfromtxt('breast-cancer-wisconsin.data', delimiter=',')

temp_x = breast_cancer_data[:,0:9];
x_breast_cancer = []

for row in temp_x:
    row = np.append(row, [1])
    x_breast_cancer.append(row)

y_breast_cancer = breast_cancer_data[:,9];


def voted_perceptron(X, Y,epochs):
    #print("length : %d",len(X[0]))
    w = np.zeros(len(X[0]))

    current_round = 0
    hasChanged = True
    weight_with_votes = []
    votes = 1
    while(current_round < epochs):
        row_index = 0

        for row in X:

            if Y[row_index] == 4:
                y = -1
            else:
                y = 1

            if np.dot(y,(np.dot(w,row))) <= 0:
                weight_with_votes.append([w, votes])
                votes = 1
                w = np.add(w, np.dot(y, row))
            else:
                votes += 1
            row_index += 1

        current_round += 1
        row_index = 0

    return weight_with_votes


print(x_breast_cancer.__len__())
w = voted_perceptron(x_breast_cancer,y_breast_cancer, 35)

print(w)


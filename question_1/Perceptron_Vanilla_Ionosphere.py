import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
import csv
import copy as cp

ionosphere_data = csv.reader(open('ionosphere.data'), delimiter=',')
x_ionosphere = []
y_ionosphere = []
for row in ionosphere_data:
    temp_row = cp.copy(row)
    temp_row[-1] = '1'
    temp_row =  [float(i) for i in temp_row]
    x_ionosphere.append(temp_row)
    y_ionosphere.append(row[34])


print(x_ionosphere)
print(y_ionosphere)


def perceptron_vanilla(X, Y,epochs):
    #print("length : %d",len(X[0]))
    w = np.zeros(len(X[0]))

    current_round = 0
    while(current_round < epochs):
        row_index = 0
        hasChanged = False
        for row in X:

            if Y[row_index] == 'g':
                y = -1
            else:
                y = 1


            if np.dot(y,(np.dot(w,row))) <= 0:
                w = np.add(w, np.dot(y, row))

            row_index += 1

        current_round += 1
        row_index = 0

    return w


w = perceptron_vanilla(x_ionosphere,y_ionosphere, 35)

print(w)


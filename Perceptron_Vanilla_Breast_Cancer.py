import numpy as np
import pandas as pd
from numpy import genfromtxt


breast_cancer_data = genfromtxt('breast_cancer_data.csv', delimiter=',')

x_breat_cancer = breast_cancer_data[:,0:10];

#print(x_breat_cancer)
y_breast_cancer = breast_cancer_data[:,10];

#print(breast_cancer_data);
#print(y_breast_cancer);




def perceptron_vanilla(X, Y,epochs):
    #print("length : %d",len(X[0]))
    w = np.zeros(len(X[0]))

    for t in range(epochs):
        for i, x in enumerate(X):

            if(np.isnan(X[i]).any()):
                continue;

            print(X[i][0])

            if Y[i] == 2:
                y = -1
            else:
                y = 1

            if (np.dot(X[i], w)*y <= 0):
                w = w + X[i]*y

    return w

w = perceptron_vanilla(x_breat_cancer,y_breast_cancer, 20)
print(w)


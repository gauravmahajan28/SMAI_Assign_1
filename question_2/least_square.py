import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plot

class_1_points = [[3,3,1], [3,0,1], [2,1,1], [0, 2, 1]]
class_2_points = [[-1,1,-1], [0, 0, -1], [-1,-1, -1], [1,0,-1]]
total_points = [[3,3,1], [3,0,1], [2,1,1], [0, 2, 1], [-1,1,-1], [0, 0, -1], [-1,-1, -1], [1,0,-1]]

eta = 0
w = np.zeros(len(class_1_points[0]))
m = 8
iterations = 10000
current = 0

while current < iterations:
    k = current % m
    current  = current + 1
    product = np.dot(np.dot(eta, np.subtract(1 if total_points[k][2] == -1 else 1, np.dot(w, total_points[k]))), total_points[k])
    w = w + product
    eta = float(1) / current

print(w)

points_class_1_x = []
points_class_1_y = []
points_class_2_x = []
points_class_2_y = []

for point in class_1_points:
    points_class_1_x.append(point[0])
    points_class_1_y.append(point[1])

for point in class_2_points:
    points_class_2_x.append(point[0])
    points_class_2_y.append(point[1])

plot.scatter(points_class_1_x, points_class_1_y, c = "yellow", label = "class one points")
plot.scatter(points_class_2_x, points_class_2_y, c = "red", label = "class two points")

x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
y = []
for point in x:
    temp = float(point) * -w[0] / w[1] - w[2] / w[1];
    y.append(temp)

classifier = plot.plot(x, y, label = "least square classifier")

plot.legend()
plot.show()

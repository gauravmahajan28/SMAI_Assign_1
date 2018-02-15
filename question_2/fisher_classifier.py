import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plot


class_1_points = [[3,3], [3,0], [2,1], [0, 2]]
class_2_points = [[-1,1], [0, 0], [-1,-1], [1,0]]


sum_x = 0
sum_y = 0

for point in class_1_points:
    sum_x += point[0]
    sum_y += point[1]

avg_class_1 = [sum_x / 4, sum_y / 4]

sum_x = 0
sum_y = 0

for point in class_2_points:
    sum_x += point[0]
    sum_y += point[1]

avg_class_2 = [sum_x / 4, sum_y / 4]

avg_class_1 = np.array(avg_class_1)
avg_class_2 = np.array(avg_class_2)

print(avg_class_1)
print(avg_class_2)

within_class1_scatter = np.dot((class_1_points - avg_class_1).T, (class_1_points - avg_class_1))
within_class2_scatter = np.dot((class_2_points - avg_class_2).T, (class_2_points - avg_class_2))

print(within_class1_scatter)
print(within_class2_scatter)

total_scatter = within_class2_scatter + within_class1_scatter

between_class_distance = avg_class_1 - avg_class_2


inverse = np.linalg.inv(total_scatter)

weight_vector = np.dot(inverse, between_class_distance)

print(weight_vector)

norm = np.linalg.norm(weight_vector)

weight_vector /= norm

print(weight_vector)

class_1_projected_points = []
class_2_projected_points = []

for point in class_1_points:
    class_1_projected_points.append(np.dot(np.dot(point, weight_vector),weight_vector))

print(class_1_projected_points)

X_points = []
Y_ = []

for point in class_1_projected_points:
    X_points.append([point[0], point[1], 1])
    Y_.append(1)



for point in class_2_points:
    class_2_projected_points.append(np.dot(np.dot(point, weight_vector),weight_vector))

for point in class_2_projected_points:
    X_points.append([point[0], point[1], 1])
    Y_.append(-1)



w = np.zeros(len(X_points[0]))
current_round = 0
epochs = 35
while(current_round < epochs):
    row_index = 0
    for row in X_points:
        if np.dot(Y_[row_index],(np.dot(w,row))) <= 0:
            w = np.add(w, np.dot(Y_[row_index], row))
        row_index += 1
    current_round += 1


print("classifier",w)

orig_points_class_1_x = []
orig_points_class_1_y = []
orig_points_class_2_x = []
orig_points_class_2_y = []

for point in class_1_points:
    orig_points_class_1_x.append(point[0])
    orig_points_class_1_y.append(point[1])

for point in class_2_points:
    orig_points_class_2_x.append(point[0])
    orig_points_class_2_y.append(point[1])

plot.scatter(orig_points_class_1_x, orig_points_class_1_y, c = "yellow", label = "class one points")
plot.scatter(orig_points_class_2_x, orig_points_class_2_y, c = "red", label = "class two points")


projected_points_class_1_x = []
projected_points_class_1_y = []
projected_points_class_2_x = []
projected_points_class_2_y = []

for point in class_1_projected_points:
    projected_points_class_1_x.append(point[0])
    projected_points_class_1_y.append(point[1])

for point in class_2_projected_points:
    projected_points_class_2_x.append(point[0])
    projected_points_class_2_y.append(point[1])

plot.scatter(projected_points_class_1_x, projected_points_class_1_y, c="green", label="class one projected points")
plot.scatter(projected_points_class_2_x, projected_points_class_2_y, c="black", label="class two projected points")


x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
y = []
for point in x:
    temp = float(point) * -w[0] / w[1] - w[2] / w[1];
    y.append(temp)

classifier = plot.plot(x, y, label = "least square classifier")



plot.legend()
plot.show()
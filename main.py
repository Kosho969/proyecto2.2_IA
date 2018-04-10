import re 
from expectation_max import Expectation_maximization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import multivariate_normal
import operator
from gaussian import Gaussian


data = []

with open('generatedPoints.txt') as f:
    for line in f: 
        # print(line, end='')
        value = line.split(' ')
        value[0] = re.sub('[[,]', '', value[0])
        value[1] = re.sub('[],]', '', value[1])
        value[1] = value[1].replace('\n','')
        data.append(value)
        #= value[0].replace('[','')

data_set = [[[float(i[0])], [float(i[1])]] for i in data]


data_set = np.asarray(data_set)

# for i in range(len(data_set)):
#     print(data_set[i])

plt.plot(data_set[:,0], data_set[:,1], 'o', color='black');


expectation_object = Expectation_maximization(data_set, 3, 150)

while (True):
    vector = input("Enter point: ")
    vector = vector.split(' ')
    vector[0] = re.sub('[[,]', '', vector[0])
    vector[1] = re.sub('[],]', '', vector[1])
    vector[1] = vector[1].replace('\n','')
    print(vector[0])
    print(vector[1])
    point = [[float(vector[0])], [float(vector[1])]]
    point = np.asarray(point)

    classification = expectation_object.check_classify(point)
    print(classification)
    index, value = max(enumerate(classification), key=operator.itemgetter(1))
    print('Gaussian Number: ', index)
    print('Probability: ', value)
    print(expectation_object.gaussians[index].to_string())

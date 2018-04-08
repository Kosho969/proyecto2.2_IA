import re 
from expectation_max import Expectation_maximization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


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

for i in range(len(data_set)):
    print(data_set[i])

plt.plot(data_set[:,0], data_set[:,1], 'o', color='black');
plt.show()

Expectation_maximization(data_set, 3, 350)



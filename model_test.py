import numpy as np
from DecisionTree import DecisionTree


def GetData(path):
    with open(path) as f:
        data = f.read()
    data = data.split("\n")[:-1]
    x, y = [],[]
    for i in range(len(data)):
        temp = [float(i) for i in data[i].split(" ")]
        x.append(temp[:-1])
        y.append(temp[-1])
    return np.array(x), np.array(y)


path_train = "train.dat.txt"
path_test = "test.dat.txt"
x_train, y_train = GetData(path_train)
x_test, y_test = GetData(path_test)

# Q14
clf = DecisionTree()
dtree = clf.fit(x_train,y_train)
print(f"Eout: {clf.error_function(dtree, x_test, y_test)}")
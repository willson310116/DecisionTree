import numpy as np
import random

def GetTheta(x):
    x = np.sort(x)
    x = np.unique(x)
    theta = (x[1:] + x[:-1]) / 2
    theta = np.r_[[x[0] - 1], theta]
    theta = np.r_[theta, [x[-1] + 1]]
    return theta

class Node:
    def __init__(self, theta, col, value=None):
        self.theta = theta
        self.col = col
        self.value = value
        self.left = None
        self.right = None

class DecisionTree:
    
    def DecisionStump(self,x,y):
        n,dim = x.shape
        col = 0
        score_best = float("inf")
        for i in range(dim):
            x_i = x[:,i]
            theta_list = GetTheta(x_i)
            for theta in theta_list:
                score = self.Loss(x_i,y,theta)
                if score < score_best:
                    score_best = score
                    col = i
                    theta_best = theta
        return theta_best, col, score_best
    
    def fit(self,x,y):
        if self.terminate(x,y):
            return Node(None, None, y[0])
        
        theta, col, score = self.DecisionStump(x, y)
        tree = Node(theta, col)

        condition_left = x[:, col] < theta
        x_left = x[condition_left]
        y_left = y[condition_left]

        condition_right = x[:, col] >= theta
        x_right = x[condition_right]
        y_right = y[condition_right]

        tree_left = self.fit(x_left, y_left)
        tree_right = self.fit(x_right, y_right)

        tree.left = tree_left
        tree.right = tree_right

        return tree
    
    def terminate(self,x,y):
        n = x.shape[0]
        condition_1 = np.sum(x!=x[0, :]) # all xn the same: no decision stumps
        condition_2 = np.sum(y!=y[0]) # all yn the same: impurity = 0 -> gt(x) = yn
        return condition_1 == 0 or condition_2 == 0
    
    def Loss(self,x,y,theta):
        left = y[x < theta]
        right = y[x >= theta]
        gini_left = self.Gini(left)
        gini_right = self.Gini(right)
        return len(left)*gini_left + len(right)*gini_right
    
    def Gini(self,y):
        if len(y) == 0:
            return 1
        purity_pos = np.mean(y==1)
        purity_neg = np.mean(y==-1)
        return 1 - (purity_pos)**2 - (purity_neg)**2
    
    def predict(self,tree,x):
        if tree.value != None:
            return tree.value
        elif x[tree.col] < tree.theta:
            return self.predict(tree.left, x)
        else:
            return self.predict(tree.right, x)
    
    def error_function(self,tree, x, y):
        y_pred = [self.predict(tree, sample) for sample in x]
        return np.mean(y_pred != y)


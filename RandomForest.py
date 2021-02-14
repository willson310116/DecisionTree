import numpy as np
import random 
from DecisionTree import DecisionTree

class RandomForest:
    def __init__(self,n_clf=50):
        self.n_clf = n_clf
        self.tree_list = None
        self.error_list = None
        self.oob_list = None
        
    def getForest(self, x, y, boot_ratio=0.5):
        error_list = []
        tree_list = []
        oob_list = []
        n = x.shape[0]
        for i in range(self.n_clf):
            print(f"{i}-tree")
            bootstrap = np.random.randint(0, n, int(n*boot_ratio))
            oob = sorted(list(set([i for i in range(n)]) - set(bootstrap))) # the oob samples of the i-th tree
            oob_list.append(oob)
            x_boot = x[bootstrap]
            y_boot = y[bootstrap]
            clf = DecisionTree()
            dtree = clf.fit(x_boot,y_boot)
            
            tree_list.append(dtree)
            error_list.append(clf.error_function(dtree,x,y))
        error_list = np.array(error_list)
#         return tree_list, error_list, oob_list
        self.tree_list = tree_list
        self.error_list = error_list    
        self.oob_list = oob_list
        
    def fit(self, x, y):
        self.getForest(x,y)
        y_pred_list = []
        for i in range(x.shape[0]):
            temp = []
            for tree in self.tree_list:
                temp.append(self.predict_one(tree,x[i]))
            y_pred_list.append(max(set(temp), key=temp.count)) # append mode
        y_pred_list = np.array(y_pred_list)
        return y_pred_list, np.mean(y_pred_list)
    
    def predict_one(self,tree,x):
        if tree.value != None:
            return tree.value
        elif x[tree.col] < tree.theta:
            return self.predict_one(tree.left, x)
        else:
            return self.predict_one(tree.right, x)
    
    def error_function(self,tree, x, y):
        y_pred = [self.predict_one(tree, sample) for sample in x]
        return np.mean(y_pred != y)
        
    def Calculate_Eoob(self,x,y):
    # tree_list -> g-
        y_pred_list = []
        for i in range(x.shape[0]): # i-th sample
            G_minus = []
            temp = []
            for j in range(len(self.oob_list)):
                if i in oob_list[j]:
                    G_minus.append(self.tree_list[j])
#                     break
                elif j==(x.shape[0]-1) and i not in self.oob_list[j]:
                    G_minus.append(-1)
            if (-1) in G_minus:
                y_pred_list.append(-1)
            else:
                for g_minus in G_minus:
                    temp.append(self.predict_one(g_minus,x[i]))
                y_pred = max(set(temp), key=temp.count) # g- vote to get y_pred
                y_pred_list.append(y_pred)
        y_pred_list = np.array(y_pred_list)
        return (np.mean(y_pred_list!=y),y_pred_list)
    
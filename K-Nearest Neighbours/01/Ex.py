# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
import numpy as np

"""
I decided to take a supervised learning with KNN approach due to problem
being a multiclassification one.
"""

file=pd.ExcelFile("MBA1.xlsx")

X=pd.read_excel(file,'Data',skiprows=1,nrows=2126,usecols="K:AE")
Y=pd.read_excel(file,'Data',skiprows=1,nrows=2126,usecols='AT')

"""
Scale down the features for easier computation
"""
X = StandardScaler().fit_transform(X)

"""
Reduce the number of components for standardization of the calculations
to exclude variables which do not make much of a difference
"""
pca = PCA(n_components=5)
X = pca.fit_transform(X)

"""
Create testing sets with test_size of 25%
"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=100, stratify=Y)

"""
Evaluate N_neighbours

knn_k= KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 25)}
knn_gscv = GridSearchCV(knn_k, param_grid, cv=50) (50 folds)
knn_gscv.fit(X,np.ravel(Y,order='C'))
print(knn_gscv.best_params_)

Output : 6
"""

knn=KNeighborsClassifier(n_neighbors=6)
"""
Transform from [yi,1] to [1,yi] to vectorize the matrix
"""

knn.fit(X_train, np.ravel(Y_train,order='C'))
Y_pred=knn.predict(X_test)

"""
Accuracy Test over Training Set
"""
print("Accuracy Test of T Set = {}".format(accuracy_score(Y_test, Y_pred)))

"""
Cross validation
"""
scores=cross_val_score(knn,X_train,np.ravel(Y_train,order='C'), cv=5)
print("Cross Validation of T Set(10 Folds) = {}".format(np.mean(scores)))


"""
Prediction Score of whole Dataset
"""
print("Prediction Score of W Set = {}".format(knn.score(X,Y)))

"""
Evaluation
"""

Y_pred = tuple(knn.predict(X))
Sum=0
i=0

while i < len(Y):
    if(Y_pred[i]!=Y["NSP"][i]):
        if(Y["NSP"][i]==3):
            if(Y_pred[i]==1):
                Sum+=1000
            else:
                Sum+=100
        elif(Y["NSP"][i]==1):
            if(Y_pred[i]==2 or Y_pred[i]==3):
                Sum+=100
        elif(Y["NSP"][i]==2):
            if(Y_pred[i]==1):
                Sum+=300
            else:
                Sum+=100
    i+=1


print("Cost in money of Model: {}".format(Sum))
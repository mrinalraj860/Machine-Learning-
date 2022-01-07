import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score


boston=datasets.load_boston()

#x=feature
x=boston.data
#y=label/target
y=boston.target

print(x.shape)
print(x)

print(y.shape)
print(y)

l_reg=linear_model.LinearRegression()
#plt.scatter(x.T[0],y)
#plt.show()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=l_reg.fit(x_train,y_train)
prec=model.predict(x_test)

print("prediction=")
print(prec)
#as it is continous the acc_score cant work on this model 
#case there is nothing to predict
#acc=accuracy_score(y_test,prec)
#print(acc)

#r^2 value proportion of the variance for a dependent variable

print("R^2 value: ",l_reg.score(x,y))
print("coeff:" , l_reg.rank_)
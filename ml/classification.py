from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split


iris=datasets.load_iris()
x=iris.data
y=iris.target

print(x.shape,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)#test_size is the is size of test case we want to consider
#testing data is too low then accuracy is also going to be low

print(x_train.shape)#.shape returns tupple
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
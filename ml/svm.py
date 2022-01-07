import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



iris=datasets.load_iris()
x=iris.data
y=iris.target

classes=["IRIS Setosa","Iris Versicolour"," Iris Virginica"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
model =svm.SVC()
model.fit(x_train,y_train)


pri=model.predict(x_test)
acc=accuracy_score(y_test,pri)
print(y_test)
print(pri)

print("acc score is : ", acc)
for i in pri:
    print(classes[pri[i]])
print()
for i in range(len(pri)):
    print(classes[pri[i]])
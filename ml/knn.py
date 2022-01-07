import numpy as np
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#resp needed for knn

data=pd.read_csv('car.data')

print(data.head)




X=data[[
'buying',
'maint',
'safety'
]].values
Y=data[['Class']]

#print(X,Y)


#tranforming string attributes to number
Le=LabelEncoder()
for i in range(len(X[0])):
	X[:, i]=Le.fit_transform(X[:, i])

#print(X,Y)




#transforming string attributed to number via dict can call it maping

label_maping={
	'unacc': 0,
	'acc' : 1,
	'good' : 2,
	'vgood' : 3
}

Y['Class']=Y['Class'].map(label_maping)

Y=np.array(Y)
#print(y)




#creating model 

Knn=neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)


Knn.fit(x_train,y_train)
predic=Knn.predict(x_test)
acc=metrics.accuracy_score(y_test,predic)
print(predic)
print(acc)

print(Knn.predict(X)[20])
print(Y[20])

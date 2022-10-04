# import the required files for ML
# pandas and numpy are the mst important lib files 

import pandas as pd  # used to upload the dataset

import numpy as np   #used for working with arrays

# dataset gets readed by pd 
dataset =pd.read_csv("set.csv")
#print(dataset)








# inputs 
X=dataset.iloc[:, [0,1]].values
#print(X)

# ouput
Y=dataset.iloc[:, 2].values
#print(Y)


# used to train the model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state =0)



# have to scale the object
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
#print(X_train)


# # this model is under Logistic Regression
# from sklearn.linear_model import LogisticRegression
# model=LogisticRegression(random_state=0)
# model.fit(X_train,y_train)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

course=input("Enter The total courses: ")
hours=input("Enter The total Hours: ")

new =[[course,hours]]                                   
result=model.predict(sc.transform(new))
print(result)


y_pred=model.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.model_selection import cross_val_score
a=cross_val_score(LinearRegression(),X,Y,cv=3 )  #the cross_val_score returns the accuracy for all the folds
print(a)

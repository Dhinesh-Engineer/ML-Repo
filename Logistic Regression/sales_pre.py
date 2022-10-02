# import the required files for ML
# pandas and numpy are the mst important lib files 
import pandas as pd  # used to upload the dataset

import numpy as np   #used for working with arrays

# dataset gets readed by pd 
dataset =pd.read_csv('data_set.csv')
#print(dataset)

# inputs 
X=dataset.iloc[:, :-1].values
#print(X)

# ouput
Y=dataset.iloc[:, -1].values
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

# this model is under Logistic Regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(random_state=0)
model.fit(X_train,y_train)

age=int(input("Enter New Customer's Age:"))
sal=int(input("Enter New Customer's Salary:"))
newCust=[[age,sal]]                  
result=model.predict(sc.transform(newCust))
print(result)
if result ==1:
    print("Customer will Buy the product")
else:              
    print("Customer won't Buy the product")

# to check the accuracy of the model 

y_pred=model.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,y_pred)
print("Accuracy of the Model:{0}%".format(accuracy_score(y_test,y_pred)*100))

 


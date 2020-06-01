#importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklrean.cross_validation import train_test_split
#matplotlib inline
#using pandas to read the database stored in the same folder
data=pd.read_csv('mnist.csv')
#viewing column heads
data.head{}
#extracting data from the dataset and viewing then up dose
a=data.iloc[3,1:].values

#reshaping the extracted data into a reasonable size
a=a.reshape(28,28).astype('uint8')
plt.inshow(a)

#preparing the data
#separating labels and data values
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]

#creating test and train sizes/batches x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)
#check data
x_train.head() //similar to y_train

#call rf classifier
rf=RandomForestClassifier(n_estimators=100)

#fit the model
rf.fit(x_train,y_train)

#prediction on test data
pred=rf.predict(x_test)

#check prediction accuracy
s=y_test_values

#calculate number of correctly predicted values
count=0
for i in range(len(pred)):
    if pred(i)==s[i]:
       count=count+i

#total values that to prediction code was run on
len(pred)

#accuracy value
8090/8400 

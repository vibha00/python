#loading the data
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master.iris.cav"
names=['sepal-length','sepal-width','petal-length','class']
dataset=read_csv(url,names=names)

#dimensions of the dataset
print(dataset.shape)
#take a peek at the data
print(dataset.head(20))
#statistical summary
print(dataset.describe())
#class distribution
print(dataset.groupby('class').size())

#univariate plots-box and whisker plots
dataset.plot(kind='box'.subplots=true,layout=(2,2),sharex=False,sharey=False)
pyplot.show()

#historgram of the variable
dataset.hist()
pyplot.show()
#multivariate plots
scatter_matrix(dataset)
pyplot.show()

#creating a validation dataset
#splitting dataset
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
X_train,X_validation,y_train,Y_validation=train_test_split(X,Y,test_size=0.2,random_state=1)

#logistic Regression
#linear discriminant analysis
#K-Nearest neighbors
#Classification and regression Trees
#Gaussian Naive Bayes
#Support Vector Machine

#buliding models
model=[]
models.append(('LR',Logistic Regression(solver='liblinear'.multi_class='ovr')))
models.append(('LDA',linear discriminant analysis()))
models.append(('KNN',KNeighbors classifier()))
models.append(('NB',Gaussian NB()))
models.append(('SVM',SVC(gamma='auto')))

#evaluate the created models
results=[]
names=[] 
for name,model is models:
kfold=Stratifiedkfold(n_splits=10,random_state=1)
cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
results.append(cv_results)
names.append(name)
print('%s:%f(%f)%(name cv_results.mean(),cv_results.std()))

#compare our models
pyplot.boxplot(results,labels=names)
pyplot.title('algorithm comparison')
pyplot.show()

#make predictions onsun
model=svc(games='auto')
model.fit(X_train,Y_train)
predictions=model.predict(X_validation)

#evaluate our predictions
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_repeat(Y_validation,predictions))

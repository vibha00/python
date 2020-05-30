#regression
from sklearn.linear_model import
LinearRegression/lasso/Ridge

#creating a regression model
reg=LinearRegression()

#fit the model
reg.fit(data,target)

#prediction
pred=reg.predict(data)

#entering that matplotlib is working inside the notebook
%matplotlib inline
plt.scatter(data,target,color='red')
plt.plot(data,pred,color='green')
plt.xlabel('lower income popularity')
plt.ylabel('cost of house')
plt.show()

#circumventing curve issue using polynomial model
from sklearn.preprocessing import
PolynomialFeatured

#to allow merging of modles
from sklearn.pipeline import make_pipeline

model=make_pipeline(PolynomialFeatures(3),reg)
model.fit(data,target)
pred=model.predict(data)

#entering that matplotlib is working inside the notebook
%matplotlib inline
plt.scatter(data,target,color='red')
plt.plot(data,pred,color='green')
plt.xlabel('lower income popularity')
plt.ylabel('cost of house')
plt.show()

#r_2 matric
from sklearn.matrices import r2_score

#predict
r2_score(pred,target)
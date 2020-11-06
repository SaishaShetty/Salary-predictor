import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('PS.csv')

X=data.iloc[:,1:2].values
y=data.iloc[:,2].values

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures()
X_poly=poly.fit_transform(X)
poly.fit(X_poly,y)
regressor2=LinearRegression()
regressor2.fit(X_poly,y)

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

plt.scatter(X,y,color='red')
plt.plot(X,regressor2.predict(poly.fit_transform(X)),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor2.predict(poly.fit_transform(X_grid)),color='blue')
plt.title('Truth or bluffsdas')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

regressor.predict([[4]])

regressor2.predict(poly.fit_transform([[4]]))
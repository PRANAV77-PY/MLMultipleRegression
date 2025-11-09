import pandas
from sklearn import linear_model

df = pandas.read_csv('D:\Software engg Professional training\Python advanced\Machine learning\MultipleRegression.py\dataM.csv')

x = df[['Weight','Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(x,y)

print(regr.coef_)
import pandas
from sklearn import linear_model

df = pandas.read_csv('D:\\Software engg Professional training\\Python advanced\Machine learning\\MultipleRegression.py\\dataM.csv')

X = df[['Weight','Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X,y)

#predict the CO2 emission
predictedCO2 = regr.predict([[3500,1500]])

print(predictedCO2)

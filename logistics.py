# LOGISTICS REGRESSION (BINARY CALLIFICATION)

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


dataset = pd.read_csv("term_insurance.csv")
# print(dataset)

dataset["Bought_Insurance"].replace({"no":0,"yes":1},inplace=True)
# print(dataset)
# print(dataset.columns)

plt.scatter(x="Age",y="Bought_Insurance",data=dataset)
# plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(dataset[["Age"]],dataset["Bought_Insurance"],test_size=0.2,random_state=0)

# print(len(x_train))

print(x_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)

print(lr.predict(x_test))

print(lr.predict([[65]]))

print(lr.predict([[22]]))




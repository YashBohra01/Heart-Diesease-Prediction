import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import tree

df=pd.read_csv('heartdisease/processed.cleveland.data')
# print(df.head())
df['disease_found']=df['num']>0
df.replace('?', -99999, inplace=True)

#df.drop(['id'], 1, inplace=True)
# print(df.columns)

X= np.array(df.drop(['num'],1))
y= np.array(df['num'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)

accuracy= clf.score(X_test,y_test)
print(accuracy)
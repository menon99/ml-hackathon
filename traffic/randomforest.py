import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from cleaner import clean

s='DataSets/Train.csv'
#Correlation matrix
df=clean(s)
corr=df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':10}, vmin=-1, vmax=1)
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

#Model training
Y=df['traffic_volume'].values
X=df.drop(['date_time', 'traffic_volume', 'dew_point'], axis=1)

x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.25, random_state=42)

model=RandomForestRegressor(n_jobs=-1)

estimators=np.arange(10, 511, 50) #(10, 20, 30, ...)
scores=[]

#n_estimators indicate the no. of trees
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(x_train, y_train)
    scores.append(model.score(x_test, y_test))
    y_pred=model.predict(x_test)
    print(r2_score(y_test, y_pred),' ', mean_squared_error(y_test, y_pred))

plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
plt.show()

'''
df=df.drop(['date_time', 'traffic_volume'])
df.to_csv('removeduplicates.csv', index=False)'''
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

f='%m/%d/%Y %H:%M'

df=pd.read_csv('Train.csv')

#converting is_holiday to numerical
df['is_holiday']=df['is_holiday'].map(lambda s:1 if s=='None' else 2)

#converting weather_type to numerical
df['weather_type']=df['weather_type'].astype('category')
df['weather_type']=df['weather_type'].cat.codes

#converting weather_description to numerical
df['weather_description']=df['weather_description'].astype('category')
df['weather_description']=df['weather_description'].cat.codes

df2=pd.DataFrame(data=[],columns=['year','date','month','hour'])

#coverting into datetime format
df['date_time']=df['date_time'].map(lambda s:dt.datetime.strptime(s,f))

#extracting date,year,month and hour
df2['year']=df['date_time'].map(lambda s:s.year)
df2['month']=df['date_time'].map(lambda s:s.month)
df2['date']=df['date_time'].map(lambda s:s.day)
df2['hour']=df['date_time'].map(lambda s:s.hour)

#merging with original
df_merged=pd.concat([df,df2],axis=1)
df_merged.drop_duplicates(subset ="date_time", keep = "first", inplace = True)

#Correlation matrix
corr=df_merged.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':10}, vmin=-1, vmax=1)
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

#Model training
Y=df_merged['traffic_volume'].values
X=df_merged.drop(['date_time', 'traffic_volume', 'dew_point'], axis=1)

x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.25, random_state=42)

model=RandomForestRegressor(n_jobs=-1)

estimators=np.arange(10, 200, 10) #(10, 20, 30, ...)
scores=[]

#n_estimators indicate the no. of trees
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(x_train, y_train)
    scores.append(model.score(x_test, y_test))
    y_pred=model.predict(x_test)
    print(r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred))

plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
plt.show()

'''
df_merged=df_merged.drop(['date_time', 'traffic_volume'])
df_merged.to_csv('removeduplicates.csv', index=False)'''
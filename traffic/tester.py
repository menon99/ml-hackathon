from cleaner import clean
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import datetime as dt

s='DataSets/Train.csv'
df=clean(s)
df.drop_duplicates('date_time', keep='first', inplace=True)
Y=df['traffic_volume'].values
X=df.drop(['date_time', 'traffic_volume', 'dew_point'], axis=1)

x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.25, random_state=42)

model=RandomForestRegressor(n_jobs=-1,n_estimators=360)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
err=mean_squared_error(y_test, y_pred)

print('mean squared error is %r'%err)

s='DataSets/Test.csv'
df2=clean(s)
dates=df2['date_time']
X=df2.drop(['date_time','dew_point'], axis=1)

traffic_volume=model.predict(X)

df_submit=pd.DataFrame(columns=['date_time','traffic_volume'])
df_submit['date_time']=dates

f='%m/%d/%Y %H:%M:%S'

#df_submit['date_time']=df['date_time'].map(lambda s: dt.datetime.strftime(s, f))
df_submit['traffic_volume']=traffic_volume
print(df_submit.head())
df_submit.to_csv('DataSets/submit.csv',index=False)
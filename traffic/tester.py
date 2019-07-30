from cleaner import clean
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_squared_log_error
import pandas as pd
import datetime as dt

s='DataSets/Train.csv'
#s='DataSets/rem_outliers.csv'
df=clean(s)
df.drop_duplicates('date_time', keep='first', inplace=True)
Y=df['traffic_volume'].values
X=df.drop(['date_time', 'traffic_volume', 'dew_point'], axis=1)

x_new=pd.DataFrame()
x_new['temperature']=df['temperature']
x_new['hour']=df['hour']
x_new['weather_description']=df['weather_description']
x_new['is_holiday']=df['is_holiday']
x_new['clouds_all']=df['clouds_all']
x_new['weather_type']=df['weather_type']
x_new['humidity']=df['humidity']
#x_new['date']=df['date']
#x_new['month']=df['month']

#x_train, x_test, y_train, y_test=train_test_split(x_new, Y, test_size=0.25, random_state=42)

model=RandomForestRegressor(n_jobs=-1,n_estimators=100,max_depth=6) #original is 100 and 6

#model.fit(x_train,y_train)
model.fit(x_new,Y)
y_pred=model.predict(x_new)
#y_pred=model.predict(x_test)
print('msle is:%r',mean_squared_log_error(Y,y_pred))
#err=mean_squared_error(y_test, y_pred)

#print('mean squared error is %r'%err)


s='DataSets/Test.csv'
df2=clean(s)
dates=df2['date_time']
#X=df2.drop(['date_time','dew_point'], axis=1)

x_new=pd.DataFrame()
x_new['temperature']=df2['temperature']
x_new['hour']=df2['hour']
x_new['weather_description']=df2['weather_description']
x_new['is_holiday']=df2['is_holiday']
x_new['clouds_all']=df2['clouds_all']
x_new['weather_type']=df2['weather_type']
x_new['humidity']=df2['humidity']
#x_new['date']=df2['date']
#x_new['month']=df2['month']

traffic_volume=model.predict(x_new)


df_submit=pd.DataFrame(columns=['date_time','traffic_volume'])
df_submit['date_time']=dates

#f='%m/%d/%Y %H:%M:%S'

#df_submit['date_time']=df['date_time'].map(lambda s: dt.datetime.strftime(s, f))
df_submit['traffic_volume']=traffic_volume
print(df_submit.head())
#df_submit.to_csv('DataSets/submit_outlier.csv',index=False)
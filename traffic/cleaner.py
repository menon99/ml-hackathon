import pandas as pd
import datetime as dt

f='%Y-%m-%d %H:%M:%S'

df=pd.read_csv('DataSets/Train.csv')

#converting is_holiday to numerical
df['is_holiday']=df['is_holiday'].map(lambda s:0 if s=='None' else 1)

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
df_merged.to_csv('DataSets/cleaned.csv')
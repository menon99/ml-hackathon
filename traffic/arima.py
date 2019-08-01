import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.plotly import plot_mpl
import numpy as np
from pyramid.arima import auto_arima

data = pd.read_csv("removeduplicates.csv",usecols=['date_time','traffic_volume'])[:28500]
Traffic=data['traffic_volume']

result = seasonal_decompose (Traffic,model='additive',freq=1)
stepwise_model = auto_arima(Traffic, start_p=1, start_q=1,max_p=2, max_q=2, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',  suppress_warnings=True, stepwise=False)
#print(stepwise_model.aic())

stepwise_model.fit(Traffic)
future_forecast = stepwise_model.predict(n_periods=15555)
print (future_forecast)

writer = pd.ExcelWriter('predicted_values.xlsx',engine='openpyxl')
df=pd.DataFrame({'Pred':future_forecast})
df.to_excel("./test2.xlsx",index=False)
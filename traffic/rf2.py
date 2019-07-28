from cleaner import clean
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error,mean_squared_log_error
import numpy as np


def rfr_model(X, y):
    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3, 7),
            #'n_estimators':(1000,2000),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1,return_train_score=True)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    print('best_params are:',best_params)
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=210, random_state=False, verbose=False)
    # Perform K-Fold CV
    rfr.fit(X,y)
    #scores = cross_val_score(rfr, X, y, cv=10, scoring='neg_mean_absolute_error')

    #return [scores,rfr]
    return rfr

s = 'DataSets/Train.csv'
df = clean(s)
print(df.head())

Y=df['traffic_volume'].values
X=df.drop(['date_time', 'traffic_volume', 'dew_point'], axis=1)

x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.25, random_state=42)

rfr=rfr_model(X,Y)
#print('scores are:',scores)
y_pred=rfr.predict(x_test)
err=mean_squared_error(y_test, y_pred)
err_log=mean_squared_log_error(y_test,y_pred)
print('mean squared error is %r\n'%err)
print('mean squared log error is %r\n'%err_log)
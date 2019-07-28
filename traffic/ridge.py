from cleaner import clean
import numpy as np
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.metrics import mean_squared_error,mean_squared_log_error
from sklearn.model_selection import cross_val_score, GridSearchCV,train_test_split

def ridge(X,Y):
    gsc = GridSearchCV(
            estimator=RidgeCV(),
            param_grid={
                'fit_intercept':[True,False],
                'normalize':[True,False],
                'cv':[None,3,5,10],
            },
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1,return_train_score=True)

    grid_result = gsc.fit(X, Y)
    best_params = grid_result.best_params_
    model=RidgeCV(alphas=[0.1,0.3,0.5,0.7,1,3,6,10],fit_intercept=best_params['fit_intercept'],normalize=best_params['normalize'],cv=best_params['cv'])
    model.fit(X,Y)
    return model

s = 'DataSets/Train.csv'
df = clean(s)
print(df.head())

Y=df['traffic_volume'].values
X=df.drop(['date_time', 'traffic_volume', 'dew_point'], axis=1)

model=ridge(X,Y)

x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.25, random_state=42)
y_pred=model.predict(x_test)
err=mean_squared_error(y_test, y_pred)
#err_log=mean_squared_log_error(y_test,y_pred)
print('mean squared error is %r\n'%err)
#print('mean squared log error is %r\n'%err_log)

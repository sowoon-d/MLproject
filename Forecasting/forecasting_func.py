import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, make_scorer
import sklearn.metrics as metrics
from sklearn.pipeline import Pipeline
import pickle
from datetime import datetime
import yaml



def load_data(**context):

    # conf 값 가져오기
    with open('conf.yml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    url = conf['url']

    data = pd.read_csv(url,sep=",")
    data = data.dropna()

    # converting data to correct format 
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    data_consumption = data.loc[:,['Consumption']]
    data_consumption['Yesterday'] = data_consumption.loc[:,['Consumption']].shift()
    data_consumption.loc[:,'Yesterday'] = data_consumption.loc[:,['Consumption']].shift()
    data_consumption['Yesterday_Diff'] = data_consumption['Yesterday'].diff()
    data_consumption = data_consumption.dropna()
    context['task_instance'].xcom_push(key='data_consumption', value=data_consumption)
    # return data_consumption


def split_dataset(**context):
    data_consumption = context['task_instance'].xcom_pull(task_ids='load_data', key='data_consumption')
    # data_consumption = context['task_instance'].xcom_pull(task_ids='load_data')

    X_train = data_consumption[:'2016'].drop(['Consumption'], axis=1)
    y_train = data_consumption.loc[:'2016', 'Consumption']

    X_test = data_consumption['2017'].drop(['Consumption'], axis=1)
    y_test = data_consumption.loc['2017', 'Consumption']

    context['task_instance'].xcom_push(key='X_train', value=X_train)
    context['task_instance'].xcom_push(key='y_train', value=y_train)
    context['task_instance'].xcom_push(key='X_test', value=X_test)
    context['task_instance'].xcom_push(key='y_test', value=y_test)
    # return X_train, y_train, X_test, y_test





# Helper function: performance metrics
def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))             
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score





def model_gridsearchCV(**context):
    X_train = context['task_instance'].xcom_pull(task_ids='split_dataset', key='X_train')
    y_train = context['task_instance'].xcom_pull(task_ids='split_dataset', key='y_train')

    # conf에서 models 가져오기
    with open('conf.yml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    models = conf['models']

    tscv = TimeSeriesSplit(n_splits=10)
    rmse_score = make_scorer(rmse, greater_is_better = False)

    results_estimator = {}
    results_score = {}

    for model in models:
        est = eval(model['class'])()

        gsearch = GridSearchCV(estimator=est, cv=tscv, param_grid=model['params'], scoring=rmse_score)
        gsearch.fit(X_train, y_train)

        results_estimator[model['name']]= gsearch.best_estimator_
        results_score[model['name']]= gsearch.best_score_

    best_model_name = max(results_score,key=results_score.get)
    best_model = results_estimator[best_model_name]

    context['task_instance'].xcom_push(key='best_model', value=best_model)
    context['task_instance'].xcom_push(key='best_model_name', value=best_model_name)
    # return best_model, best_model_name


def save_model(**context):
    best_model = context['task_instance'].xcom_pull(task_ids='model_gridsearchCV', key='best_model')
    best_model_name = context['task_instance'].xcom_pull(task_ids='model_gridsearchCV', key='best_model_name')

    filename = best_model_name + '_'+ datetime.now().strftime("%Y%m%d_%H%M%S") + '.pkl'
    pickle.dump(best_model, open('./model/'+filename,'wb'))
    context['task_instance'].xcom_push(key='filename', value=filename)
    # return filename


def import_model_predict(**context):
    filename = context['task_instance'].xcom_pull(task_ids='save_model', key='filename')
    X_test = context['task_instance'].xcom_pull(task_ids='split_dataset', key='X_test')
    y_test = context['task_instance'].xcom_pull(task_ids='split_dataset', key='y_test')

    loaded_model = pickle.load(open('./model/'+filename, 'rb'))
    result = loaded_model.score(X_test, y_test)

    print(result)

    y_true = y_test.values
    y_pred = loaded_model.predict(X_test)
    regression_results(y_true, y_pred)






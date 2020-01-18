'''
Created on Jun 3, 2019

@author: Leo82
'''

import datetime as dt
import sqlite3
import logging
logging.basicConfig(level=logging.INFO, format= '%(asctime)s - MAIN - [%(name)s] [%(levelname)s] : %(message)s')

import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.linear_model import LinearRegression

import os.path
save_path = r'C:\Users\leo82\eclipse-workspace\capstone\results\test'

# This info used for access to database.  If recreating, match your database to database spec, using same data source.
HOST            = #####
USER            = #####
#PASSWORD        = #####
DB_NAME         = #####

class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]

class TradingDays(object):
    def __init__(self, start_date, end_date): # input dates format yyyy-mm-dd
        self.start_date = start_date
        self.end_date = end_date
        self.daily = self.get_trading_days()

    def _get_trading_close_holidays(self, year):
        inst = USTradingCalendar()
        return inst.holidays(dt.datetime(year-1, 12, 31), dt.datetime(year, 12, 31))

    def get_trading_days(self): 
        all_bdays = pd.bdate_range(self.start_date, self.end_date)
        years = set(map(lambda d: d.year, all_bdays))
        
        holidays =[]
        for year in years:
            holidays += self._get_trading_close_holidays(year) 
        return list(filter(lambda d: d not in holidays, all_bdays))        

    def get_trading_days_weekly(self):
        days = self.daily
        days_df = pd.DataFrame(index=days)
        days_weekly = days_df.groupby((days_df.index.year, days_df.index.week)).apply(pd.Series.tail,1).reset_index(level=(0,1), drop=True)
             
        days_weekly.to_csv(os.path.join(save_path, r'days_wk.csv'))
        
        return days_weekly.index.tolist()

    def get_trading_days_monthly(self):
        days = self.daily
        days_df = pd.DataFrame(index=days)
        days_monthly = days_df.groupby((days_df.index.year, days_df.index.month)).apply(pd.Series.tail,1).reset_index(level=(0,1), drop=True)
        return days_monthly.index.tolist()

def sqlite_connect():
    try:
        conn = sqlite3.connect(r'C:\Users\leo82\eclipse-workspace\capstone\db\database.db', check_same_thread=False, timeout=3000)
        return conn
    except:
        return None
    
def psql_connect():
    import psycopg2 # it is installed in warriors machine (scu computer)
    com = "host=%s user=%s dbname=%s" % (HOST, USER, DB_NAME)
    try:
        conn = psycopg2.connect(com)
    except:
        logging.warning('unable to connect to: %s' % DB_NAME)
        exit(1)
    return conn

def predict_components_AR(principal_components):
    results_aic = pd.DataFrame(columns=['trading_day','component','lag_length','aic'])
    predictions = pd.DataFrame()
    for component in range(principal_components.shape[1]):
        train = principal_components[:, component] 
        model = AR(train)
        model_fit = model.fit(ic = 'aic') # Criterion used for selecting the optimal lag length.
        prediction = model_fit.predict(start=len(train), end=len(train), dynamic=False)
        predictions[str(component+1)] = prediction
        results0 = {'component': component+1}
        results0['lag_length'] = model_fit.k_ar
        results0['aic'] = model_fit.aic
        results_aic = results_aic.append(results0, ignore_index=True)
    return results_aic, predictions

def predict_components_ARMA(principal_components):
    predictions = pd.DataFrame()
    for component in range(principal_components.shape[1]):
        train = principal_components[:, component] 
        model = auto_arima(train, start_p=1, start_q=0,
                           information_criterion='aic', 
                           test='adf',                  # use adftest to find optimal 'd'
                           max_p=20, max_q=1,           # maximum p and q
                           m=1,                         # frequency of series
                           d=0,                         # we know components are stationary
                           seasonal=False,              # No Seasonality
                           start_P=0, 
                           D=0, 
                           trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True
                           )
        model_fit = model.fit(train) 
        prediction = model_fit.predict(n_periods=1)
        predictions[str(component+1)] = prediction
    return predictions

def predict_returns_lreg(principal_components, returns_matrix, predicted_components):
    train_x = principal_components
    train_y = returns_matrix
    predicted_components = predicted_components
    linear_regr = LinearRegression()
    linear_regr.fit(train_x, train_y)
    lreg_returns_prediction = linear_regr.predict(predicted_components)
    return lreg_returns_prediction
    
def predict_components_AR_roll(principal_components, n_days_to_predict=1, component_to_plot = 1):
    predictions = pd.DataFrame()
    expected = pd.DataFrame()
    for component in range(principal_components.shape[1]):
        test = principal_components[len(principal_components)-n_days_to_predict:, component]
        n_roll = 0
        predictions0_list = []
        while n_roll < n_days_to_predict:
            train = principal_components[0+n_roll:len(principal_components)-n_days_to_predict+n_roll, component] 
            model = AR(train)
            model_fit = model.fit(ic = 'aic') # Criterion used for selecting the optimal lag length.
            predictions0 = model_fit.predict(start=len(train), end=len(train), dynamic=False)[0]
            predictions0_list.append(predictions0)
            n_roll += 1
        predictions[str(component+1)] = predictions0_list
        expected[str(component+1)] = test
    results = {}
    results['predictions'] = predictions
    results['expected'] = expected
    results['MAE'] = sum(abs(predictions.values-expected.values))/len(expected)
    # plotting a component for illustration of results
    plt.grid(True)
    plt.title('Principal Component #%d Prediction' % component_to_plot)
    plt.xlabel('Number of days')
    plt.ylabel('Principal Component Value')
    plt.plot(predictions[str(component_to_plot)], color = 'r', label = 'Predicted')
    plt.plot(expected[str(component_to_plot)], color = 'b', label = 'Expected')
    plt.legend(loc='best')
    return predictions #results, plt.show()

def predict_components_VAR(principal_components, n_days_to_predict=1, components_to_use=24, component_to_plot=1):
    train = principal_components[:len(principal_components)-n_days_to_predict,0:components_to_use]
    test = principal_components[len(principal_components)-n_days_to_predict:,0:components_to_use]
    model = VAR(endog=train)
    model_fit = model.fit(ic='aic')
    prediction = model_fit.forecast(model_fit.y, steps=len(test))
    results = {}
    results['predictions'] = prediction
    results['expected'] = test
    results['MAE'] = sum(abs(prediction-test))/len(test)
    # plotting a component for illustration of results
    plt.grid(True)
    plt.title('Principal Component #%d Prediction' % component_to_plot)
    plt.xlabel('Number of days')
    plt.ylabel('Principal Component Value')
    plt.plot(prediction[:,component_to_plot-1], color = 'r', label = 'Predicted')
    plt.plot(test[:,component_to_plot-1], color = 'b', label = 'Expected')
    plt.legend(loc='best')
    return prediction #results, plt.show()

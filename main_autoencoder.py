'''
Created on Sep 14, 2019

@author: leo82
'''

import logging
logging.basicConfig(level=logging.INFO, format= '%(asctime)s - MAIN - [%(name)s] [%(levelname)s] : %(message)s')
import time
import numpy as np
import composition
import autoencoder
import returns
import market_cap
import utilities
import pandas as pd
from multiprocessing import Pool
import os.path


def autoencoder_serial(index_name, start, end, lookback = 504): # 1 year => 252 trading days
    c = composition.Composition(index_name)
    
    days = utilities.TradingDays(start, end).get_trading_days()
    basket = {}
    for d in days:
        logging.info('getting composition for %s' % d)
        basket[d] = c.get_composition(d)
    basket = pd.DataFrame.from_dict(basket, orient='index')
    r = returns.Returns(basket, start, end)
    returns_matrix_all = r.get_returns_matrix_all(frequency='daily') # match frequency to that of trading days
    
    ARprediction_aic_auto = pd.DataFrame() # trading_day, component, lag_length, aic
    lreg_returns_predictions_AR_auto = pd.DataFrame() # predicting returns using the AR predicted components
    n_day = 0
    for trading_day in range(lookback,len(returns_matrix_all)):
        returns_matrix = returns_matrix_all[n_day:trading_day]     
        n_day += 1
        # getting autoencoder components (encoded image), predicting the next trading day using AR.  
        a = autoencoder.Autoencoder(returns_matrix = returns_matrix, n_factors=100)
        ARprediction_aic1, AR_predicted_components_auto = utilities.predict_components_AR(principal_components = a.encoded_images)
        ARprediction_aic1['trading_day'] = trading_day + 1
        ARprediction_aic_auto = ARprediction_aic_auto.append(ARprediction_aic1)
        # fitting a linear regression to predict returns from components, and using the AR predicted components (next trading day) to predict the returns. Return_Matrix must be multiplied by 100 (Autoencoder input was multiplied by 100 during training)
        lreg_returns_prediction_auto = utilities.predict_returns_lreg(principal_components = a.encoded_images, returns_matrix = returns_matrix*100, predicted_components = AR_predicted_components_auto)
        lreg_returns_predictions_AR_auto = lreg_returns_predictions_AR_auto.append(pd.DataFrame(lreg_returns_prediction_auto, columns=list(range(1,returns_matrix.shape[1]+1))), ignore_index = True)
        print(lreg_returns_predictions_AR_auto.shape)
        print(lreg_returns_predictions_AR_auto)
    AR_auto_same_direction = np.where(lreg_returns_predictions_AR_auto.values*returns_matrix_all[lookback:] > 0, 1, 0)
    AR_auto_same_direction_df = pd.DataFrame(AR_auto_same_direction, columns=list(range(1,returns_matrix_all.shape[1]+1)))
    AR_auto_same_direction_percent = sum(np.where(lreg_returns_predictions_AR_auto.values*returns_matrix_all[lookback:] > 0, 1, 0))/len(returns_matrix_all[lookback:])
    AR_auto_same_direction_percent_df = pd.DataFrame(AR_auto_same_direction_percent)
    return ARprediction_aic_auto, lreg_returns_predictions_AR_auto, AR_auto_same_direction_df, AR_auto_same_direction_percent_df

 
class MultiprocessingAutoencoder(object):
    
    def __init__(self, index_name, start, end, frequency='daily', lookback = 504): # 1 year => 252 trading days
        self.index_name = index_name
        self.start = start
        self.end = end
        self.frequency = frequency
        self.lookback = lookback
        self.basket = self._get_days_composition()
        self.returns_matrix_all = self._get_returns_matrix_all()
        self.market_cap_matrix_all = self._get_market_cap_matrix_all()
 
    def _get_days_composition(self):
        c = composition.Composition(self.index_name)
        assert (self.frequency=='daily' or self.frequency=='weekly' or self.frequency=='monthly'), 'frequency must be "daily", "weekly", or "monthly"'
        if self.frequency=='daily':
            days = utilities.TradingDays(self.start, self.end).get_trading_days()
        elif self.frequency=='weekly':
            days = utilities.TradingDays(self.start, self.end).get_trading_days_weekly()
        else:
            days = utilities.TradingDays(self.start, self.end).get_trading_days_monthly()
        basket ={}
        for d in days:
            logging.info('getting composition for %s' % d)
            basket[d] = c.get_composition(d)
        basket = pd.DataFrame.from_dict(basket, orient='index')
        return basket
 
    def _get_returns_matrix_all(self): 
        r = returns.Returns(self.basket, self.start, self.end)
        returns_matrix_all = r.get_returns_matrix_all(self.frequency) # match frequency to that of trading days
        return returns_matrix_all
    
    def _get_market_cap_matrix_all(self):
        mc = market_cap.MarketCap(self.basket, self.start, self.end)
        market_cap_matrix_all = mc.market_cap_matrix_all
        return market_cap_matrix_all
    
    def autoencoder_one_day(self, trading_day):
        returns_matrix = self.returns_matrix_all[trading_day-self.lookback:trading_day]     
        # getting autoencoder components (encoded image), predicting the next trading day using AR 
        a = autoencoder.Autoencoder(returns_matrix = returns_matrix, n_factors=100)
        ARprediction_aic, AR_predicted_components_auto = utilities.predict_components_AR(principal_components = a.encoded_images)
        ARprediction_aic['trading_day'] = trading_day + 1
        # fitting a linear regression to predict returns from components, and using the AR predicted components (next trading day) to predict the returns. 
        # Return_Matrix must be multiplied by 100 (Autoencoder input was multiplied by 100 during training)
        lreg_returns_prediction_auto_AR = utilities.predict_returns_lreg(principal_components = a.encoded_images, returns_matrix = returns_matrix*100, predicted_components = AR_predicted_components_auto)
        lreg_returns_prediction_AR_auto = pd.DataFrame(lreg_returns_prediction_auto_AR, columns=list(range(1,returns_matrix.shape[1]+1)))
        lreg_returns_prediction_AR_auto['trading_day'] = trading_day + 1
        # predicting returns from AR components using decoder
        deco_returns_prediction_auto_AR = a.decoder.predict(AR_predicted_components_auto.to_numpy())
        deco_returns_prediction_AR_auto = pd.DataFrame(deco_returns_prediction_auto_AR, columns=list(range(1,returns_matrix.shape[1]+1)))
        deco_returns_prediction_AR_auto['trading_day'] = trading_day + 1
        # predicting the next trading day using ARMA 
        ARMA_predicted_components_auto = utilities.predict_components_ARMA(principal_components = a.encoded_images)
        # fitting a linear regression to predict returns from components, and using the ARMA predicted components (next trading day) to predict the returns. 
        # Return_Matrix must be multiplied by 100 (Autoencoder input was multiplied by 100 during training)
        lreg_returns_prediction_auto_ARMA = utilities.predict_returns_lreg(principal_components = a.encoded_images, returns_matrix = returns_matrix*100, predicted_components = ARMA_predicted_components_auto)
        lreg_returns_prediction_ARMA_auto = pd.DataFrame(lreg_returns_prediction_auto_ARMA, columns=list(range(1,returns_matrix.shape[1]+1)))
        lreg_returns_prediction_ARMA_auto['trading_day'] = trading_day + 1
        # predicting returns from ARMA components using decoder
        deco_returns_prediction_auto_ARMA = a.decoder.predict(ARMA_predicted_components_auto.to_numpy())
        deco_returns_prediction_ARMA_auto = pd.DataFrame(deco_returns_prediction_auto_ARMA, columns=list(range(1,returns_matrix.shape[1]+1)))
        deco_returns_prediction_ARMA_auto['trading_day'] = trading_day + 1
        print(lreg_returns_prediction_ARMA_auto)
        return ARprediction_aic, lreg_returns_prediction_AR_auto, lreg_returns_prediction_ARMA_auto, deco_returns_prediction_AR_auto, deco_returns_prediction_ARMA_auto
    


if __name__ == '__main__':
    
    index_name = 'sp500'
    start_date = '1992-11-01'
    end_date = '2018-12-31'
    frequency = 'monthly'
    lookback = 252 # 252 trading days in 1 year
    save_path = r'C:\Users\leo82\eclipse-workspace\capstone\results\main_autoencoder'
    
    # running using multiprocessing
    start_time = time.time()
    ma = MultiprocessingAutoencoder(index_name, start_date, end_date, frequency = frequency, lookback = lookback)  
    trading_days = range(lookback,len(ma.returns_matrix_all))
    p = Pool()
    results = p.map(ma.autoencoder_one_day, trading_days)
    ARprediction_aic_auto = pd.DataFrame()
    lreg_returns_predictions_AR_auto = pd.DataFrame()
    lreg_returns_predictions_ARMA_auto = pd.DataFrame()
    deco_returns_predictions_AR_auto = pd.DataFrame()
    deco_returns_predictions_ARMA_auto = pd.DataFrame()
    # unpacking results
    for result in results:
        ARprediction_aic_auto = ARprediction_aic_auto.append(result[0])
        lreg_returns_predictions_AR_auto = lreg_returns_predictions_AR_auto.append(result[1])    
        lreg_returns_predictions_ARMA_auto = lreg_returns_predictions_ARMA_auto.append(result[2])
        deco_returns_predictions_AR_auto = deco_returns_predictions_AR_auto.append(result[3])    
        deco_returns_predictions_ARMA_auto = deco_returns_predictions_ARMA_auto.append(result[4])
    print(ARprediction_aic_auto)
    ARprediction_aic_auto.to_csv(os.path.join(save_path, frequency + r'\mp_ARprediction_aic_auto.csv'))
    print(lreg_returns_predictions_AR_auto)
    lreg_returns_predictions_AR_auto.to_csv(os.path.join(save_path, frequency + r'\mp_lreg_returns_predictions_AR_auto.csv'))
    lreg_returns_predictions_AR_auto.drop(columns=['trading_day'], inplace = True)
    print(lreg_returns_predictions_ARMA_auto)
    lreg_returns_predictions_ARMA_auto.to_csv(os.path.join(save_path, frequency + r'\mp_lreg_returns_predictions_ARMA_auto.csv'))
    lreg_returns_predictions_ARMA_auto.drop(columns=['trading_day'], inplace = True)
    print(deco_returns_predictions_AR_auto)
    deco_returns_predictions_AR_auto.to_csv(os.path.join(save_path, frequency + r'\mp_deco_returns_predictions_AR_auto.csv'))
    deco_returns_predictions_AR_auto.drop(columns=['trading_day'], inplace = True)
    print(deco_returns_predictions_ARMA_auto)
    deco_returns_predictions_ARMA_auto.to_csv(os.path.join(save_path, frequency + r'\mp_deco_returns_predictions_ARMA_auto.csv'))
    deco_returns_predictions_ARMA_auto.drop(columns=['trading_day'], inplace = True)
    # getting same direction, percent by day and by stock for AR / lreg
    returns_actual = pd.DataFrame(ma.returns_matrix_all[lookback:])
    returns_actual.to_csv(os.path.join(save_path, frequency + r'\returns_actual_' + frequency + r'.csv'), index=False)
    AR_auto_same_direction = np.where(lreg_returns_predictions_AR_auto.values*ma.returns_matrix_all[lookback:] > 0, 1, 0)
    AR_auto_same_direction_df = pd.DataFrame(AR_auto_same_direction, columns=list(range(1,ma.returns_matrix_all.shape[1]+1)))
    AR_auto_same_direction_percent_stock = 100*sum(np.where(lreg_returns_predictions_AR_auto.values*ma.returns_matrix_all[lookback:] > 0, 1, 0))/len(ma.returns_matrix_all[lookback:])
    AR_auto_same_direction_percent_stock_df = pd.DataFrame(AR_auto_same_direction_percent_stock)
    AR_auto_same_direction_percent_day = 100*AR_auto_same_direction.sum(axis=1)/AR_auto_same_direction.shape[1]
    AR_auto_same_direction_percent_day_df = pd.DataFrame(AR_auto_same_direction_percent_day)
    print(AR_auto_same_direction_df)
    AR_auto_same_direction_df.to_csv(os.path.join(save_path, frequency + r'\mp_AR_auto_same_direction_df.csv'))
    print(AR_auto_same_direction_percent_stock_df)
    AR_auto_same_direction_percent_stock_df.to_csv(os.path.join(save_path, frequency + r'\mp_AR_auto_same_direction_percent_stock_df.csv'))
    print(AR_auto_same_direction_percent_day_df)
    AR_auto_same_direction_percent_day_df.to_csv(os.path.join(save_path, frequency + r'\mp_AR_auto_same_direction_percent_day_df.csv'))
    # getting same direction, percent by day and by stock for AR / decoder
    AR_deco_auto_same_direction = np.where(deco_returns_predictions_AR_auto.values*ma.returns_matrix_all[lookback:] > 0, 1, 0)
    AR_deco_auto_same_direction_df = pd.DataFrame(AR_deco_auto_same_direction, columns=list(range(1,ma.returns_matrix_all.shape[1]+1)))
    AR_deco_auto_same_direction_percent_stock = 100*sum(np.where(deco_returns_predictions_AR_auto.values*ma.returns_matrix_all[lookback:] > 0, 1, 0))/len(ma.returns_matrix_all[lookback:])
    AR_deco_auto_same_direction_percent_stock_df = pd.DataFrame(AR_deco_auto_same_direction_percent_stock)
    AR_deco_auto_same_direction_percent_day = 100*AR_deco_auto_same_direction.sum(axis=1)/AR_deco_auto_same_direction.shape[1]
    AR_deco_auto_same_direction_percent_day_df = pd.DataFrame(AR_deco_auto_same_direction_percent_day)
    print(AR_deco_auto_same_direction_df)
    AR_deco_auto_same_direction_df.to_csv(os.path.join(save_path, frequency + r'\mp_AR_deco_auto_same_direction_df.csv'))
    print(AR_deco_auto_same_direction_percent_stock_df)
    AR_deco_auto_same_direction_percent_stock_df.to_csv(os.path.join(save_path, frequency + r'\mp_AR_deco_auto_same_direction_percent_stock_df.csv'))
    print(AR_deco_auto_same_direction_percent_day_df)
    AR_deco_auto_same_direction_percent_day_df.to_csv(os.path.join(save_path, frequency + r'\mp_AR_deco_auto_same_direction_percent_day_df.csv'))
    # getting same direction, percent by day and by stock for ARMA / lreg
    ARMA_auto_same_direction = np.where(lreg_returns_predictions_ARMA_auto.values*ma.returns_matrix_all[lookback:] > 0, 1, 0)
    ARMA_auto_same_direction_df = pd.DataFrame(ARMA_auto_same_direction, columns=list(range(1,ma.returns_matrix_all.shape[1]+1)))
    ARMA_auto_same_direction_percent_stock = 100*sum(np.where(lreg_returns_predictions_ARMA_auto.values*ma.returns_matrix_all[lookback:] > 0, 1, 0))/len(ma.returns_matrix_all[lookback:])
    ARMA_auto_same_direction_percent_stock_df = pd.DataFrame(ARMA_auto_same_direction_percent_stock)
    ARMA_auto_same_direction_percent_day = 100*ARMA_auto_same_direction.sum(axis=1)/ARMA_auto_same_direction.shape[1]
    ARMA_auto_same_direction_percent_day_df = pd.DataFrame(ARMA_auto_same_direction_percent_day)
    print(ARMA_auto_same_direction_df)
    ARMA_auto_same_direction_df.to_csv(os.path.join(save_path, frequency + r'\mp_ARMA_auto_same_direction_df.csv'))
    print(ARMA_auto_same_direction_percent_stock_df)
    ARMA_auto_same_direction_percent_stock_df.to_csv(os.path.join(save_path, frequency + r'\mp_ARMA_auto_same_direction_percent_stock_df.csv'))
    print(ARMA_auto_same_direction_percent_day_df)
    ARMA_auto_same_direction_percent_day_df.to_csv(os.path.join(save_path, frequency + r'\mp_ARMA_auto_same_direction_percent_day_df.csv'))
    # getting same direction, percent by day and by stock for ARMA / decoder
    ARMA_deco_auto_same_direction = np.where(deco_returns_predictions_ARMA_auto.values*ma.returns_matrix_all[lookback:] > 0, 1, 0)
    ARMA_deco_auto_same_direction_df = pd.DataFrame(ARMA_deco_auto_same_direction, columns=list(range(1,ma.returns_matrix_all.shape[1]+1)))
    ARMA_deco_auto_same_direction_percent_stock = 100*sum(np.where(deco_returns_predictions_ARMA_auto.values*ma.returns_matrix_all[lookback:] > 0, 1, 0))/len(ma.returns_matrix_all[lookback:])
    ARMA_deco_auto_same_direction_percent_stock_df = pd.DataFrame(ARMA_deco_auto_same_direction_percent_stock)
    ARMA_deco_auto_same_direction_percent_day = 100*ARMA_deco_auto_same_direction.sum(axis=1)/ARMA_deco_auto_same_direction.shape[1]
    ARMA_deco_auto_same_direction_percent_day_df = pd.DataFrame(ARMA_deco_auto_same_direction_percent_day)
    print(ARMA_deco_auto_same_direction_df)
    ARMA_deco_auto_same_direction_df.to_csv(os.path.join(save_path, frequency + r'\mp_ARMA_deco_auto_same_direction_df.csv'))
    print(ARMA_deco_auto_same_direction_percent_stock_df)
    ARMA_deco_auto_same_direction_percent_stock_df.to_csv(os.path.join(save_path, frequency + r'\mp_ARMA_deco_auto_same_direction_percent_stock_df.csv'))
    print(ARMA_deco_auto_same_direction_percent_day_df)
    ARMA_deco_auto_same_direction_percent_day_df.to_csv(os.path.join(save_path, frequency + r'\mp_ARMA_deco_auto_same_direction_percent_day_df.csv'))
    # getting market_cap of those predicted in right direction AR / lreg
    market_cap_day_total = ma.market_cap_matrix_all[lookback:].sum(axis=1)
    market_cap_same_direction_AR = ma.market_cap_matrix_all[lookback:]*AR_auto_same_direction
    market_cap_same_direction_AR_percent_day = 100*market_cap_same_direction_AR.sum(axis=1)/market_cap_day_total
    market_cap_same_direction_AR_percent_day_df = pd.DataFrame(market_cap_same_direction_AR_percent_day)
    print(market_cap_same_direction_AR_percent_day_df)
    market_cap_same_direction_AR_percent_day_df.to_csv(os.path.join(save_path, frequency + r'\mp_market_cap_same_direction_AR_percent_day_df.csv'))
    # getting market_cap of those predicted in right direction AR / decoder
    market_cap_same_direction_AR_deco = ma.market_cap_matrix_all[lookback:]*AR_deco_auto_same_direction
    market_cap_same_direction_AR_deco_percent_day = 100*market_cap_same_direction_AR_deco.sum(axis=1)/market_cap_day_total
    market_cap_same_direction_AR_deco_percent_day_df = pd.DataFrame(market_cap_same_direction_AR_deco_percent_day)
    print(market_cap_same_direction_AR_deco_percent_day_df)
    market_cap_same_direction_AR_deco_percent_day_df.to_csv(os.path.join(save_path, frequency + r'\mp_market_cap_same_direction_AR_deco_percent_day_df.csv'))
    # getting market_cap of those predicted in right direction ARMA / lreg
    market_cap_same_direction_ARMA = ma.market_cap_matrix_all[lookback:]*ARMA_auto_same_direction
    market_cap_same_direction_ARMA_percent_day = 100*market_cap_same_direction_ARMA.sum(axis=1)/market_cap_day_total
    market_cap_same_direction_ARMA_percent_day_df = pd.DataFrame(market_cap_same_direction_ARMA_percent_day)
    print(market_cap_same_direction_ARMA_percent_day_df)
    market_cap_same_direction_ARMA_percent_day_df.to_csv(os.path.join(save_path, frequency + r'\mp_market_cap_same_direction_ARMA_percent_day_df.csv'))
    # getting market_cap of those predicted in right direction ARMA / decoder
    market_cap_same_direction_ARMA_deco = ma.market_cap_matrix_all[lookback:]*ARMA_deco_auto_same_direction
    market_cap_same_direction_ARMA_deco_percent_day = 100*market_cap_same_direction_ARMA_deco.sum(axis=1)/market_cap_day_total
    market_cap_same_direction_ARMA_deco_percent_day_df = pd.DataFrame(market_cap_same_direction_ARMA_deco_percent_day)
    print(market_cap_same_direction_ARMA_deco_percent_day_df)
    market_cap_same_direction_ARMA_deco_percent_day_df.to_csv(os.path.join(save_path, frequency + r'\mp_market_cap_same_direction_ARMA_deco_percent_day_df.csv'))
    time_mp = time.time() - start_time

#     # running without multiprocessing
#     start_time = time.time()
#     ARprediction_aic_auto, lreg_returns_predictions_AR_auto, AR_auto_same_direction_df, AR_auto_same_direction_percent_df= autoencoder_serial(index_name, start_date, end_date, lookback = lookback)
#     print(ARprediction_aic_auto)
#     ARprediction_aic_auto.to_csv(r'.\results\main_autoencoder\ARprediction_aic_auto.csv')
#     print(lreg_returns_predictions_AR_auto)
#     lreg_returns_predictions_AR_auto.to_csv(r'.\results\main_autoencoder\lreg_returns_predictions_AR_auto.csv')
#     print(AR_auto_same_direction_df)
#     AR_auto_same_direction_df.to_csv(r'.\results\main_autoencoder\AR_auto_same_direction_df.csv')
#     print(AR_auto_same_direction_percent_df)
#     AR_auto_same_direction_percent_df.to_csv(r'.\results\main_autoencoder\AR_auto_same_direction_percent_df.csv')
#     time_no_mp = time.time() - start_time
    
    # printing time for runs with and without multiprocessing
    print("Multiprocessing run: --- %s seconds ---" % time_mp)
#     print("Serial run: --- %s seconds ---" % time_no_mp)
    
    
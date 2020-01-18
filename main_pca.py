'''
Created on Sep 26, 2019

@author: leo82
'''

import logging
logging.basicConfig(level=logging.INFO, format= '%(asctime)s - MAIN - [%(name)s] [%(levelname)s] : %(message)s')
import time
import numpy as np
import composition
import pca
import returns
import market_cap
import utilities
import pandas as pd
import os.path



def main(index_name, start, end, lookback = 504, frequency='monthly'): # 1 year => 252 trading days
    c = composition.Composition(index_name)
    assert (frequency=='daily' or frequency=='weekly' or frequency=='monthly'), 'frequency must be "daily", "weekly", or "monthly"'
    if frequency=='daily':
        days = utilities.TradingDays(start, end).get_trading_days()
    elif frequency=='weekly':
        days = utilities.TradingDays(start, end).get_trading_days_weekly()
    else:
        days = utilities.TradingDays(start, end).get_trading_days_monthly()
    basket ={}
    for d in days:
        logging.info('getting composition for %s' % d)
        basket[d] = c.get_composition(d)
    basket = pd.DataFrame.from_dict(basket, orient='index')
    r = returns.Returns(basket, start, end)
    returns_matrix_all = r.get_returns_matrix_all(frequency=frequency) # match frequency to that of trading days
    
    ARprediction_aic_pca = pd.DataFrame() # trading_day, component, lag_length, aic
    lreg_returns_predictions_AR_pca = pd.DataFrame() # predicting returns using the AR predicted components
    lreg_returns_predictions_ARMA_pca = pd.DataFrame() # predicting returns using the AR predicted components
    n_day = 0
    for trading_day in range(lookback,len(returns_matrix_all)):
        returns_matrix = returns_matrix_all[n_day:trading_day]     
        n_day += 1
        # getting pca components, predicting next trading day using AR.  
        p = pca.PrincipalComponentAnalysis(returns_matrix, n_components = 100) 
        ARprediction_aic0, AR_predicted_components_pca = utilities.predict_components_AR(principal_components = p.principal_components)
        ARprediction_aic0['trading_day'] = trading_day + 1
        ARprediction_aic_pca = ARprediction_aic_pca.append(ARprediction_aic0)
        # fitting a linear regression to predict returns from components, and using the AR predicted components (next trading day) to predict the returns
        lreg_returns_prediction_pca_AR = utilities.predict_returns_lreg(principal_components = p.principal_components, returns_matrix = returns_matrix, predicted_components = AR_predicted_components_pca)
        lreg_returns_predictions_AR_pca = lreg_returns_predictions_AR_pca.append(pd.DataFrame(lreg_returns_prediction_pca_AR, columns=list(range(1,returns_matrix.shape[1]+1))), ignore_index = True)
        # predicting the next trading day using ARMA 
        ARMA_predicted_components_pca = utilities.predict_components_ARMA(principal_components = p.principal_components)
        ####
        ARMA_predicted_components_pca_df = pd.DataFrame(ARMA_predicted_components_pca)
        print('trading_day =',trading_day, '\n ARMA_predicted_components_pca \n', ARMA_predicted_components_pca_df)
        ####
        # fitting a linear regression to predict returns from components, and using the ARMA predicted components (next trading day) to predict the returns. 
        lreg_returns_prediction_pca_ARMA = utilities.predict_returns_lreg(principal_components = p.principal_components, returns_matrix = returns_matrix, predicted_components = ARMA_predicted_components_pca)
        lreg_returns_predictions_ARMA_pca = lreg_returns_predictions_ARMA_pca.append(pd.DataFrame(lreg_returns_prediction_pca_ARMA, columns=list(range(1,returns_matrix.shape[1]+1))), ignore_index = True)
    # getting same direction, percent by day and by stock for AR
    returns_actual = pd.DataFrame(returns_matrix_all[lookback:])
    returns_actual.to_csv(os.path.join(save_path, frequency + r'\returns_actual_' + frequency + r'.csv'), index=False)
    AR_pca_same_direction = np.where(lreg_returns_predictions_AR_pca.values*returns_matrix_all[lookback:] > 0, 1, 0)
    AR_pca_same_direction_df = pd.DataFrame(AR_pca_same_direction, columns=list(range(1,returns_matrix.shape[1]+1)))
    AR_pca_same_direction_percent_stock = 100*sum(np.where(lreg_returns_predictions_AR_pca.values*returns_matrix_all[lookback:] > 0, 1, 0))/len(returns_matrix_all[lookback:])
    AR_pca_same_direction_percent_stock_df = pd.DataFrame(AR_pca_same_direction_percent_stock)
    AR_pca_same_direction_percent_day = 100*AR_pca_same_direction.sum(axis=1)/AR_pca_same_direction.shape[1]
    AR_pca_same_direction_percent_day_df = pd.DataFrame(AR_pca_same_direction_percent_day)
    # getting same direction, percent by day and by stock for ARMA
    ARMA_pca_same_direction = np.where(lreg_returns_predictions_ARMA_pca.values*returns_matrix_all[lookback:] > 0, 1, 0)
    ARMA_pca_same_direction_df = pd.DataFrame(ARMA_pca_same_direction, columns=list(range(1,returns_matrix.shape[1]+1)))
    ARMA_pca_same_direction_percent_stock = 100*sum(np.where(lreg_returns_predictions_ARMA_pca.values*returns_matrix_all[lookback:] > 0, 1, 0))/len(returns_matrix_all[lookback:])
    ARMA_pca_same_direction_percent_stock_df = pd.DataFrame(ARMA_pca_same_direction_percent_stock)
    ARMA_pca_same_direction_percent_day = 100*ARMA_pca_same_direction.sum(axis=1)/ARMA_pca_same_direction.shape[1]
    ARMA_pca_same_direction_percent_day_df = pd.DataFrame(ARMA_pca_same_direction_percent_day)
    # getting market_cap of those predicted in right direction AR
    mc = market_cap.MarketCap(basket, start, end)
    market_cap_matrix_all = mc.market_cap_matrix_all
    market_cap_day_total = market_cap_matrix_all[lookback:].sum(axis=1)
    market_cap_same_direction_AR = market_cap_matrix_all[lookback:]*AR_pca_same_direction
    market_cap_same_direction_AR_percent_day = 100*market_cap_same_direction_AR.sum(axis=1)/market_cap_day_total
    market_cap_same_direction_AR_percent_day_df = pd.DataFrame(market_cap_same_direction_AR_percent_day)
    # getting market_cap of those predicted in right direction ARMA
    market_cap_same_direction_ARMA = market_cap_matrix_all[lookback:]*ARMA_pca_same_direction
    market_cap_same_direction_ARMA_percent_day = 100*market_cap_same_direction_ARMA.sum(axis=1)/market_cap_day_total
    market_cap_same_direction_ARMA_percent_day_df = pd.DataFrame(market_cap_same_direction_ARMA_percent_day)   
    
    return ARprediction_aic_pca, lreg_returns_predictions_AR_pca, lreg_returns_predictions_ARMA_pca, AR_pca_same_direction_df, AR_pca_same_direction_percent_stock_df, \
        AR_pca_same_direction_percent_day_df, ARMA_pca_same_direction_df, ARMA_pca_same_direction_percent_stock_df, ARMA_pca_same_direction_percent_day_df, \
        market_cap_same_direction_AR_percent_day_df, market_cap_same_direction_ARMA_percent_day_df



if __name__ == '__main__':
    start_time = time.time()
    
    index_name = 'sp500'
    start_date = '1992-11-01'
    end_date = '2018-12-31'
    frequency = 'monthly'
    lookback = 252 # 252 trading days in 1 year
    
    ARprediction_aic_pca, lreg_returns_predictions_AR_pca, lreg_returns_predictions_ARMA_pca, AR_pca_same_direction_df, AR_pca_same_direction_percent_stock_df, \
        AR_pca_same_direction_percent_day_df, ARMA_pca_same_direction_df, ARMA_pca_same_direction_percent_stock_df, ARMA_pca_same_direction_percent_day_df, \
        market_cap_same_direction_AR_percent_day_df, market_cap_same_direction_ARMA_percent_day_df = main(index_name, start_date, end_date, lookback = lookback, frequency = frequency) 
    save_path = r'C:\Users\leo82\eclipse-workspace\capstone\results\main_pca'
    print('ARprediction_aic_pca \n', ARprediction_aic_pca)
    ARprediction_aic_pca.to_csv(os.path.join(save_path, frequency + r'\ARprediction_aic_pca.csv'))
    print('lreg_returns_predictions_AR_pca \n', lreg_returns_predictions_AR_pca)
    lreg_returns_predictions_AR_pca.to_csv(os.path.join(save_path, frequency + r'\lreg_returns_predictions_AR_pca.csv'))
    print('lreg_returns_predictions_ARMA_pca \n', lreg_returns_predictions_ARMA_pca)
    lreg_returns_predictions_ARMA_pca.to_csv(os.path.join(save_path, frequency + r'\lreg_returns_predictions_ARMA_pca.csv'))
    print('AR_pca_same_direction_df \n', AR_pca_same_direction_df)
    AR_pca_same_direction_df.to_csv(os.path.join(save_path, frequency + r'\AR_pca_same_direction_df.csv'))
    print('AR_pca_same_direction_percent_stock_df', AR_pca_same_direction_percent_stock_df)
    AR_pca_same_direction_percent_stock_df.to_csv(os.path.join(save_path, frequency + r'\AR_pca_same_direction_percent_stock_df.csv'))
    print('AR_pca_same_direction_percent_day_df \n', AR_pca_same_direction_percent_day_df)
    AR_pca_same_direction_percent_day_df.to_csv(os.path.join(save_path, frequency + r'\AR_pca_same_direction_percent_day_df.csv'))
    print('ARMA_pca_same_direction_df \n', ARMA_pca_same_direction_df)
    ARMA_pca_same_direction_df.to_csv(os.path.join(save_path, frequency + r'\ARMA_pca_same_direction_df.csv'))
    print('ARMA_pca_same_direction_percent_stock_df', ARMA_pca_same_direction_percent_stock_df)
    ARMA_pca_same_direction_percent_stock_df.to_csv(os.path.join(save_path, frequency + r'\ARMA_pca_same_direction_percent_stock_df.csv'))
    print('ARMA_pca_same_direction_percent_day_df \n', ARMA_pca_same_direction_percent_day_df)
    ARMA_pca_same_direction_percent_day_df.to_csv(os.path.join(save_path, frequency + r'\ARMA_pca_same_direction_percent_day_df.csv'))
    print('market_cap_same_direction_AR_percent_day_df \n', market_cap_same_direction_AR_percent_day_df)
    market_cap_same_direction_AR_percent_day_df.to_csv(os.path.join(save_path, frequency + r'\market_cap_same_direction_AR_percent_day_df.csv'))
    print('market_cap_same_direction_ARMA_percent_day_df \n', market_cap_same_direction_ARMA_percent_day_df)
    market_cap_same_direction_ARMA_percent_day_df.to_csv(os.path.join(save_path, frequency + r'\market_cap_same_direction_ARMA_percent_day_df.csv'))
    print("--- %s seconds ---" % (time.time() - start_time))   
    
    
    
    
    
    
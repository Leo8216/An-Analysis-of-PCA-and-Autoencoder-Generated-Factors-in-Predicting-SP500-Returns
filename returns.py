'''
Created on Jun 23, 2019

@author: leo82
'''


import logging
logging.basicConfig(level=logging.INFO, format= '%(asctime)s - MAIN - [%(name)s] [%(levelname)s] : %(message)s')
import utilities
import pandas as pd
import os.path

save_path = r'C:\Users\leo82\eclipse-workspace\capstone\results\test'



class Returns(object):
    '''
    calculates returns for individual stocks...
    '''

    def __init__(self, composition_df, start = '1984-01-01', end = '2018-12-31'):
        self.start = pd.to_datetime(start, format='%Y-%m-%d')
        self.end = pd.to_datetime(end, format='%Y-%m-%d')
        self.closing_prices = self._get_closing_prices()
        self.frame = self._get_frame_for_mergers(composition_df)
    
    def _get_closing_prices(self):
        conn = utilities.sqlite_connect()
        sql_stmt = "select date, gvkey, iid, price_close from daily"
        closing_prices_raw = pd.read_sql_query(sql_stmt, conn)
        conn.close()
        closing_prices_raw.date = pd.to_datetime(closing_prices_raw.date, format='%Y-%m-%d') 
        closing_prices_raw.price_close = pd.to_numeric(closing_prices_raw.price_close)
        closing_prices = closing_prices_raw.loc[(closing_prices_raw['date'] >= self.start) & (closing_prices_raw['date'] <= self.end)]
        closing_prices['gvkey_iid'] = closing_prices.gvkey.astype(str) + '_' + closing_prices.iid.astype(str)
        closing_prices = closing_prices[['date','gvkey_iid','price_close']]
        logging.info('loaded, formatted and sorted closing prices')
        
        closing_prices.to_csv(os.path.join(save_path, r'closing_prices.csv'), index=False)
        
        return closing_prices
    
    def _get_daily_returns(self):
        df = self.closing_prices
        df = df.sort_values(by=['gvkey_iid','date']) # sorting df by gvkey_iid and then by date. Needed for proper calculation of returns
        df['prev_price_close'] = df.price_close.shift(1)
        df['prev_gvkey_iid'] = df.gvkey_iid.shift(1)
        df['same_stock'] = df.apply(lambda row: 'Y' if  row['gvkey_iid'] == row['prev_gvkey_iid'] else 'N', axis=1) # creating same_stock column - this will allow us to filter out invalid returns
        
        df.to_csv(os.path.join(save_path, r'daily_returns_df.csv'), index=False)
        
        df = df.loc[df['same_stock'] != 'N'] # removing those date/stocks that have 'N' for same_stock
        df['ret'] = (df.price_close - df.prev_price_close)/df.prev_price_close
        daily_returns = df[['date','gvkey_iid','ret']]
        
        daily_returns.to_csv(os.path.join(save_path, r'daily_returns.csv'), index=False)
        
        logging.info('Calculated daily returns')
        return daily_returns

    def _get_weekly_returns(self):
        days = utilities.TradingDays(self.start, self.end).get_trading_days_weekly() 
        days_weekly = pd.DataFrame(days,columns=['date'])
        df = pd.merge(days_weekly, self.closing_prices, on=['date'])
        df = df.sort_values(by=['gvkey_iid','date']) # sorting df by gvkey_iid and then by date. Needed for proper calculation of returns
        df['prev_price_close'] = df.price_close.shift(1)
        df['prev_gvkey_iid'] = df.gvkey_iid.shift(1)
        df['same_stock'] = df.apply(lambda row: 'Y' if  row['gvkey_iid'] == row['prev_gvkey_iid'] else 'N', axis=1) # creating same_stock column - this will allow us to filter out invalid returns
        df = df.loc[df['same_stock'] != 'N'] # removing those date/stocks that have 'N' for same_stock
        df['ret'] = (df.price_close - df.prev_price_close)/df.prev_price_close
        weekly_returns = df[['date','gvkey_iid','ret']]
        
        weekly_returns.to_csv(os.path.join(save_path, r'weekly_returns.csv'), index=False)
        
        logging.info('Calculated weekly returns')
        return weekly_returns
    
    def _get_monthly_returns(self):
        days = utilities.TradingDays(self.start, self.end).get_trading_days_monthly() 
        days_monthly = pd.DataFrame(days,columns=['date'])
        df = pd.merge(days_monthly, self.closing_prices, on=['date'])
        df = df.sort_values(by=['gvkey_iid','date']) # sorting df by gvkey_iid and then by date. Needed for proper calculation of returns
        df['prev_price_close'] = df.price_close.shift(1)
        df['prev_gvkey_iid'] = df.gvkey_iid.shift(1)
        df['same_stock'] = df.apply(lambda row: 'Y' if  row['gvkey_iid'] == row['prev_gvkey_iid'] else 'N', axis=1) # creating same_stock column - this will allow us to filter out invalid returns
        df = df.loc[df['same_stock'] != 'N'] # removing those date/stocks that have 'N' for same_stock
        df['ret'] = (df.price_close - df.prev_price_close)/df.prev_price_close
        monthly_returns = df[['date','gvkey_iid','ret']]
        logging.info('Calculated monthly returns')
        return monthly_returns

    def _get_frame_for_mergers(self, composition_df):
        frame = pd.DataFrame({'date':[], 'gvkey_iid':[]})
        for r in range(0,len(composition_df)):
            frame0 = pd.DataFrame({'date':[], 'gvkey_iid':[]})
            frame0.gvkey_iid = composition_df.iloc[r]
            frame0.date = composition_df.index[r]
            frame = frame.append(frame0, ignore_index=True)
            
        frame.to_csv(os.path.join(save_path, r'frame.csv'))

        logging.info('Created chronological frame of dates & gvkey_iids for mergers' )
        return frame
    
    def get_returns_matrix_all(self, frequency='daily'):
        assert (frequency=='daily' or frequency=='weekly' or frequency=='monthly'), 'frequency must be "daily", "weekly", or "monthly"'
        if frequency=='daily':
            panel = pd.merge(self.frame, self._get_daily_returns(), on=['date','gvkey_iid']) # merges, if done as left join, keep some dates that do not have any prices - not trading days
        elif frequency=='weekly':
            panel = pd.merge(self.frame, self._get_weekly_returns(), on=['date','gvkey_iid'])
        else:
            panel = pd.merge(self.frame, self._get_monthly_returns(), on=['date','gvkey_iid'])
                
        panel.to_csv(os.path.join(save_path, r'panel_all.csv'), index=False)

        temp = panel.groupby('date').first()
        days = temp.index
        returns_matrix_all = pd.DataFrame()
        iterations = 0         
        for d in days:
            panel0 = panel.loc[panel['date'] == d]
            panel0 = panel0.reset_index()
            returns = panel0.ret
            returns_matrix_all = returns_matrix_all.append(returns, ignore_index=True)   
            print(". ", end="") # printing '.' for every iteration and 'Row: i' every 50
            iterations += 1
            if iterations % 50 == 0:
                print('Frequency: {}. Trading periods: {}'.format(frequency,iterations))
                
        returns_matrix_all.to_csv(os.path.join(save_path, r'rm.csv'))
        
        return returns_matrix_all.fillna(0).values[:,0:500] # dropping first row(date) since no returns are calculated for it
        
        
     
import composition
import time
   
def main(index_name, start, end, lookback = 504, frequency = 'daily'): # 1 year => 252 trading days
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
     
    basket.to_csv(os.path.join(save_path, r'basket.csv'))
       
    r = Returns(basket, start, end)
    test = r.get_returns_matrix_all(frequency=frequency)
    return test
      
if __name__ == '__main__':
    start_time = time.time()
    
    lookback = 252
    frequency = r'weekly'
    
    test = main('sp500', '2012-12-01', '2018-12-31', lookback = lookback, frequency = frequency)  
    returns_actual = pd.DataFrame(test[lookback:])
    returns_actual.to_csv(os.path.join(save_path, r'returns_actual_'+frequency+r'.csv'), index=False)
    print('\ntest shape: ', test.shape)
    print(test)
    print(returns_actual)
#     print('\nrm dtypes: \n', rm.dtypes)
#     print('\ntest dtypes: \n', test.dtypes)
    print("--- %s seconds ---" % (time.time() - start_time))




    
    
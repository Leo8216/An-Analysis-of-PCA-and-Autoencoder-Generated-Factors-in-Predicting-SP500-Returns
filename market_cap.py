'''
Created on Sep 16, 2019

@author: leo82
'''
      
import logging
logging.basicConfig(level=logging.INFO, format= '%(asctime)s - MAIN - [%(name)s] [%(levelname)s] : %(message)s')
import utilities
import pandas as pd
import os.path

save_path = r'C:\Users\leo82\eclipse-workspace\capstone\results\test'


class MarketCap(object):
    '''
    calculates returns for individual stocks...
    '''

    def __init__(self, composition_df, start = '1984-01-01', end = '2018-12-31'):
        self.start = pd.to_datetime(start, format='%Y-%m-%d')
        self.end = pd.to_datetime(end, format='%Y-%m-%d')
        self.market_cap = self._get_market_cap()
        self.frame = self._get_frame_for_merge(composition_df)
        self.market_cap_matrix_all = self._get_market_cap_matrix_all()
    
    def _get_market_cap(self):
        conn = utilities.sqlite_connect()
        sql_stmt = "select date, gvkey, iid, shares_out, price_close from daily"
        market_cap_raw = pd.read_sql_query(sql_stmt, conn)
        conn.close()
        market_cap_raw.date = pd.to_datetime(market_cap_raw.date, format='%Y-%m-%d') 
        market_cap_raw.shares_out = pd.to_numeric(market_cap_raw.shares_out)
        market_cap_raw.price_close = pd.to_numeric(market_cap_raw.price_close)
        market_cap = market_cap_raw.loc[(market_cap_raw['date'] >= self.start) & (market_cap_raw['date'] <= self.end)]
        market_cap['gvkey_iid'] = market_cap.gvkey.astype(str) + '_' + market_cap.iid.astype(str)
        market_cap['market_cap'] = market_cap.shares_out * market_cap.price_close
        market_cap = market_cap[['date','gvkey_iid','market_cap']]
        market_cap = market_cap.sort_values(by=['gvkey_iid','date']) # sorting df by gvkey_iid and then by date.
        
        market_cap.to_csv(os.path.join(save_path, r'market_cap.csv'))
        
        logging.info('loaded shares outstanding and daily_prices, calculated market cap, formatted and sorted')
        return market_cap
    
    def _get_frame_for_merge(self, composition_df):
        frame = pd.DataFrame({'date':[], 'gvkey_iid':[]})
        for r in range(0,len(composition_df)):
            frame0 = pd.DataFrame({'date':[], 'gvkey_iid':[]})
            frame0.gvkey_iid = composition_df.iloc[r]
            frame0.date = composition_df.index[r]
            frame = frame.append(frame0, ignore_index=True)
        logging.info('Created chronological frame of dates & gvkey_iids for mergers' )
        return frame
     
    def _get_market_cap_matrix_all(self):
        panel = pd.merge(self.frame, self.market_cap, on=['date','gvkey_iid'])
        temp = panel.groupby('date').first()
        days = temp.index
        market_cap_matrix_all = pd.DataFrame()
        iterations = 0         
        for d in days:
            panel0 = panel.loc[panel['date'] == d]
            panel0 = panel0.reset_index()
            market_cap = panel0.market_cap
            market_cap_matrix_all = market_cap_matrix_all.append(market_cap, ignore_index=True)   
            print(". ", end="") # printing '.' for every iteration and 'Row: i' every 50
            iterations += 1
            if iterations % 50 == 0:
                print('Frequency same as basket. Trading periods (Market cap): {}'.format(iterations))
                
        market_cap_matrix_all.to_csv(os.path.join(save_path, r'mc.csv'))
                
        return market_cap_matrix_all.fillna(0).values[1:,0:500]  # dropping first row(date) since no returns are calculated for it



# import composition
# import time
# 
# def main(index_name, start, end, lookback=252): # 1 year => 252 trading days
#     c = composition.Composition(index_name)
#        
#     days = utilities.TradingDays(start, end).get_trading_days()
#     basket ={}
#     for d in days:
#         logging.info('getting composition for %s' % d)
#         basket[d] = c.get_composition(d)
#     basket = pd.DataFrame.from_dict(basket, orient='index')    
#     m = MarketCap(basket, start, end)
#     market_cap_matrix = m.market_cap_matrix_all
#     return market_cap_matrix
#     
# if __name__ == '__main__':
#     start_time = time.time()
#     market_cap_matrix = main('sp500', '2016-01-01', '2018-12-31', lookback = 252)  
#     print(market_cap_matrix.shape)
#     print(market_cap_matrix)
#     print(type(market_cap_matrix))
#     print("--- %s seconds ---" % (time.time() - start_time))
    
    
    
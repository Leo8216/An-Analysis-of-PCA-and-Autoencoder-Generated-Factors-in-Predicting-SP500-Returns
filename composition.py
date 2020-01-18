'''
Created on Jun 3, 2019

@author: leo82
'''

import logging
logging.basicConfig(level=logging.INFO, format= '%(asctime)s - MAIN - [%(name)s] [%(levelname)s] : %(message)s')
import utilities
import pandas as pd
import copy

# import os.path
save_path = r'C:\Users\leo82\eclipse-workspace\capstone\results\test'

   
class Composition(object):
    '''
    Gets composition for a given index and date range.
    '''
    def __init__(self, index_name, end_date='2019-06-01'):
        self.index_name = index_name
        self.raw_info = self._get_raw_info()
        self.start = min(list(self.raw_info.start_date))
        self.end = end_date
        self.days = utilities.TradingDays(self.start, self.end).daily
        self.dates_to_add = self._get_important_dates_dict_to_add()
        self.dates_to_remove = self._get_important_dates_dict_to_remove()
        self.important_dates = self._get_important_dates()
        self.compositions = self._get_compositions() # [date] = [{}, {}, ...]
    
    def _get_raw_info(self):
        '''
        Loads data as dataframe and creates a column for gvkey_iid
        '''
        conn = utilities.sqlite_connect()
#         conn = utilities.psql_connect()
        sql_stmt = "select symbol, gvkey, iid, start_date, end_date from composition where index_name='%s'" % self.index_name
        comp = pd.read_sql_query(sql_stmt, conn)
        comp.start_date = pd.to_datetime(comp.start_date, format='%Y-%m-%d')  
        comp.end_date = pd.to_datetime(comp.end_date, format='%Y-%m-%d')
        comp['gvkey_iid'] = comp.gvkey.astype(str) + '_' + comp.iid.astype(str)
        logging.info('loaded as dataframe sql query %s' % sql_stmt)
        return comp
           
    def _get_important_dates_dict_to_add(self):
        '''
        returns a dictionary with start_dates as keys and gvkey_iids as values
        d = {start_date: [list of gvkey_iids]}
        '''
        important_dates_add = sorted(list(set(self.raw_info.start_date)))
        to_add = {}
        for d in important_dates_add:
            ta = self.raw_info.loc[self.raw_info.start_date == d]
            to_add[d] = list(ta.gvkey_iid)
#         to_add0 = pd.DataFrame.from_dict(to_add, orient='index')    
#         to_add0.to_csv(os.path.join(save_path, r'to_add.csv'))
        return to_add    
    
    def _get_important_dates_dict_to_remove(self):
        '''
        returns a dictionary with next_trading_days as keys and gvkey_iids as values
        d = {next_trading_day: [list of gvkey_iids]}
        '''
        important_dates_remove = sorted(list(set(self.raw_info.end_date)))
        to_remove = {}
        for d in important_dates_remove:
            if not(pd.isnull(d)):
                tr = self.raw_info.loc[self.raw_info.end_date == d]
                if d in self.days:
                    next_trading_day = self.days[self.days.index(d)+1]
                else:
                    next_trading_day = min(filter(lambda x: x > d, self.days))
                if next_trading_day not in to_remove.keys():
                    to_remove[next_trading_day] = list(tr.gvkey_iid)
                else:
                    to_remove[next_trading_day].append(tr.gvkey_iid.values) 
#         df = pd.DataFrame(self.days, columns=["colummn"])
#         df.to_csv(os.path.join(save_path, r'days.csv'), index=False)
#         to_remove0 = pd.DataFrame.from_dict(to_remove, orient='index')        
#         to_remove0.to_csv(os.path.join(save_path, r'to_remove.csv'))
        return to_remove
    
    def _get_important_dates(self):
        '''
        gets dates where there was a change in the index
        '''
        important_dates = list(self.dates_to_add.keys())
        for d in list(self.dates_to_remove.keys()):
            if d not in important_dates:
                important_dates.append(d)
        return sorted(important_dates)   
    
    def _get_composition(self, d):
#         d = pd.to_datetime(d, format='%Y-%m-%d')
        compo = pd.DataFrame({'gvkey_iid':[]})
        for i in range(0,len(self.raw_info)):
            if ((d >= self.raw_info.iloc[i,3]) # iloc[i,3] => start_date column
                and (d <= self.raw_info.iloc[i,4] or pd.isnull(self.raw_info.iloc[i,4]))): # iloc[i,4] => end_date column
                comp0 = pd.DataFrame({'gvkey_iid':[0]})
                comp0.gvkey_iid = str(self.raw_info.iloc[i,5]) # iloc[i,5] => gvkey_iid column
                compo = compo.append(comp0, ignore_index=True)
        compo_gvkey_iid_l = list(compo.gvkey_iid)       
        return compo_gvkey_iid_l
    
    def _get_compositions(self):
        gc = self._get_composition(self.important_dates[0])
        answer = {}
        answer[self.important_dates[0]] = copy.deepcopy(gc)
        for d in self.important_dates[1:]:
            logging.info('updating composition for %s' % d)
            if d in list(self.dates_to_remove.keys()):
                for i in self.dates_to_remove[d]:
                    gc.remove(i)
                    logging.info('removing %s' % i)
            if d in list(self.dates_to_add.keys()):
                for i in self.dates_to_add[d]:
                    gc.append(i)
                    logging.info('adding %s' % i)
            answer[d] = copy.deepcopy(gc)
        return answer
    
    def get_composition(self, date):
        if date in self.compositions:
            logging.info('date %s is in self.compositions' % date)
            return self.compositions[date]            
        else:
            most_recent = max(list(filter(lambda d: d < date, self.compositions.keys())))
            logging.info('Most recent date for %s is %s' %(date, most_recent))
            return self.compositions[most_recent]
         
                

import composition
import os.path
import time
                
def main(index_name, start, end, lookback = 504): # 1 year => 252 trading days
    c = composition.Composition(index_name)
    
    days = utilities.TradingDays(start, end).get_trading_days()
    basket ={}
    for d in days:
        logging.info('getting composition for %s' % d)
        basket[d] = c.get_composition(d)
    basket = pd.DataFrame.from_dict(basket, orient='index')  
    return basket
    
if __name__=='__main__':
    start_time = time.time()
    basket = main('sp500', '1984-01-01', '2018-12-31', lookback = 252)
    basket.to_csv(os.path.join(save_path, r'basket_all.csv'), index='False')
    print("--- %s seconds ---" % (time.time() - start_time))  
    
    

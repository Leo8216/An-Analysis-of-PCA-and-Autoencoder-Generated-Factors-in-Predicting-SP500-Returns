'''
Created on Jul 21, 2019

@author: leo82
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class PrincipalComponentAnalysis(object):
    '''
    allows for linear regression using principal components
    '''

    def __init__(self, returns_matrix, n_components = 0.95):
        self.returns_matrix = returns_matrix
        self.pca = PCA(n_components = n_components) # when n_component between 0 and 1, it returns number of principal components needed to explain percentage of variance.
        self.principal_components = self._get_pca_components()
        self.pca_linear_regresion_results = self._predict_returns_linear_regr_pca()
        self.predict_components_linear_1shift = self._predict_components_linear_1shift()
        
    def _get_pca_components(self):
        principal_components = self.pca.fit_transform(self.returns_matrix)
        return principal_components
    
    def plot_pca_cummulative_explained_variance(self):
        plt.grid(True)
        plt.title('PCAs Cumulative Explained Variance')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Explained Variance')
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        return plt.show()
    
    def _predict_returns_linear_regr_pca(self):
        train_x, test_x, train_y, test_y = train_test_split(self.principal_components, self.returns_matrix, test_size=1/5, random_state=0)
        linear_regr = LinearRegression()
        linear_regr.fit(train_x, train_y)
        predicted_returns = linear_regr.predict(test_x)
        results = {}
        results['predicted_returns'] = predicted_returns
        results['R2'] = linear_regr.score(test_x, test_y) # Returns the coefficient of determination R^2 of the prediction.
        results['coefficients'] = linear_regr.coef_
        results['intercept'] = linear_regr.intercept_
        results['mean_error'] = sum(abs(predicted_returns-test_y))/len(test_y)
        results['same_direction'] = sum(np.where(predicted_returns*test_y > 0, 1, 0))/len(test_y)
        same_direction = np.where(predicted_returns*test_y > 0, 1, 0)
        results['same_direction_percent_day'] = 100*same_direction.sum(axis=1)/same_direction.shape[1]
        return results
    
    def _predict_components_linear_1shift(self):
        x = np.delete(self.principal_components, -1, axis = 0) # deleting last day, lost when shifting to target next day
        y = np.delete(self.principal_components, 0, axis = 0) # deleting first day, to line up each day (x) with the following day
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=1/5, random_state=0)
        linear_regr = LinearRegression()
        linear_regr.fit(train_x, train_y)
        predicted_components = linear_regr.predict(test_x)
        results = {}
        results['predicted_components'] = predicted_components
        results['R2'] = linear_regr.score(test_x, test_y) # Returns the coefficient of determination R^2 of the prediction.
        results['coefficients'] = linear_regr.coef_
        results['intercept'] = linear_regr.intercept_
        results['mean_error'] = sum(abs(predicted_components-test_y))/len(test_y)
        return results 



import composition
import time
import utilities
import returns
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format= '%(asctime)s - MAIN - [%(name)s] [%(levelname)s] : %(message)s')


def main(index_name, start, end, lookback = 504): # 1 year => 252 trading days
    c = composition.Composition(index_name)
    
    days = utilities.TradingDays(start, end).get_trading_days()
    basket ={}
    for d in days:
        logging.info('getting composition for %s' % d)
        basket[d] = c.get_composition(d)
    basket = pd.DataFrame.from_dict(basket, orient='index')  
    
    r = returns.Returns(basket, start, end)
    returns_matrix_all = r.get_returns_matrix_all(frequency='daily') # match frequency to that of trading days  
    p = PrincipalComponentAnalysis(returns_matrix_all, n_components = .95)   
    
    return p.plot_pca_cummulative_explained_variance()
    
if __name__=='__main__':
    start_time = time.time()
    pca_plot = main('sp500', '2010-01-01', '2018-12-31', lookback = 252)
    print(pca_plot)
    print("--- %s seconds ---" % (time.time() - start_time))  
    
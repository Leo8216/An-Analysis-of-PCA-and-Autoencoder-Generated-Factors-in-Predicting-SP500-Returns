'''
Created on Aug 11, 2019

@author: leo82
'''

from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np


class Autoencoder(object):
    '''
    creates autoencoder based on number of layers and number of encoded nodes
    '''
    def __init__(self, returns_matrix, n_factors=100):
        self.returns_matrix = returns_matrix
        self.encoding_dimension = n_factors
        self.encoded_images = self.autoencoder()[0]
        self.decoded_images = self.autoencoder()[1]
        self.decoder = self.autoencoder()[2]
        self.auto_linear_regression_results = self._predict_returns_linear_regr_auto()
        
    def autoencoder(self):
        returns_100 = self.returns_matrix*100 # helps optimization - larger loss function
        x_train = returns_100
        x_test = returns_100   
        
        input_img = Input(shape=(500,))
        encoded = Dense(360, activation='tanh')(input_img)
        encoded = Dense(220, activation='tanh')(encoded)
        encoded = Dense(self.encoding_dimension, activation='tanh')(encoded)  #Middle layer
        decoded = Dense(220, activation='tanh')(encoded)
        decoded = Dense(360, activation='tanh')(decoded)
        decoded = Dense(500, activation='linear')(decoded)
        
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
        
        autoencoder.fit(x_train, x_train,
                        epochs=100, # 100
                        batch_size=64, # 64
                        shuffle=False,
                        validation_data=(x_test, x_test))
        
        # Needed only for generating output of encoder and decoder
        encoder = Model(input_img, encoded)
        encoded_input = Input(shape=(self.encoding_dimension,))
        decoded = autoencoder.layers[-3](encoded_input)
        decoded = autoencoder.layers[-2](decoded)
        decoded = autoencoder.layers[-1](decoded)
        decoder = Model(encoded_input, decoded)
        
        encoded_images = encoder.predict(x_train)
        decoded_images = decoder.predict(encoded_images)
        
        return encoded_images, decoded_images, decoder  
    
    def plot_encoded_decoded(self, factor = 0):
        plt.grid(True)
        plt.title('Autoencoders Factors')
        plt.xlabel('Number of days')
        plt.ylabel('Factor value')
        plt.plot(self.encoded_images[:,factor], color = 'r', label = 'Encoded')
        plt.plot(self.decoded_images[:,factor], color = 'b', label = 'Decoded')
        plt.legend(loc='best')
        return plt.show()        
    
    def _predict_returns_linear_regr_auto(self):
        # Return_Matrix must be multiplied by 100 (Autoencoder input was multiplied by 100 during training)
        train_x, test_x, train_y, test_y = train_test_split(self.encoded_images, self.returns_matrix*100, test_size=1/5, random_state=0)
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
    a = Autoencoder(returns_matrix = returns_matrix_all, n_factors=100) 
    
    return a.encoded_images, a.decoder
    
if __name__=='__main__':
    start_time = time.time()
    encoded_images, decoder = main('sp500', '2016-01-01', '2018-12-31', lookback = 252)
    print(encoded_images.shape)
    print(encoded_images)
    test0 = np.array([encoded_images[0,:]])
    print(test0.shape)
    print(test0)
    print(type(test0))
    test0_decoded = decoder.predict(test0)
    print(test0_decoded.shape)
    print(test0_decoded)
    print("--- %s seconds ---" % (time.time() - start_time))  
        
        
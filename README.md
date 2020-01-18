# An Analysis of PCA and Autoencoder Generated Factors in Predicting S&P Returns

### *By: Leo Barbosa. Santa Clara University, School of Business, Santa Clara, CA 95053, USA; lbarbosa@scu.edu*

__Abstract:__ The S&P500 is difficult to predict.  Multi-factor models provide a useful framework for making returns predictions and for controlling portfolio risk.  This paper explores a three-step process in predicting PCA and Autoencoders factors to generate multi-factor models from the S&P500 component securities.  The first is to decompose the variance of individual S&P stocks by using PCA or Autoencoders.  Step two is to forecast the next trading period factors using autoregressive techniques (AR, ARMA).  Finally, individual stock returns are predicted from the factor forecasts.  This approach does not consistently outperform a coin toss, making it unusable for trading purposes. Regardless, it sets up a framework to try different techniques that may lead to better results.

==============================================================================

## 1.	Introduction

Multifactor models have been a staple of quantitative finance.  Asset Managers use these factors to generate alpha and to manage risk.  These models explain up to 95% (Investopedia, n.d.) of individual stock returns.  Quantitative Asset Managers spend considerable effort developing proprietary factors to generate alpha and to better manage risk.  Managers are generally able to reduce volatility of returns through these factors. Extracting a meaningful alpha from such models can prove challenging even for leading asset managers.

The field of artificial intelligence is evolving rapidly and there’s an opportunity to apply novel techniques to generating alpha and managing risk.  This project compared the performance of deep learning (autoencoders) generated factors to a more traditional technique (principal component analysis - PCA) generated factors in predicting returns of individual stock components of the indexes. This is done using historical returns for the S&P500 index from January 1984 to December 2018 in the form of daily, weekly and monthly returns.

The results show that none of the techniques used significantly outperforms a simple guess (50% split) on which stocks will go up or down in the next trading cycle, with overall results (daily, weekly and monthly) having a mean of 49.95% and a standard deviation of 1.36%. Results for some trading periods may outperform, others underperform, however the average over time appears random. 

## 2.	Approach

The approach to generate returns from factors is outlined in flowchart below:

IMAGE HERE

This was done using a rolling window with size equal to lookback (# of trading periods). Meaning a returns matrix, of dimensions (lookback, # of components of index), was used to generate factors (requiring PCA and Autoencoder factors) which then were used to predict the factors for the next trading period. 

Results were calculated based on correct prediction of direction the constituent would go during the next trading period. 

## 3.	Index basket

The index basket was generated using data from Compustat – Capital IQ’s North America Daily Index Constituents database (Compustat - Capital IQ, n.d.). The count for each day was checked, with most days having 500 constituents or greater, and 14.7% of the trading days (between 1984-01-01 and 2018 12 31) had less than 500 constituents:

* From 1984-01-01 to 1989-11-30, Compustat composition data is available on monthly intervals, with updates at end of month. If a constituent was in the index for most of the month, but was not by end of the month, the end date (for being in index) reported was the last day of the previous month, leading to some days not having some constituent(s).
* After 1990 and up to 2013, the difference can be explained by corporate actions such as mergers, acquisitions, or de-listings.
* Many trading days, from 2013 on, consistently have more than 500 constituents. It is unclear why Compustat data reports more than 500 constituents for these days. The total number of constituents in basket never exceeded 505.

## 4.	Returns matrix

Returns were calculated for the given period (daily, weekly or monthly) with respect to the previous closing price of the constituents. For daily, returns were calculated with respect to previous trading day. For weekly, returns were calculated with respect to the last trading day of the week, in most cases being Friday. For monthly, returns were calculated with respect to the last trading day of the month. Closing prices data was retrieved from Compustat – Capital IQ’s North America Daily Security Daily database (Compustat - Capital IQ, n.d.).

The returns matrix was capped at 500 constituents. For those days were constituents were less than 500, a zero value was given to the ‘blank’ constituents. While not ideal, a zero value is neutral (no gain or loss) and the number of stocks with N/A was relatively small (less than 5, or 1%). This was necessary, as the techniques used (PCA, Autoencoder, AR, ARIMA) would not run with N/A values. 

For days that exceeded 500 constituents, the last constituents were dropped - the last values of the index basket consist of those constituents that have more recently been added to the index. These number of constituents exceeding 500 was never greater than 5. While this adds a bit of uncertainty relative to S&P500, results still serve as a good reference for comparison to the index.

With the returns for the respective trading period, a return matrix was generated organized in the same way the Index basket was organized. This returns matrix was of dimensions (lookback, # of components of index). The lookback was defined as the number of trading periods (days, weeks, or months) that the matrix should include. A lookback of 252 trading periods was used. A sample matrix is shown in Appendix A “Sample Returns Matrix - Dow 30, 2018 returns, weekly trading period”.

## 5.	Factors matrix

With returns matrix for all trading days and a lookback period (# of days to include), a rolling window for a subset of the risk matrix was taken, with dimension (lookback, # of components of index), starting from the first trading period. Factor analysis was then performed on the subset risk matrix using PCA or Autoencoders for each prediction. The generated factors matrix was of dimensions (lookback, # of factors).

__PCA:__ 100 components were used. 100 components were used so that >70% of the variance was retained (~75% retained with 100 components). The original intent was to use as many components needed to explain 95% of the variance, but this would have resulted in 300+ components for the index – refer to Figure 1 below. Such large number of components defeats the purpose of PCA, in addition to being computationally expensive. We, therefore, decided to explore what kind of results we would get if using only 100 components. 

FIGURE 1 HERE

To have a frame of reference, a regression was run predicting individual stock returns from 100 PCAs that were generated from known returns, to understand how well a linear regression could do if all PCAs are predicted accurately. This regression was on daily returns, from 2010-01-01 to 2018-12-31, 80% training, 20% test (equivalent to 453 trading days). This regression had an R2 of 0.5392, a mean of 73.61% (same direction per trading day) and a standard deviation of 10.52%. Therefore, if PCA are predicted perfectly, a 74% of constituents predicted in the correct direction are within the realm of possibilities.

__Autoencoders:__ A fully connected neural network was used, with ‘adadelta’ optimizer (a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients) and ‘mean_squared_error’ as the loss function, running 100 epochs for training. The middle layer number of nodes equals the number of generated factors. Autoencoder structure below. Refer to Figure 2 below for illustration on how returns were feed to Autoencoder:

* Input layer: 500 nodes
*	Encoder hidden layer 1: Dense(360, activation='tanh')
*	Encoder hidden layer 2: Dense(220, activation='tanh')
*	Encoder hidden layer 3: Dense(100, activation='tanh') [middle layer]
*	Decoder hidden layer 4: Dense(220, activation='tanh')
*	Decoder hidden layer 5: Dense(360, activation='tanh')
*	Output layer: Dense(500, activation='linear')

FIGURE 2 HERE

To have a frame of reference, a similar approach to PCA was taken, using factors generated from the encoder layer instead of the principal components (PCA generated). This regression had an R2 of 0.2593, a mean of 72.01% (same direction per trading day) and a standard deviation of 10.89%. Therefore, if Autoencoder factors are predicted perfectly, a 72% of constituents predicted in the correct direction are within the realm of possibilities.

For the autoencoder factors, values were extracted from the middle layer (hidden layer 3). Both, PCA and Autoencoder were run so that the same number of factors was generated, 100.

## 6.	Predicting factors

With the factor matrix (containing factors for each trading period in window), a prediction of the next trading period (day, week, or month) was made using autoregressive techniques, AR and ARMA:

__Autoregressive Model – AR:__ statsmodels.tsa.ar_model.AR was used. The autoregressive model was fit using Akaike Information Criterion (AIC) for selecting the optimal lag length. The lag length which results in the lowest AIC was selected. 

__Autoregressive Model with Moving Average – ARMA:__ pmdarima.arima.auto_arima was used.  Pmdarima is a statistical Python library with objective to bring R's auto.arima functionality to Python. The auto_arima function was fit using AIC. Inputs were:

*	max_p = 20.  From inspection of max lags from AR model, max lags selected was 17.
*	d = 0.  Factors were found to be stationary, so differencing was unnecessary.
*	max_q = 1. Factors seemed random from one day to the next. 
*	Seasonal = False.  Factors had no seasonality
*	m = 1. The number of observations per seasonal cycle is 1 since there is no seasonality.
*	Stepwise = True

AR and ARMA were run for each of the factors in the factors matrix, predicting 100 factors for the next trading period.

## 7.	Predicting returns

To predict returns, a linear regression was fit using the PCA and Autoencoder generated factors to predict the returns for each of the components. The fitted linear regression was then fed the predicted 100 factors to predict the next trading period returns of each of the constituents of the index. For reference, Table 1 below presents the balance of the out of sample returns for the daily, weekly and monthly trading periods.

TABLE 1 HERE

## 8.	Results

The predicted returns were then compared to the actual returns. Results were calculated based on correct prediction of direction the index constituent would go during the next trading period. The results were calculated for overall, by trading day (# of stocks predicted in same direction) and by index constituent (# of days constituent was predicted in same direction). The overall results are listed in tables below.

TABLE 2, 3 AND 4 HERE

In addition, market cap weighted results were calculated for each trading period.  For weekly and monthly trading periods, results show that at times it is possible to predict significantly above 50% of the market cap of the index.  These results, in addition to being inconsistent, do not allow us to say that the index will go up or down, given that the % of rightly predicted market cap may have gone up or down by a magnitude greater or less than those we are unable to predict. Results for ~60 periods for daily, weekly and monthly trading periods are displayed in Appendix B.

## 9.	Conclusion

This paper presents the results of a different approach to predicting stock market returns by utilizing PCAs and Autoencoders to generate factors, and then predicting factors and returns for next trading period (daily, weekly, or monthly). The overall results do not significantly outperform a random guess (50%) of whether the market will go up or down. When looking at results for individual trading periods, some do show significant outperformance, however this happens inconsistently in an unpredictable manner. The market weighted results, while interesting for weekly and monthly trading periods, are inconsistent and unreliable for trading.   It is clear that the factors generated with PCA and Autoencoder during this work do not outperform those currently used in industry such as the Fama French factors (French, 1993). Use of other techniques for predicting the next day factors, and predicting the returns from the forecasted factors may yield better results.

# References

* Compustat - Capital IQ. (n.d.). North America Daily / Index Constituents. Retrieved June 16, 2019, from Wharton Research Data Services: "WRDS" wrds.wharton.upenn.edu
* Compustat - Capital IQ. (n.d.). North America Daily / Security Daily. Retrieved June 16, 2019, from Wharton Research Data Services: "WRDS" wrds.wharton.upenn.edu
* French, E. F. (1993). Common risk factors in the returns on stocks and bonds. Journal of Financial Economics 33, 3-56.
* Investopedia. (n.d.). Fama and French Three Factor Model. Retrieved 11 10, 2019, from Investopedia: https://www.investopedia.com/terms/f/famaandfrenchthreefactormodel.asp

# Appendix A – Sample Returns Matrix - Dow 30, 2018 returns, weekly trading period

DOW 30 2018 RETURNS HERE

# Appendix B – Results for ~60 periods for daily, weekly and monthly trading periods

Trading period: Daily, start_date: 10/1/2017, end_date: 12/31/2018

TRADING DAY HERE

Trading period: Daily, start_date: 10/1/2017, end_date: 12/31/2018

STOCK HERE

Trading period: Daily, start_date: 10/1/2017, end_date: 12/31/2018

MARKET CAP HERE

Trading period: Weekly, start_date: 12/1/2012, end_date: 12/31/2018

TRADING DAY HERE

Trading period: Weekly, start_date: 12/1/2012, end_date: 12/31/2018

STOCK HERE

Trading period: Weekly, start_date: 12/1/2012, end_date: 12/31/2018

MARKET CAP HERE

Trading period: Monthly, start_date: 11/1/1992, end_date: 12/31/2018

TRADING DAY HERE

Trading period: Monthly, start_date: 11/1/1992, end_date: 12/31/2018

STOCK HERE

Trading period: Monthly, start_date: 11/1/1992, end_date: 12/31/2018

MARKET CAP HERE







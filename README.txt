 Time Series Analysis

For the purpose of time series analysis, monthly personal expenditure data of US population/economy was used. The end goal was to apply different time- series models (Average, Naïve, Drift, Simple Exponential Smoothing, Holt- Linear, Holt-Winter, Linear Regression and Auto Regressive Moving Average (ARMA)) on the data set and then choose the one that best forecasts the future values. It turned out that ARMA(1,0) predicted values that most closely resembled the actual data.

Source of Data:
https://fred.stlouisfed.org/

Table of Contents:
1. Abstract
2. Introduction
3. DescriptionofDataset
- Time Series plots
- ACF plot
- Correlation Matrix
- Checking for missing values
- Splitting data into train & test
4. CheckforStationarity
- Visual check
- ADF test
- First Order Differencing
5. TimeSeriesDecomposition - Additive
- Multiplicative
6. Holt-Winter&otherforecastingMethods
7. FeatureSelection&LinearRegressionModel
8. ARMAModel
- Auto-ARIMA
9. PickingthebestModel
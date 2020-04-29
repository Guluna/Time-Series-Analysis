import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import statsmodels.formula.api as sm_ols        # for ols
import statsmodels.api as sm   # for ARMA
from sklearn.model_selection import train_test_split
from Toolbox import ADF_test, correlation_coefficient_cal, autocorrelation_cal, plot_ACF, visual_stationary_check, autocorrelation_cal_k_lags
from Toolbox import holt_linear, holt_winter_seasonal, avg_method_hstep, ses_method_hstep, drift_method_hstep, naive_method_hstep
from Toolbox import phi_kk, calc_ARMA, autocorrelation_cal_k_lags, plot_ACF_title, RMSE_calc
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import chi2
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore")


## *********************************************************************************************************
# RAW DATA
## *********************************************************************************************************

# Dependent variable
personal_consump_exp = pd.read_csv('./Data/PCE.csv')
# Independent variables
unempl_rate = pd.read_csv('./Data/UNRATE.csv')
personal_income = pd.read_csv('./Data/PI.csv')



# plotting features and target variables against time
plt.subplot(2,1,1)
plt.plot(personal_income.DATE, personal_income.PI, label='Personal Income')
plt.plot(personal_consump_exp.PCE, label='Personal Expenditure')
plt.title('Personal Income & Expenditure (USA)')
plt.xlabel('Time')
plt.xticks(personal_income.DATE[::48], rotation=45)
plt.ylabel('Billions of USD')
plt.legend()

plt.subplot(2,1,2)
plt.plot(unempl_rate.DATE, unempl_rate.UNRATE, label='Unemployment rate')
plt.title('Unemployment rate (USA)')
plt.xlabel('Time')
plt.xticks(unempl_rate.DATE[::48], rotation=45)
plt.ylabel('Percentage')
plt.legend()
plt.show()

## *********************************************************************************************************
# MERGED DATA
## *********************************************************************************************************

# From plots above, we observe that starting and ending period for all columns is not same so need to fix that.
merged_inner = pd.merge(left=personal_income, right=personal_consump_exp, left_on='DATE', right_on='DATE')
merged_inner = pd.merge(left=merged_inner, right=unempl_rate, left_on='DATE', right_on='DATE')      # all values start from 1959-01-01 and end on 2019-12-01

# Since we have monthly instead of daily data so creating a new column that contains only the month and year of date
merged_inner['Date_Month'] = pd.to_datetime(merged_inner['DATE']).dt.to_period('M')
merged_inner.head()

# Now that we have a new Date_Month column, removing old DATE column
del merged_inner['DATE']
merged_inner.head()

# converting df to time series object by setting index col = Date_Month for easier manipulation e.g. in plots
# Period data type must 1st be converted into string or number type before setting it as index col for df
merged_inner['Date_Month'] = merged_inner['Date_Month'].astype(str)
merged_inner.set_index('Date_Month', inplace=True)

# creating a copy of combined dataframe so that original data remains safe
df = merged_inner.copy()
PCE = df.PCE



# Plotting the same plot as above but with similar starting and ending dates & formatted values on x-axis for time
plt.subplot(2,1,1)
plt.plot(df.PI, label='Personal Income')
plt.plot(df.PCE, label='Personal Expenditure')
plt.title('Personal Income & Expenditure (USA)')
plt.xlabel('Time')
plt.xticks(df.index[::48], rotation=45)
plt.ylabel('Billions of USD')
plt.legend()

plt.subplot(2,1,2)
plt.plot(df.UNRATE, label='Unemployment rate')
plt.title('Unemployment Rate (USA)')
plt.xlabel('Time')
plt.xticks(df.index[::48], rotation=45)
plt.ylabel('Percentage')
plt.legend()

plt.tight_layout()
plt.show()



# ----------------------------------------------------------------------
# Make sure that your dataset has at least 500 samples or more.
# ----------------------------------------------------------------------

print('Number of rows and columns in dataframe are:', df.shape, '\n')
N = len(df)     # total obs in df

# 1st 5 rows
df.head()
# last 5 rows
df.tail()



# ----------------------------------------------------------------------
# You must make sure that the dataset does not misses any samples. And the dataset is clean.
# ----------------------------------------------------------------------
null_rows = df[df.isna().any(axis=1)]
print('Rows containing missing (NaN) values are:', null_rows)       # Nil

df.info()

# ----------------------------------------------------------------------
# Make sure the time steps are equal.
# ----------------------------------------------------------------------

print('The first date in our data set is:', df.index[0])
print('The last date in our data set is:', df.index[-1])
print('So we have monthly data for ', 2019 - 1959 + 1, 'years. \nConsequently, 61*12 = ', 61*12, 'months which is exactly '
'equal to \nthe number of rows we have in our dataframe.')


# ----------------------------------------------------------------------
# 8 (f)- Plot the ACF of the original dataset and see if the ACF decays.
# ----------------------------------------------------------------------
# all lags i.e. 732
acf = autocorrelation_cal(df.PCE)
a = plot_ACF(acf)

# ----------------------------------------------------------------------
# Correlation Matrix with seaborn heatmap and pearson’s correlation coefficient.
# ----------------------------------------------------------------------

sns.heatmap(df.corr(), vmin=-1, cmap='coolwarm', annot=True)
plt.show()

print('From heatmap, it is observed that there is a strong correlation between Personal Income & Personal Expenditure '
'whereas these two variable have hardly any correlation with Unemployment. Hence, Unemployment should be dropped from dataset'
'because apparently it does not hold any predictive power.')


# ----------------------------------------------------------------------
# SPLITTING DATA SET INTO TRAIN & TEST SET
# ----------------------------------------------------------------------
# dividing whole dataset into train & test set

train, test = train_test_split(df, test_size=0.2, shuffle=False)

# ----------------------------------------------------------------------
# CHECKING STATIONARITY OF DEPENDENT VARIABLE
# ----------------------------------------------------------------------


# visually checking if mean & var are constant
visual_stationary_check(df.PCE, df.index)

# objective test to confirm sty
adf_result = ADF_test(df.PCE)   # non-sty with p-value of 1


# ----------------------------------------------------------------------
# APPLYING FIRST ORDER DIFFERENCING TO TACKLE NON-STATIONARITY
# ----------------------------------------------------------------------

# to detrend, we are going to apply first order differencing technique
a = np.array((df.PCE)).copy()
PCE_differenced = a[1:] - a[0: len(a)-1]

# creating a separate df for differenced/Stationary data, it starts from Feb 1959 (instead of Jan 1959)
sty_df = pd.DataFrame(PCE_differenced, index=df.index[1:])
sty_df.rename(columns={0:'PCE'}, inplace=True)


# visually checking if mean & var are constant
visual_stationary_check(sty_df.PCE, sty_df.index)

# objective test to confirm sty
adf_result = ADF_test(sty_df.PCE)   # sty with p-value of 0.02


# # # log transformation
# log_df = np.log(a)
# adf_result = ADF_test(log_df)       # 0.07 hence failed ADF test
#
# # 1st order diff of log transformation
# PCE_log_diff = log_df[1:] - log_df[0: len(log_df)-1]
# ADF_test(PCE_log_diff)

# ----------------------------------------------------------------------
# REVERSE TRANSFORMATION OF DETRENDED DATA
# ----------------------------------------------------------------------

# reverse_diff = np.array(df.PCE.iloc[0: N-1]) + np.array(sty_df.PCE.iloc[0: ])   # 731 obs starting with Feb 1959
# ----------------------------------------------------------------------
# TIME SERIES DECOMPOSITION (original )
# ----------------------------------------------------------------------

result = seasonal_decompose(df.PCE, model='additive', freq=12)  # data collected every month hence freq=12
result.plot()
plt.suptitle('Additive Decomposition', fontsize=16 )
plt.xlabel('Time (monthly)')
plt.show()

result = seasonal_decompose(df.PCE, model='multiplicative', freq=12)
result.plot()
plt.suptitle('Multiplicative Decomposition', fontsize=16)
plt.xlabel('Time (monthly)')
plt.show()


print('************* Results for simple & advanced forecasting methods start here **********')

# # ----------------------------------------------------------------------
# 6- Holt-Winters method: Using the Holt-Winters method try to find the best fit using the train dataset and
# make a prediction using the test set.
# # ----------------------------------------------------------------------

l = len(test.PCE)


avg_forecast = avg_method_hstep(train.PCE)
# print('Forecast for Average Method', avg_forecast)

naive_forecast = naive_method_hstep(train.PCE)
# print('Forecast for Naive Method', naive_forecast)

drift_forecast = drift_method_hstep(train.PCE, l)
# print('Forecast for Drift Method', drift_forecast)

ses_forecast = ses_method_hstep(train.PCE)
# print('Forecast for SES Method', ses_forecast[-1])
# print(ses_forecast)



holt_linear_forecast = holt_linear(train.PCE, test.PCE)

holt_seasonal_forecast = holt_winter_seasonal(train.PCE, test.PCE)

# creating prediction plot
empty_list = [None for i in train.PCE]
plt.plot(df.PCE, label='True values')
plt.plot(empty_list + [avg_forecast]*l, label='Average')
plt.plot(empty_list + [naive_forecast]*l, label='Naive')
plt.plot(empty_list + drift_forecast, label='Drift')
plt.plot(empty_list + [ses_forecast[-1]]*l, label='SES')
plt.plot(empty_list + list(holt_linear_forecast), label='Holt Linear')
plt.plot(empty_list + list(holt_seasonal_forecast), label='Holt Winter Seasonal')
plt.legend()
plt.xlabel('Time')
plt.xticks(df.index[::48], rotation=45)
plt.ylabel('Values obtained')
plt.title('True vs Predicted values for Personal Expenditure Data')
plt.show()


# calculating residuals
residuals_avg   = test.PCE - [avg_forecast]*l
residuals_naive = test.PCE - [naive_forecast]*l
residuals_drift = test.PCE - drift_forecast
residuals_ses = test.PCE - [ses_forecast[-1]]*l
residuals_holt_linear = test.PCE - list(holt_linear_forecast)
residuals_holt_winter = test.PCE - list(holt_seasonal_forecast)

# ACF of Residuals
acf_residuals_avg = autocorrelation_cal_k_lags(residuals_avg, 20)
plot_ACF_title(acf_residuals_avg, 'Residuals (Average Method)')

acf_residuals_naive = autocorrelation_cal_k_lags(residuals_naive, 20)
plot_ACF_title(acf_residuals_naive, 'Residuals (Naive Method)')

acf_residuals_drift = autocorrelation_cal_k_lags(residuals_drift, 20)
plot_ACF_title(acf_residuals_drift, 'Residuals (Drift Method)')

acf_residuals_ses = autocorrelation_cal_k_lags(residuals_ses, 20)
plot_ACF_title(acf_residuals_ses, 'Residuals (SES Method)')

acf_residuals_holt_linear = autocorrelation_cal_k_lags(residuals_holt_linear, 20)
plot_ACF_title(acf_residuals_holt_linear, 'Residuals (Holt-Linear Method)')

acf_residuals_holt_winter = autocorrelation_cal_k_lags(residuals_holt_winter, 20)
plot_ACF_title(acf_residuals_holt_winter, 'Residuals (Holt-Winter Method)')


# RMSE of residuals
RMSE_avg = RMSE_calc(residuals_avg)
RMSE_naive = RMSE_calc(residuals_naive)
RMSE_drift = RMSE_calc(residuals_drift)
RMSE_ses = RMSE_calc(residuals_ses)
RMSE_holt_linear = RMSE_calc(residuals_holt_linear)     # has least RMSE= 703
RMSE_holt_winter = RMSE_calc(residuals_holt_winter)

print('Root Mean Square Error of Average Method is: ', RMSE_avg)
print('Root Mean Square Error of Naive Method is: ', RMSE_naive)
print('Root Mean Square Error of Drift Method is: ', RMSE_drift)
print('Root Mean Square Error of SES Method is: ', RMSE_ses)
print('Root Mean Square Error of Holt-Linear Method is: ', RMSE_holt_linear)
print('Root Mean Square Error of Holt-Winter Method is: ', RMSE_holt_winter)

# mean & variance of residuals for Holt-Linear method
print('Mean of residuals for Holt-Linear method is : ', np.mean(residuals_holt_linear))     # -628 biased
print('Variance of residuals for Holt-Linear method is : ', np.var(residuals_holt_linear))




# # ----------------------------------------------------------------------
# 8- Develop the multiple linear regression model that represent the dataset. Check the accuracy of the developed model.
# a. You need to include the complete regression analysis into your report.
# b. Hypothesis tests like F-test, t-test
# c. AIC, BIC, RMSE, R-squared and Adjusted R-squared
# d. ACF of residuals.
# e. Q-value
# f. Variance and mean of the residuals.
# # ----------------------------------------------------------------------


ols_model = sm_ols.ols("PCE ~ PI + UNRATE", data=train).fit()
ols_model_summary = ols_model.summary()
print(ols_model_summary)

predictions_ols = ols_model.predict(test)
# print(predictions_ols)
prediction_error_ols = test['PCE'] - predictions_ols

# creating prediction plot
empty_list = [None]*len(train)
plt.plot(df.PCE, label='True values')
plt.plot(empty_list + list(predictions_ols), label='Multiple Linear Regression')
plt.legend()
plt.xlabel('Time')
plt.xticks(df.index[::48], rotation=45)
plt.ylabel('Values obtained')
plt.title('True vs Predicted values (Linear Regression)')
plt.show()

# acf of Residuals
acf_residuals_ols = autocorrelation_cal_k_lags(prediction_error_ols, 20)
plot_ACF_title(acf_residuals_ols, 'Residuals (Multiple Linear Regression Method)')

mean_residuals_ols = np.mean(prediction_error_ols)
var_residuals_ols = np.var(prediction_error_ols)
RMSE_residuals_ols = np.sqrt(np.mean(prediction_error_ols**2))

# Q-value for OLS
T = len(test)
h = 20
Q_ols = T * np.sum(acf_residuals_ols[1:h]**2)

print('Mean, Variance, Q-value & RMSE of prediction error (OLS model) are {:.2f}, {:.2f}, {:.2f} & {:.2f} respectively.'.format(mean_residuals_ols, var_residuals_ols, Q_ols, RMSE_residuals_ols))

## RERUN OLS WITH Y-INTERCEPT REMOVED

rerun_ols_model = sm_ols.ols("PCE ~ 0 + PI + UNRATE", data=train).fit()
rerun_model_summary = rerun_ols_model.summary()
print(rerun_model_summary)

rerun_predictions_ols = rerun_ols_model.predict(test)
# print(predictions_ols)
rerun_prediction_error_ols = test['PCE'] - rerun_predictions_ols

# creating prediction plot
empty_list = [None]*len(train)
plt.plot(df.PCE, label='True values')
plt.plot(empty_list + list(rerun_predictions_ols), label='Multiple Linear Regression')
plt.legend()
plt.xlabel('Time')
plt.xticks(df.index[::48], rotation=45)
plt.ylabel('Values obtained')
plt.title('Predictions for LR with Y-intercept removed')
plt.show()

# acf of Residuals
rerun_acf_residuals_ols = autocorrelation_cal_k_lags(rerun_prediction_error_ols, 20)
plot_ACF_title(rerun_acf_residuals_ols, 'Residuals (LR Method Y-intercept removed)')

rerun_mean_residuals_ols = np.mean(rerun_prediction_error_ols)
rerun_var_residuals_ols = np.var(rerun_prediction_error_ols)
rerun_RMSE_residuals_ols = np.sqrt(np.mean(rerun_prediction_error_ols**2))

# Q-value for OLS
T = len(test)
h = 20
Q_ols_rerun = T * np.sum(rerun_acf_residuals_ols[1:h]**2)

print('Mean, Variance, Q-value & RMSE of prediction error (OLS model with y-intercept removed) are {:.2f}, {:.2f}, {:.2f} & {:.2f} respectively.'.format(mean_residuals_ols, var_residuals_ols, Q_ols, RMSE_residuals_ols))

# ----------------------------------------------------------------------
# ARMA MODEL
# ----------------------------------------------------------------------

# using 1st order differenced (Stationary PCE) values for ARMA model
# calc ACF for GPAC
acf_sty_df = autocorrelation_cal_k_lags(sty_df.PCE, 20)      # lag 20
plot_ACF_title(acf_sty_df, 'Differenced PCE')


# creating a GPAC with j=8 & k=8
phi = []
for nb in range(8):
    for na in range(1,9):
        phi.append(phi_kk(nb, na,  acf_sty_df))


gpac = np.array(phi).reshape(8,8)
gpac_df = pd.DataFrame(gpac)
cols = np.arange(1,9)
gpac_df.columns = cols
print(gpac_df)
print()

sns.heatmap(gpac_df, annot=True)
plt.xlabel('AR process (k)')
plt.ylabel('MA process (j)')
plt.title('Heatmap of GPAC (ARMA process)')
plt.show()

# ***********************************************************************************************
# estimate the corresponding parameters for the AR and MA part.
# ***********************************************************************************************
def ARMA_estimates(na, nb):

    train_arma, test_arma = train_test_split(sty_df.PCE, test_size=0.2, shuffle=False)

    # ARMA parameter estimatation
    arma_model = sm.tsa.ARMA(train_arma, (na, nb)).fit(trend='nc', disp=0)

    for i in range(na):
        print('The AR coefficient a{}'.format(i), 'is:', arma_model.params[i])
    for i in range(nb):
        print('The MA coefficient b{}'.format(i), 'is:', arma_model.params[i+na])
    print()

    print(arma_model.summary())

    print('Confidence Interval (95%) for estimated paramters are:\n' , arma_model.conf_int(alpha=0.05))
    print()

    # ***********************************************************************************************
    # 5- Display the estimated covariance matrix.
    # ***********************************************************************************************

    print('The estimated covariance matrix for for model is: \n' , arma_model.cov_params())
    print()

    # ***********************************************************************************************
    # 6- Display the estimated variance of the prediction error.
    # ***********************************************************************************************

    print('The estimated variance & standard deviation of prediction error :\n' , arma_model.sigma2, np.sqrt(arma_model.sigma2))
    print()

    # ***********************************************************************************************
    # 8- Plot the true Sales value versus the estimated Sales value.
    # ***********************************************************************************************

    # Prediction
    # model_hat = arma_model.predict(start=test_arma.index[0], end=len(train_arma)+len(test_arma)-1)
    model_hat = arma_model.predict(start=test_arma.index[0], end=test_arma.index[-1])


    # ADF_test(sty_df.PCE)

    # creating prediction plot
    empty_list = [None]*len(train_arma)
    plt.plot(sty_df.PCE, label= 'Original (differenced) values')
    plt.plot(empty_list + list(model_hat), label='ARMA Predictions')
    plt.legend()
    plt.xlabel('Time')
    plt.xticks(sty_df.index[::48], rotation=45)
    plt.ylabel('Values obtained')
    plt.title('True vs Predicted values (ARMA({},{}) Model)'.format(na,nb))
    plt.show()


    # ***********************************************************************************************
    # 9- Plot the ACF of the residuals.
    # ***********************************************************************************************

    residuals_arma = pd.DataFrame(  model_hat - sty_df.PCE[len(train_arma):])
    a = np.array(residuals_arma[0])
    # print(a)
    a = np.delete(a, -1)

    print('Mean of ARMA residuals', a.mean())
    print('Variance of ARMA residuals', a.var())
    print('RMSE of ARMA residuals', np.sqrt(np.mean(a**2)))



    acf_residuals_arma = autocorrelation_cal_k_lags(a, 20)      #  lag 20
    title = 'Residuals (Differenced) ARMA({},{})'.format(na,nb)
    # N = len(test_arma)
    plot_ACF_title(acf_residuals_arma, title)

    # ***********************************************************************************************
    # 10 - Find Q value.
    # ***********************************************************************************************
    N = len(test_arma)
    Q_arma = N * (np.sum(acf_residuals_arma[1:]**2))

    # ***********************************************************************************************
    # 11 - Are the residuals errors white? Knowing the DOF and alfa = .01 perform a   test and check if the residuals pass the whiteness test.

    # Q must be less than Qc
    # ***********************************************************************************************

    DOF = 20 - na - nb
    # define probability
    alpha = 0.01

    # retrieve value <= probability
    Q_critical = chi2.ppf(1-alpha, DOF)

    if Q_arma < Q_critical:
        print('Q ({}) < Q_c ({}) hence residuals pass the whiteness test confirming that suggested model is good.'.format(Q_arma, Q_critical))
    else:
        print('Residuals fail Chi-Square test.')

    # Reverse transforming the ARMA predictions
    reverse_diff_arma_predictions = np.array(df.PCE.iloc[584: 731]) + np.array(model_hat[0: ])   # 731 obs starting with Feb 1959

    # creating prediction plot
    empty_list = [None]*len(train)
    plt.plot(df.PCE, label='True values')
    plt.plot(empty_list + list(reverse_diff_arma_predictions), label='ARMA')
    plt.legend()
    plt.xlabel('Time')
    plt.xticks(df.index[::48], rotation=45)
    plt.ylabel('Values obtained')
    plt.title('True vs Predicted values ARMA({},{}) (After reverse Transform)'.format(na,nb))
    plt.show()

    return residuals_arma, reverse_diff_arma_predictions

print('*********  Results for ARMA(1,0) start here *************')

residuals_arma, reverse_diff_arma_predictions = ARMA_estimates(1,0)
# reverse_diff_arma_predictions = ARMA_estimates(1,0)
# print(reverse_diff_arma_predictions)
residuals_arma_reverse_transformed = df.PCE[584:731] - reverse_diff_arma_predictions

# calc ACF for ARMA residuals after reverse transforming data
acf_sty_df = autocorrelation_cal_k_lags(residuals_arma_reverse_transformed, 20)      # lag 20
plot_ACF_title(acf_sty_df, 'ARMA (Reverse Transformed) residuals')



print('*********  Results for ARMA(0,1) start here *************')
ARMA_estimates(0,1)
print()

print('*********  Results for ARMA(0,2) start here *************')
residuals_arma = ARMA_estimates(0,2)
print()

# ARMA(2,2) RAISES ERROR
# print('*********  Results for ARMA(2,2) start here *************')
# residuals_arma = ARMA_estimates(2,2)
# print()

# PASSING ARMA(1,0)'S RESIDUAL to GPAC
def create_gpac(ts):
    acf_sty_df = autocorrelation_cal_k_lags(ts, 30)      # lag 30
    # plot_ACF_title(acf_sty_df, 'Residuals GPAC')


    # creating a GPAC with j=8 & k=8
    phi = []
    for nb in range(15):
        for na in range(1,16):
            phi.append(phi_kk(nb, na,  acf_sty_df))


    gpac = np.array(phi).reshape(15,15)
    gpac_df = pd.DataFrame(gpac)
    cols = np.arange(1,16)
    gpac_df.columns = cols
    print('GPAC of residuals ARMA(1,0)')
    print(gpac_df)
    print()

    return gpac_df

gpac_arma_residuals = create_gpac(np.array(residuals_arma[0]))

print('*********  Results for ARMA(10,0) start here *************')
ARMA_estimates(10,0)
print('*********  Results for ARMA(11,0) start here *************')
ARMA_estimates(11,0)
print('*********  Results for ARMA(16,0) start here *************')
ARMA_estimates(16,0)


### ARIMA MODEL
print('****************** Auto ARIMA Model results start here *************')
import pmdarima as pm

# fitting arima model
arima_model = pm.auto_arima(train.PCE, test='adf')
print(arima_model.summary())
arima_model.plot_diagnostics(figsize=(8,8))
plt.show()

# predicting
arima_model_hat = arima_model.predict(test.shape[0])  # predict N steps into the future


# creating prediction plot
empty_list = [None]*len(train)
plt.plot(df.PCE, label= 'Original values')
plt.plot(empty_list + list(arima_model_hat), label='ARIMA(2,1,2) Predictions')
plt.legend()
plt.xlabel('Time')
plt.xticks(sty_df.index[::48], rotation=45)
plt.ylabel('Values obtained')
plt.title('True vs Predicted values (ARIMA(2,1,2) Model)')
plt.show()


# ***********************************************************************************************
# 9- Plot the ACF of the residuals.
# ***********************************************************************************************

residuals_arima = pd.DataFrame( arima_model_hat - test.PCE)
a = np.array(residuals_arima.PCE)
# print(a)
# a = np.delete(a, -1)

# print(len(residuals_arima))
# print(len(test))

print('Mean of ARIMA residuals', a.mean())
print('Variance of ARIMA residuals', a.var())
print('RMSE of ARIMA residuals', np.sqrt(np.mean(a**2)))


acf_residuals_arima = autocorrelation_cal_k_lags(a, 20)      #  lag 20
title = 'Residuals ARIMA(2,1,2)'
# N = len(test_arma)
plot_ACF_title(acf_residuals_arima, title)

# ***********************************************************************************************
# 10 - Find Q value.
# ***********************************************************************************************
N = len(test)
Q_arima = N * (np.sum(acf_residuals_arima[1:]**2))

# ***********************************************************************************************
# 11 - Are the residuals errors white? Knowing the DOF and alfa = .01 perform a   test and check if the residuals pass the whiteness test.

# Q must be less than Qc
# ***********************************************************************************************

DOF = 20 - 2 - 2
# define probability
alpha = 0.01

# retrieve value <= probability
Q_critical = chi2.ppf(1-alpha, DOF)

if Q_arima < Q_critical:
    print('Q ({}) < Q_c ({}) hence residuals pass the whiteness test confirming that suggested model is good.'.format(Q_arma, Q_critical))
else:
    print('ARIMA Residuals fail Chi-Square test.')


# print('****************** Auto ARIMA Model with y-intercept removed *************')
# arima_model = pm.auto_arima(train.PCE, test='adf', with_intercept=False)
# print(arima_model.summary())
# arima_model.plot_diagnostics(figsize=(8,8))
# plt.show()

#
# plotting just holt-linear, LR & ARMA(0,1), ARIMA(2,1,2) predictions for final comparison
# creating prediction plot
empty_list = [None for i in train.PCE]
plt.plot(df.PCE, label='True values')
plt.plot(empty_list + list(holt_linear_forecast), color='blue', label='Holt Linear')
plt.plot(empty_list + list(rerun_predictions_ols), color='green', label='Linear Regression (Intercept removed)')
plt.plot(empty_list + list(reverse_diff_arma_predictions), color='pink', label='ARMA(1,0)')
plt.plot(empty_list + list(arima_model_hat), color='yellow', label='ARIMA(2,1,2)')
plt.legend()
plt.xlabel('Time')
plt.xticks(df.index[::48], rotation=45)
plt.ylabel('Values obtained')
plt.title('True vs Predicted values for Personal Expenditure Data')
plt.show()
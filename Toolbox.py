import warnings
warnings.filterwarnings("ignore", category=FutureWarning)   # always place it before the line that produces warning

from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.holtwinters as ets
# import statsmodels.tsa.holtwinters. as ets_results
from scipy import signal


def ADF_test(df_col):
    result = adfuller(df_col)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def correlation_coefficient_cal(x,y):
    x_avg = sum(x) / len(x)
    y_avg = sum(y) / len(y)
    diff_x = x - x_avg
    diff_y = y - y_avg
    numerator = sum(diff_x * diff_y)
    denominator = round(sqrt(sum(diff_x ** 2)) * sqrt(sum(diff_y ** 2)), 2)
    r = numerator / denominator
    return r


def standard_error(T, k, error):        # T is # of errors, k is # of predictors, e is list of errors
    num = np.sum(error**2)
    denom = T-k-1
    std_error = sqrt(num/denom)
    return std_error


def R_squared(predictions, test_y):
    y_bar = np.mean(test_y)
    num = np.sum((predictions - y_bar)**2)
    denom = np.sum((test_y - y_bar)**2)
    return num/denom


def adj_R_squared(T, k, R_2):
    a = 1-R_2**2
    b = (T-1)/(T-k-1)
    adj_R = a/b
    return 1 - adj_R


def autocorrelation_cal(ts):
    avg = ts.mean()
    deviation_from_mean = ts - avg
    denominator = np.sum(deviation_from_mean**2)

    T = len(ts)
    acf_all_lags = []

    for k in range(0, T):       # k is # of lags
        t = k+1
        numerator = 0
        for i in range(t, T+1):
            numerator += (ts[i-1]-avg)*(ts[i-1-k]-avg)       # -1 is added in original equation bc python indexing starts at 0 instead of 1
        acf = numerator/denominator
        acf_all_lags.append(acf)
        # if (k == 100):
        #     break


    return pd.Series(acf_all_lags)


def autocorrelation_cal_k_lags(ts, lags):
    avg = ts.mean()
    deviation_from_mean = ts - avg
    denominator = np.sum(deviation_from_mean**2)

    T = len(ts)
    acf_all_lags = []

    for k in range(0, T):       # k is # of lags
        t = k+1
        numerator = 0
        for i in range(t, T+1):
            numerator += (ts[i-1]-avg)*(ts[i-1-k]-avg)       # -1 is added in original equation bc python indexing starts at 0 instead of 1
        acf = numerator/denominator
        acf_all_lags.append(acf)
        if (k == lags):
            break


    return pd.Series(acf_all_lags)


def plot_ACF(ts):                   # .reset_index(drop=True)

    plt.stem(ts, use_line_collection=True)
    plt.stem(np.arange(0, -len(ts), -1), ts, use_line_collection=True)  # for symmetry
    plt.title('ACF Plot for Personal Expenditure')
    plt.xlabel('Time Lags')
    plt.ylabel('ACF values')
    plt.show()



def Q_value(acf_values):
    q = len(acf_values) * np.sum(acf_values[1:]**2)     # k i.e. lag starts from 1
    return q


def visual_stationary_check(ts, index):
    '''Plot the mean and variance over time by incrementing samples.'''

    mean = pd.Series(ts).rolling(window=5).mean()
    variance = pd.Series(ts).rolling(window=5).var()

    plt.plot(ts, '-r', label='Personal Expenditure')
    plt.plot(mean, '-b', label='Mean')
    plt.plot(variance, '-g', label='Variance')
    plt.xticks(index[::48], rotation=45)
    plt.title('Plotting Mean & Variance (Visual Sty check)')
    plt.xlabel('Time')
    plt.ylabel('Personal Expenditure in $B')
    plt.legend()
    plt.show()


########   SIMPLE FORECASTING METHODS #########

def avg_method_hstep(train):
    y_hat_avg = train.mean()
    return y_hat_avg


def naive_method_hstep(train):
    return train.iloc[-1]


def drift_method_hstep(train, l):
    y_hat_drift = []
    def cal(train):
            y_t = train[len(train)-1]
            m = (y_t - train[0]) / len(train)
            h = 1
            y_hat_drift.append((y_t + m * h))
            return y_hat_drift

    for i in range(l):
        if i == 0:
            y_hat_drift = cal(list(train))
        else:
            y_hat_drift = cal(list(train)+y_hat_drift)
    return y_hat_drift


def ses_method_hstep(train):
    alpha = 0.8
    l0 = 0
    result = [alpha*train[0] + (1-alpha)*l0]
    for i in range(1, len(train)):
        result.append(alpha * train[i] + (1 - alpha) * result[i - 1])
    return result


def holt_linear(train, l):
    model = ets.ExponentialSmoothing(train, trend='mul', damped=True)     # Use the multiplicative version, unless the data has been logged before.

    fit1 = model.fit()
    print('Holt Linear method summary\n',fit1.summary())
    return fit1.predict(start=l.index[0], end=l.index[-1])


def holt_winter_seasonal(train, l):
    model = ets.ExponentialSmoothing(train, trend='mul', seasonal='mul')     # Use the multiplicative version, unless the data has been logged before.
    fit1 = model.fit()
    # print('Holt Winter method summary\n',fit1.summary())
    # return fit1.forecast(l)
    return fit1.predict(start=l.index[0], end=l.index[-1])



def phi_kk(j,k,acf):

    den = np.zeros((k,k))   # denominator matrix for phi_kk
    for row in range(k):
        for col in range(k):
            den[row, col] = acf[abs(j+row-col)]

    num = den.copy()
    num[:, -1] = acf[j+1:j+k+1]     # replacing last column of num for cramer's rule

    phi = round(np.linalg.det(num)/np.linalg.det(den), 3)

    return phi



def calc_ARMA(samples, coeff_MA, coeff_AR):
    np.random.seed(42)
    mean = 0; std = 1
    e = std * (np.random.randn(samples)+mean)
    system = (coeff_MA, coeff_AR, 1)
    y_dlsim = signal.dlsim(system, e)
    y = y_dlsim[1].flatten()        # # dlsim fn returns a tuple (the [1] element of which is our desired ARMA process but it is in 2d so flatten)

    return y


def plot_ACF_title(lag_20, title):
    plt.stem(lag_20, use_line_collection=True)
    plt.stem(np.arange(0, -len(lag_20), -1), lag_20, use_line_collection=True)  # for symmetry
    plt.title('ACF Plot of ' + title)
    plt.xlabel('Time Lags')
    plt.ylabel('ACF values')

    plt.show()


def RMSE_calc(residuals):
    return np.sqrt(np.mean(residuals**2))
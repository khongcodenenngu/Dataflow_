import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import pywt


def clean(df):
    """
    Clean the dataframe from the raw csv file
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df["Price"] = df["Price"].str.replace(",", "").astype(float)
    df["Open"] = df["Open"].str.replace(",", "").astype(float)
    df["High"] = df["High"].str.replace(",", "").astype(float)
    df["Low"] = df["Low"].str.replace(",", "").astype(float)
    df["Vol."] = df["Vol."].replace({'T': 'e12', 'B': 'e9', 'M': 'e6', 'K': 'e3'}, regex=True).astype(float)
    df["Change %"] = df["Change %"].str.replace('%', '', regex=True).astype(float)

    return df


def reverse_inplace(df):
    df.iloc[:] = df.iloc[::-1].reset_index(drop=True)
    return df


def convert_day_to_month(df, date_col="Date"):
    """
    Convert the df recorded daily to df recorded monthly
    The rows extracted is the last trading day of that month and must be after 13th of that month
    """
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    df = df.sort_values(by=date_col)
    
    # Group by month and get the last entry for each month
    monthly_df = df[df[date_col].dt.to_period('M') != df[date_col].shift(-1).dt.to_period('M')]
    
    # Filter for days after the 13th
    monthly_df = monthly_df[monthly_df[date_col].dt.day > 13]

    monthly_df = monthly_df.rename(columns={date_col: "Month"})
    monthly_df["Month"] = monthly_df["Month"].dt.strftime('%Y-%m')

    return monthly_df.reset_index(drop=True)


def cal_rsi(df, column="Price", window=14):
    """
    Relative Strength Index
    """
    delta = df[column].diff(1)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gains).rolling(window=window).mean()
    avg_loss = pd.Series(losses).rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rs[avg_loss == 0] = np.inf
    
    rsi = 100 - (100 / (1 + rs))

    return rsi
    

def cal_stochastic_oscillator(df, price_col="Price", high_col="High", low_col="Low", window=14):
    """
    Stochastic Oscillator
    """
    L = df[low_col].rolling(window=window).min()
    H = df[high_col].rolling(window=window).max()
    C = df[price_col] # current closing price

    stochastic_k = 100 * ((C - L) / (H - L))

    return stochastic_k


def cal_willr(df, price_col="Price", high_col="High", low_col="Low", window=14):
    """
    Williams %R indicator
    """
    L = df[low_col].rolling(window=window).min()
    H = df[high_col].rolling(window=window).max()
    C = df[price_col] # current closing price

    willr = -100 * ((H - C) / (H - L))

    return willr


def cal_SMA(df, window, column="Price"):
    return df[column].rolling(window=window).mean()


def cal_EMA(df, window, column="Price"):
    return df[column].ewm(span=window, adjust=False).mean()


def cal_MACD(df, column='Price', short_window=12, long_window=26, signal_window=9):
    short_ema = df[column].ewm(span=short_window, adjust=False).mean()
    long_ema = df[column].ewm(span=long_window, adjust=False).mean()

    MACD_Line = short_ema - long_ema
    Signal_Line = MACD_Line.ewm(span=signal_window, adjust=False).mean()
    MACD_Histogram = MACD_Line - Signal_Line

    return MACD_Line, Signal_Line, MACD_Histogram


def cal_ROC(df, window, column="Price"):
    prev_price = df[column].shift(window)
    roc = ((df[column] - prev_price) / prev_price) * 100

    return roc


def cal_OBV(df, price_col="Price", volume_col="Vol."):
    obv = np.zeros(len(df))
    for i in range(1, len(df)):
        if df[price_col].iloc[i] > df[price_col].iloc[i - 1]:
            obv[i] = obv[i - 1] + df[volume_col].iloc[i]
        elif df[price_col].iloc[i] < df[price_col].iloc[i - 1]:
            obv[i] = obv[i - 1] - df[volume_col].iloc[i]
        else:
            obv[i] = obv[i - 1]

    return pd.Series(obv, index=df.index)


def cal_ATR(df, window, price_col="Price", high_col="High", low_col="Low"):
    high_low = df[high_col] - df[low_col]
    high_close = np.abs(df[high_col] - df[price_col].shift(1))
    low_close = np.abs(df[low_col] - df[price_col].shift(1))

    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=window).mean()

    return atr


def cal_momentum(df, window, column="Price"):
    momentum = df[column] - df[column].shift(window)
    
    return momentum


def cal_closeratio(df, window, column="Price"):
    rolling_avg = df[column].rolling(window=window).mean()
    
    return df[column] / rolling_avg
    

def cal_updays(df, window, column="Price"):
    updays = df[column].diff() > 0
    
    return updays.rolling(window=window).sum()


def cal_log_return(df, column="Price"):
    """
    Log of returns after 1 days
    """
    log_return = np.log(df[column] / df[column].shift(1))

    return log_return


def cal_volume_spike(df, window, volume_col="Vol."):
    avg_volume = df[volume_col].rolling(window=window).mean()

    return df[volume_col] / avg_volume


def adf_test(series):
    result = adfuller(series.dropna())  # ADF Test requires no NaN values
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] > 0.05:
        print("Data is non-stationary (p-value > 0.05).")
    else:
        print("Data is stationary (p-value < 0.05).")


def cal_monthly_return(df, month_col="Month", price_col="Price"):
    df = df.copy()
    df[month_col] = pd.to_datetime(df[month_col], format='%Y-%m')

    # Sort by month in case it's not ordered
    df = df.sort_values(by=month_col)
    monthly_returns = df[price_col].pct_change() * 100

    return monthly_returns


def wavelet_denoising(signal, wavelet="db4", level=1):
    """
    Perform wavelet denoising on a time series.
    
    Parameters:
    - signal: 1D numpy array (time series data)
    - wavelet: Wavelet type (default: 'db4' - Daubechies 4)
    - level: Decomposition level (default: 1)
    
    Returns:
    - Denoised signal as numpy array
    """
    # 1. Decompose signal using Discrete Wavelet Transform (DWT)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 2. Estimate noise level using Median Absolute Deviation (MAD)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Robust noise estimation
    
    # 3. Compute threshold based on the noise estimate
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # 4. Apply soft thresholding to remove noise
    coeffs[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
    
    # 5. Reconstruct the denoised signal
    return pywt.waverec(coeffs, wavelet)








"""
Potential indicator: 
- BB -> categorical
"""
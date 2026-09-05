"""Technical indicator calculations."""
import numpy as np
import pandas as pd


def calculate_technicals(df):
    delta = df['Close'].diff()

    # 1. Corrected RSI (Wilder's Smoothing)
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. MACD with Histogram (B3)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # 3. Bollinger Bands with %B and Bandwidth (B4)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
    bb_range = df['BB_Upper'] - df['BB_Lower']
    df['BB_PctB'] = (df['Close'] - df['BB_Lower']) / bb_range.replace(0, np.nan)
    df['BB_Width'] = bb_range / df['SMA_20'].replace(0, np.nan)

    # 4. Corrected ATR (Wilder's Smoothing)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.ewm(alpha=1/14, adjust=False).mean()

    # 5. StochRSI with %K/%D Smoothing (B2)
    min_rsi = df['RSI'].rolling(window=14).min()
    max_rsi = df['RSI'].rolling(window=14).max()
    # Guard divide-by-zero when RSI is flat for the full window
    rsi_range = (max_rsi - min_rsi).replace(0, np.nan)
    raw_stoch = (df['RSI'] - min_rsi) / rsi_range
    df['StochRSI_K'] = raw_stoch.rolling(3).mean()
    df['StochRSI_D'] = df['StochRSI_K'].rolling(3).mean()
    df['StochRSI'] = df['StochRSI_K']

    # 6. VWAP with Daily Reset (B1)
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    # Group by trading date for proper daily resets
    df['_trade_date'] = df.index.normalize()
    df['_tp_vol'] = df['TP'] * df['Volume']
    df['VWAP'] = df.groupby('_trade_date')['_tp_vol'].cumsum() / df.groupby('_trade_date')['Volume'].cumsum()
    df.drop(columns=['_trade_date', '_tp_vol'], inplace=True)

    # 7. OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_SMA'] = df['OBV'].rolling(20).mean()

    # 8. ADX with Directional Info (B5)
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)

    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    # Guard divide-by-zero in flat markets where +DI + -DI == 0
    di_sum = (df['+DI'] + df['-DI']).replace(0, np.nan)
    df['DX'] = 100 * np.abs(df['+DI'] - df['-DI']) / di_sum
    df['ADX'] = df['DX'].ewm(alpha=1/14, adjust=False).mean()

    # 9. Williams %R (D1)
    highest_14 = df['High'].rolling(14).max()
    lowest_14 = df['Low'].rolling(14).min()
    df['Williams_R'] = -100 * (highest_14 - df['Close']) / (highest_14 - lowest_14).replace(0, np.nan)

    # 10. CCI - Commodity Channel Index (D2)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI'] = (tp - tp_sma) / (0.015 * tp_mad)

    # Drop helper/intermediate columns so the cached DataFrame stays clean.
    # (Keep +DI/-DI/ATR because the UI layer reads them.)
    df.drop(columns=['UpMove', 'DownMove', '+DM', '-DM', 'DX', 'TP'],
            inplace=True, errors='ignore')

    return df

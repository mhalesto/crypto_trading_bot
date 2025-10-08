import os, time
from typing import Tuple
import numpy as np
import pandas as pd
import ccxt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from dotenv import load_dotenv
load_dotenv()

def _exchange():
    name = os.getenv("EXCHANGE", "binance").lower()
    ex = getattr(ccxt, name)()
    api, secret = os.getenv("API_KEY",""), os.getenv("API_SECRET","")
    if api and secret and hasattr(ex, 'apiKey'):
        ex.apiKey, ex.secret = api, secret
    ex.enableRateLimit = True
    return ex

def fetch_ohlcv(symbol:str, timeframe:str="1m", limit:int=1000) -> pd.DataFrame:
    ex = _exchange()
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rsi = RSIIndicator(close=df["close"], window=14)
    macd = MACD(close=df["close"])
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["rsi"] = rsi.rsi()
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["ret_1"] = df["close"].pct_change()
    df["atr"] = atr.average_true_range()
    df.dropna(inplace=True)
    return df

def make_supervised(df: pd.DataFrame, horizon:int=1, lookback:int=60) -> Tuple[np.ndarray,np.ndarray, pd.DataFrame]:
    df = df.copy()
    df["y"] = (df["close"].shift(-horizon) > df["close"]).astype(int)
    feat_cols = [
        "close",
        "volume",
        "rsi",
        "macd",
        "macd_signal",
        "bb_high",
        "bb_low",
        "ret_1",
        "atr",
    ]
    df.dropna(inplace=True)
    X_list, y_list = [], []
    values = df[feat_cols].values
    yvals = df["y"].values
    for i in range(lookback, len(df)-horizon):
        X_list.append(values[i-lookback:i])
        y_list.append(yvals[i])
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, df.iloc[lookback:len(df)-horizon]

def load_dataset(symbol=None, timeframe=None, limit=1500, horizon=None, lookback=None):
    symbol = symbol or os.getenv("SYMBOL","BTC/USDT")
    timeframe = timeframe or os.getenv("TIMEFRAME","1m")
    horizon = int(horizon or os.getenv("PREDICTION_HORIZON", "1"))
    lookback = int(lookback or os.getenv("LOOKBACK_WINDOW", "60"))
    df = fetch_ohlcv(symbol, timeframe, limit)
    df = add_indicators(df)
    X, y, aligned = make_supervised(df, horizon, lookback)
    return X, y, aligned

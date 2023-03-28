import pandas as pd
import numpy as np
import requests
import os

from sklego.preprocessing import RepeatingBasisFunction


def load(currency: str, interval: str) -> pd.DataFrame:
    """
    Loads stock price for currency pair
    args:
        currency : currency pair (e.g. EURUSD)
    returns : dataframe with OHLCV values for pari
    """
    file_path = os.path.join("data", interval, f"{currency}=X.csv")
    if os.path.exists(file_path):
        curr_df = pd.read_csv(
            file_path, parse_dates=["Datetime"], index_col=["Datetime"]
        ).astype(np.float32)
    return curr_df


def donwload_currency(currency: str, quote_currency: str, interval: str):
    """
    Download new data
    """
    response = requests.get(
        f"https://eodhistoricaldata.com/api/eod/{currency}{quote_currency}.FOREX?api_token={API_KEY}&order=d&fmt=csv"
    )
    response.raise_for_status()
    with open(
        f"Data/CurrencyConversionRates/{currency}_{quote_currency}.csv", "w"
    ) as f:
        f.write(response.text)


def _resample_to_interval(data: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Resamples dataframe to new interaval.
    args:
        data : dataframe with datetime index
        new_interval : interval code e.g. 1m/5m/15m/1h/4h/1d/...
    returns : dataframe with data for new index
    """
    return data.resample(interval, closed="right").agg(
        {
            "Open": "first",
            "Low": np.min,
            "High": np.max,
            "Close": "last",
            "Volume": np.sum,
        }
    )


def add_variables(ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Adds additional variables to OHLC data.

    args:
        ohlc : dataframe with OHLC values

    returns : pandas dataframe
    """
    ohlc = ohlc.loc[~ohlc['Close'].isna(), :]
    
    ohlc = ohlc.assign(
        # roi_1, roi_3, roi_5, roi_10, roi_15, roi_20
        roi_1=ohlc["Close"].pct_change(1),
        roi_3=ohlc["Close"].pct_change(3),
        roi_5=ohlc["Close"].pct_change(5),
        roi_10=ohlc["Close"].pct_change(10),
        roi_20=ohlc["Close"].pct_change(20),
        # candle body size
        candle_body_size=(ohlc["Close"] - ohlc["Open"]) / (ohlc["High"] - ohlc["Low"]),
        # vol_change_1, vol_change_2
        vol_change_1=ohlc["Volume"].pct_change(1),
        vol_change_2=ohlc["Volume"].pct_change(2),
        vol_change_3=ohlc["Volume"].pct_change(3),
        # SMA-50 and SMA-200
        sma_50=ohlc['Close'].rolling(50).mean(),
        sma_200=ohlc['Close'].rolling(200).mean(),
        # target_value: next close, target_direction: whether 12'th close from now is over/under current closing price
        target_value=ohlc["Close"].shift(-1),
        target_direction=(ohlc["Close"].pct_change().shift(-12) > 0).astype(int),
    )

    hour_of_day = _encode_hour_of_day(ohlc)
    day_of_week = _encode_day_of_week(ohlc)
    impulse = _is_impulse(ohlc)

    ohlc = pd.concat([ohlc, hour_of_day, day_of_week, impulse], axis=1)
    ohlc = ohlc.loc[:, ~ohlc.columns.duplicated()].copy()

    return ohlc


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleanes dataframe from invalid values.
    args:
        data : dataframe
    """
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.loc[~data.isna().any(axis=1), :]
    
    return data


"""
Functions for calculating additional variables
"""


def _encode_hour_of_day(ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes hour of the day using Repeating Basis Function from sklego package.
    more information under:
    https://scikit-lego.netlify.app/_modules/sklego/preprocessing/repeatingbasis.html#RepeatingBasisFunction

    args:
        ohlc : dataframe with OHLC values

    returns : pandas dataframe
    """
    ohlc = ohlc.copy()
    ohlc["hour_of_day"] = ohlc.index.hour
    rbf = RepeatingBasisFunction(
        n_periods=24, column="hour_of_day", input_range=(1, 24), remainder="drop"
    )

    rbf.fit(ohlc)
    res = pd.DataFrame(index=ohlc.index, data=rbf.transform(ohlc)).rename(
        lambda x: f"hour_{x}", axis=1
    ).astype(np.float32)

    return res


def _encode_day_of_week(ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes day of the week using Repeating Basis Function from sklego package
    more information under:
    https://scikit-lego.netlify.app/_modules/sklego/preprocessing/repeatingbasis.html#RepeatingBasisFunction

    args:
        ohlc : dataframe with OHLC values

    returns : pandas dataframe
    """
    ohlc = ohlc.copy()
    ohlc["week_of_day"] = ohlc.index.weekday
    rbf = RepeatingBasisFunction(
        n_periods=7, column="week_of_day", input_range=(1, 7), remainder="drop"
    )

    rbf.fit(ohlc)
    res = pd.DataFrame(index=ohlc.index, data=rbf.transform(ohlc)).rename(
        lambda x: f"weekday_{x}", axis=1
    ).astype(np.float32)

    return res


def _is_impulse(
    ohlc: pd.DataFrame, window: int = 50, quantile: float = 0.95
) -> pd.DataFrame:
    """
    Marks observation as impulse if its pips_range (Close - Open) is bigger than 98 percentile of
    trailing pips range over last 50 values.
    args:
        ohlc : dataframe with OHLC values
        window : window size for trailin part
        quantile : percentile from which pip range has to be bigger to be concidered as impulse

    returns : pandas dataframe with impulse column
    """
    ohlc["candle_size"] = ohlc["High"] - ohlc["Low"]
    pips_range_95_percentile = ohlc["candle_size"].abs().rolling(window).quantile(quantile)
    ohlc["impulse"] = (ohlc["candle_size"] > pips_range_95_percentile).astype(np.int0)

    return ohlc

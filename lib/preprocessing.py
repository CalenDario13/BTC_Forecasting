import investpy
import talib as tb
from ta import add_all_ta_features

import warnings

import pandas as pd
import numpy as np
import os

from pytrends import dailydata

import pickle

import logging.config
import yaml

from lib.path_retriever import get_path

def get_n_best_cryptos(n: int =10) -> np.array:

    """ 
		This function retrieves the n most traded crypto currencies name (based on trading volumes)
		
        Parameters
        ----------
        n: int, (default=10)
            the number of values to retrieve (for example if n=10 it retrieves the top 10 cryptos)

        Returns
        -------
        array_like
            It returns an array containing the names
    """
    
    CRYPTO_NAMES = os.path.join(get_path('configs'), 'crypto_names.pickle')
    
    if os.path.isfile(CRYPTO_NAMES):
        # Load the array with the names:
        with open(CRYPTO_NAMES, 'rb') as handle:
            crypto_names =  pickle.load(handle)
    else:
        # Retreive info:
        all_crypto = investpy.crypto.get_cryptos_overview()
        # Remove not available cryptos:
        all_crypto = all_crypto.loc[(all_crypto.name!='Binance USD') & (all_crypto.name!='BNB') & (all_crypto.name!='Bitcoin'), ]
        # Get top n best features based on volumes:
        all_crypto.total_volume = all_crypto.total_volume .apply( lambda x: np.float16(x[:-1]) )
        crypto_names = all_crypto.sort_values(by='total_volume', ascending=False).name.iloc[:10].values
        # Save names:
        with open(CRYPTO_NAMES, 'wb') as handle:
            pickle.dump(crypto_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return crypto_names
    
def create_crypto_df(crypto_names:np.array, start_period:str, end_period:str) -> pd.DataFrame:

    """
    It retrieves the daily prices of the given cryptos by the usage of investing API and creates a DataFrame
    
    Parameters
    ----------
        crypto_names: np.array
            An array containing the names of the crypto to retrieve
        start_period: str
            A string containing the starting date for the analysis
        end_period: str
            A string containing the ending date for the analysis
    Returns
    -------
    pd.DataFrame
        A dataframe contianing the daily closing prices of the given cryptos
    """

    crypto_df = pd.DataFrame()
    for name in crypto_names:
        try:
            df = investpy.get_crypto_historical_data(crypto=name, from_date=start_period, to_date=end_period)
            crypto_df = pd.concat([crypto_df, df.Close], axis=1)
        except Exception as e:
            logging.error(e)
            break
        
    crypto_df.columns = crypto_names
    return crypto_df
    
def compute_moving_average(target_df:pd.DataFrame, col:str, ma_type:str='SMA', periods:list=None)->pd.DataFrame:

    """
    This function compute the simple moving average (MA) or the Exponential Moving Average (EMA),
    for a given number of periods (or for the default ones) of the Closing Prices of a column of a dataset.

    Parameters
    ----------
    target_df: pd.DataFrame
      It is the original dataframe from which take the column to modify
    col: str
      the colimn of the traget_df to use to compute the MA
    ma_type: str, (default='SMA), {'SMA', 'EMA'}
      it indicate sthe kind of ma to compute (only SImpe and Exponential are allowed)
    periods: list, (default=None), optional
      It is possible to specify custom periods to compute the MA
    
    Notes
    ------
    If no period is specified, the default ones are [5, 10, 20, 50, 100, 200]

    Returns
    --------
    pd.DataFrame
      The target_df which includes the new MA columns
    """
    
    # prepare periods:
    if not periods:
        periods = [5, 10, 20, 50, 100, 200]

    # Get the column to modify:
    target_col = target_df[col]

    for p in periods:

        # Compute Column:
        if ma_type=='SMA':
            new_ma = tb.SMA(target_col, p)
            new_ma.name = 'trend_{}_{}'.format(ma_type.lower(), p)
        elif ma_type == 'EMA':
            new_ma = tb.SMA(target_col, p)
            new_ma.name = 'trend_{}_{}'.format(ma_type.lower(), p)
        else:
            e = "This MA type is not allowed, choose among ('SMA' or 'EMA')"
            logging.error(e)
            raise ValueError(e)

        # Add to DataFrame:
        target_df = pd.concat([target_df, new_ma], axis=1)

    return target_df
    
def generate_gtrend_df(kw_list:list, start_year:int, start_mon:int, stop_year:int, stop_mon:int, geo:str, verbose:bool, sleep:float)->pd.DataFrame:
    """
    This function prepare the dataset with the google trends daily information (quantity of queries for a specific word)

    Parameters
    ----------
    kw_list: list of str
     List of Words to fetch daily data for
    start_year: int
      The start year
    start_mon: int
      Start 1st day of the month
    stop_year: int
      The end year
    stop_mon: int
      End at the last day of the month
    geo: str 
      Geolocation where the terms were serched
    verbose: bool 
      If True, then prints the word and current time frame we are fecthing the data for
    """

    trends_df  = pd.DataFrame()
    for i, kwd in enumerate(kw_list):
        logging.info('GTrends is retrieving kwd {} of {}'.format(i+1, len(kw_list)))
        new_trend = dailydata.get_daily_data(kwd, start_year, start_mon, stop_year, stop_mon, geo, verbose, sleep).iloc[:,-1].dropna()
        trends_df = pd.concat([trends_df, new_trend], axis=1)
    
    # Adjust columns and index names:
    trends_df.columns = ['gtrend_{}'.format(col) for col in trends_df.columns]
    trends_df.index.name = 'datetime'
    
    return trends_df
    
def retrieve_gtrend_df(kw_list:list, start_year:int=2018, start_mon:int=12, stop_year:int=2021, stop_mon:int=12, geo:str = '', verbose:bool=False, sleep:float=1.0)->pd.DataFrame:
  
    """
    This function retrieve the google trends data frame from local and creates it if it does not exists.

    Parameters
    ----------
    kw_list: list of str
     List of Words to fetch daily data for
    start_year: int, (default=2018)
      The start year
    start_mon: int, (default=12) 
      Start 1st day of the month
    stop_year: int, (default=2021)
      The end year
    stop_mon: int, (default=12) 
      End at the last day of the month
    geo: str (defaut='') 
      Geolocation where the terms were serched
    verbose: bool (default=False) 
      If True, then prints the word and current time frame we are fecthing the data for
    """
    
    PATH_GTRENDS = os.path.join(get_path('data'), 'gtrends.csv')
    if os.path.isfile(PATH_GTRENDS):
        trends_df = pd.read_csv(PATH_GTRENDS)
        trends_df.datetime = pd.to_datetime(trends_df.datetime)
        trends_df.set_index(keys='datetime', inplace=True)
        if set(trends_df.columns) == set(['gtrend_{}'.format(col) for col in kw_list]):
            return trends_df
        else:
            trends_df = generate_gtrend_df(kw_list, start_year, start_mon, stop_year, stop_mon, geo, verbose, sleep)
            trends_df.reset_index(drop=False).to_csv(PATH_GTRENDS, index=False)
            trends_df.index = pd.to_datetime(trends_df.index)
            return trends_df
    else:
        trends_df = generate_gtrend_df(kw_list, start_year, start_mon, stop_year, stop_mon, geo, verbose, sleep)
        trends_df.reset_index(drop=False).to_csv(PATH_GTRENDS, index=False)
        trends_df.index = pd.to_datetime(trends_df.index)
        return trends_df


 
def retrieve_data(start_period: str, 
                   end_period: str,
                   top_n:int=10,
                   sma_periods:list=None,
                   ema_periods:list=None,
                   kw_list:list=['bitcoin', 'BTC', 'blockchain', 'crypto']
                   ):
 
    """ 
        This function prepare the data for the analysis and models
        
        Parameters
        ----------
        start_period: str
            A string containing the starting date for the analysis
        end_period: str
            A string containing the ending date for the analysis
        top_n: int, (default=10)
            the number of values to retrieve (for example if n=10 it retrieves the top 10 cryptos)
        sma_periods: list, (default=None), optional
            A list of periods to compute the echnical indicator for the SMA  on Closing Prices of BTC
        ema_periods: list, (default=None), optional
            A list of periods to compute the technical indicator for the EMA  on Closing Prices of BTC
        kw_list: list, (default=['bitcoin', 'BTC', 'blockchain', 'crypto'])
            A list of keywords to use in google trend (find how many times that word was looked for)
            
        Returns
        -------
        tpl of pd.DataFrame
            It returns datafarems containing the variables that I am going to use in the analysis
    """

    # Configure Log:
    with open(os.path.join(get_path('configs'), 'log_configs.yml'), 'rt') as f:
      config = yaml.safe_load(f.read())
      logging.config.dictConfig(config)
      
    # Get Bitcoin data:
    logging.info( "Starting retrieving data from period '{}' to '{}...'".format(start_period, end_period) )
    target_df = investpy.get_crypto_historical_data(crypto='bitcoin', from_date=start_period, to_date=end_period)
    # Add technical Indicators:
    logging.info( "Adding techincal Indicators For the BTC Prices..." )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        target_df = add_all_ta_features(target_df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    # Add Custom MA and EMA:
    target_df = compute_moving_average(target_df, col='Close', ma_type='SMA', periods=sma_periods)
    target_df = compute_moving_average(target_df, col='Close', ma_type='EMA', periods=ema_periods)
    
    # Drop Unuseful Columns:
    target_df.drop(columns=['Currency', 'trend_sma_fast','trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow'], inplace=True) 

    # Get the top n cryptos to include in the analys:
    crypto_names = get_n_best_cryptos(n=top_n)
    logging.info('Best Cryptos are: {}'.format(', '.join(crypto_names)))
    crypto_df = create_crypto_df(crypto_names, start_period, end_period)
    
    # Get goolge serachs for key-words:
    logging.info("Retrieving GTrends Data Frame")
    gtrends_df = retrieve_gtrend_df(kw_list, start_year=int(start_period[-4:])-1, start_mon=12, stop_year=int(end_period[-4:]), stop_mon=int(end_period[3:5]) )
    gtrends_df = gtrends_df.loc[ gtrends_df.index.year >= int(start_period[-4:]) ] # Must be adjusted to select correctly the initial month (in this case, it is aslways supposed to be January)
    
    logging.info("Retrivial Proess Succesfully End!")
    
    return target_df, crypto_df, gtrends_df
    


    



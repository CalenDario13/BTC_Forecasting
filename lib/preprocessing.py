import investpy
import talib as tb
from ta import add_all_ta_features

import warnings

import pandas as pd
import numpy as np
import os

import pickle

import logging.config
import yaml

from lib.path_retriever import get_path

def get_n_best_cryptos(n: int =10):

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
    
def compute_moving_average(target_df:pd.DataFrame, col:str, ma_type:str='SMA', periods:list=None):

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

    
 
def retrieve_data(start_period: str, end_period: str, top_n:int=10, sma_periods:list=None, ema_periods:list=None):
 
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
    
    crypto_df = pd.DataFrame()
    for name in crypto_names:
        try:
            df = investpy.get_crypto_historical_data(crypto=name, from_date=start_period, to_date=end_period)
            crypto_df = pd.concat([crypto_df, df.Close], axis=1)
        except Exception as e:
            logging.error(e)
            break
        
    crypto_df.columns = crypto_names
    
    logging.info("Retrivial Proess Succesfully End!")
    
    return target_df, crypto_df
    
    
    



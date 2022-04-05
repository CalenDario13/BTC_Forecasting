import investpy

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
    
 
def retrieve_data(start_period: str, end_period: str, top_n: int =10):
 
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
    logging.info( "Starting retrieving data from period '{}' to '{}'".format(start_period, end_period) )
    target_df = investpy.get_crypto_historical_data(crypto='bitcoin', from_date=start_period, to_date=end_period)

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
    
    return target_df, crypto_df
    
    
    



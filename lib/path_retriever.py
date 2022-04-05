import os
import json

def get_path(name):
    """ 
    
		This function retrievs a desired path from a json file containing all useful paths
		
        Parameters
        ----------
        name: str, {'root', 'main', 'configs', 'log'}
            the name corresponding to the key of the path to retriev

        Returns
        -------
        str 
            It returns the required path as string
    """
    
    FOLDER = os.path.join(os.getcwd(), 'runner/configs')
    with open(os.path.join(FOLDER, 'paths.json'), 'r') as jsonFile:
        path_dict = json.load(jsonFile)
        assert name in path_dict.keys(), "The required path '{}' does not exists, available paths: ({}).".\
            format( name, ', '.join( path_dict.keys() ) )
        return path_dict[name]
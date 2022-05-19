__author__ = "Konrad"
__copyright__ = "Copyright 2022, Konrad"
__maintainer__ = "Björn S. Konrad"
__email__ = "konradb@phys.ethz.ch"
__status__ = "Development"

# Standard Libraries
import configparser
import shutil
import os





# Function to update the config_file
def Update_Config(DIR,Section,Variable,Newval,from_original_config=False):
    config = configparser.ConfigParser(inline_comment_prefixes=('#',))
    config.optionxform=str

    if os.path.isfile(DIR+'input_original.ini'):
        if from_original_config:
            config.read(DIR + 'input_original.ini')
        else:
            config.read(DIR + 'input.ini')
    else:
        shutil.copyfile(DIR+'input.ini', DIR+'input_original.ini')
        config.read(DIR + 'input.ini')

    config[Section][Variable] = Newval

    with open(DIR+'input.ini', 'w') as configfile:
        config.write(configfile)
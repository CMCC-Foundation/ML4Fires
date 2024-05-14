# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 					Copyright 2024 - CMCC Foundation						
#																			
# Site: 			https://www.cmcc.it										
# CMCC Institute:	IESP (Institute for Earth System Predictions)
# CMCC Division:	ASC (Advanced Scientific Computing)						
# Author:			Emanuele Donno											
# Email:			emanuele.donno@cmcc.it									
# 																			
# Licensed under the Apache License, Version 2.0 (the "License");			
# you may not use this file except in compliance with the License.			
# You may obtain a copy of the License at									
#																			
#				https://www.apache.org/licenses/LICENSE-2.0					
#																			
# Unless required by applicable law or agreed to in writing, software		
# distributed under the License is distributed on an "AS IS" BASIS,			
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.	
# See the License for the specific language governing permissions and		
# limitations under the License.											
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import toml
import munch

from Fires._utilities.decorators import export

_config_dir =  os.path.join(os.path.dirname(os.getcwd()), "config")
print(_config_dir)

@export
def load_global_config(dir_name:str=_config_dir , config_fname : str = "configuration.toml"):
	"""
	Loads the global configuration from a TOML file.

	Args:
		dir_name (str, optional): The directory containing the configuration file.
			Defaults to "_config_dir".
		config_fname (str, optional): The filename of the configuration file.
			Defaults to "configuration.toml".

	Returns:
		munch.Munch: A dictionary-like object containing the loaded configuration.
	"""
	filepath = os.path.join(dir_name, config_fname)
	return munch.munchify(toml.load(filepath))

@export
def save_global_config(new_config:dict, folder:str=_config_dir, filename:str="new_configuration.toml"):
	"""
	Saves a new global configuration to a TOML file.

	Args:
		new_config (dict): The new configuration to be saved.
		folder (str, optional): The directory to save the configuration file.
			Defaults to "_config_dir".
		filename (str, optional): The filename of the configuration file.
			Defaults to "new_configuration.toml".
	"""
	path = os.path.join(folder, filename)
	with open(path , "w") as file:
		toml.dump(new_config , file)
		file.write("\n")

@export
def save_exp_config(new_config:dict|list, dir_name:str='experiments', filepath:str="experiments.toml"):
	"""
    Saves experiment configurations to a TOML file.

    Args:
        new_config (dict | list): The experiment configurations to be saved.
            - If `dict`, it represents a single experiment configuration.
            - If `list`, it represents a list of experiment configurations
              (each element being a dictionary).
        dir_name (str, optional): The directory to save the configuration file
            (relative to the global configuration directory). Defaults to "experiments".
        filepath (str, optional): The filename of the configuration file.
            Defaults to "experiments.toml".
    """
	base_path = os.path.join(_config_dir, dir_name)
	os.makedirs(name=base_path, exist_ok=True)
	path = os.path.join(base_path, filepath)
	with open(path , "w") as file:
		if type(new_config) == dict:
			experiment = {'exp': new_config}
			toml.dump(experiment , file)
			file.write("\n")
		elif type(new_config) == list:
			for i, d in enumerate(new_config):
				experiment = {f'exp_{i+1}': d}
				toml.dump(experiment , file)
				file.write("\n")
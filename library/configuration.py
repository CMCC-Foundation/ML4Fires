import toml
import munch
import os
import sys
from .decorators import export
_config_dir = os.path.join(os.getcwd(), 'config')

@export
def load_global_config(dir_name : str = _config_dir , config_fname : str = "configuration.toml"):
	filepath = os.path.join(dir_name, config_fname)
	return toml.load(filepath) # munch.munchify(toml.load(filepath))

@export
def save_global_config(new_config , filepath : str = "configuration.toml"):
	with open(filepath , "w") as file:
		toml.dump(new_config , file)
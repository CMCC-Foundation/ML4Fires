# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 					Copyright 2023 - CMCC Foundation						#
#																			#
# Site: 			https://www.cmcc.it										#
# CMCC Division:	ASC (Advanced Scientific Computing)						#
# Author:			Emanuele Donno											#
# Email:			emanuele.donno@cmcc.it									#
# 																			#
# Licensed under the Apache License, Version 2.0 (the "License");			#
# you may not use this file except in compliance with the License.			#
# You may obtain a copy of the License at									#
#																			#
#				https://www.apache.org/licenses/LICENSE-2.0					#
#																			#
# Unless required by applicable law or agreed to in writing, software		#
# distributed under the License is distributed on an "AS IS" BASIS,			#
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.	#
# See the License for the specific language governing permissions and		#
# limitations under the License.											#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import toml
import munch
import os
from .decorators import export
_config_dir = os.path.join(os.getcwd(), 'config')

@export
def load_global_config(dir_name : str = _config_dir , config_fname : str = "configuration.toml"):
	"""
	Load the TOML configuration file

	Parameters
	----------
	dir_name : str, optional
		Path to directory with TOML configuration files, by default _config_dir
	config_fname : str, optional
		Configuration file name, by default "configuration.toml"

	Returns
	-------
	dict[str, Any]
		Dictionary with configuration key-values pairs
	"""
	filepath = os.path.join(dir_name, config_fname)
	return toml.load(filepath) # munch.munchify(toml.load(filepath))

@export
def save_global_config(new_config , filepath : str = os.path.join(_config_dir, "configuration.toml")):
	"""
	Save new TOML configuration file

	Parameters
	----------
	new_config : dict
		Dictionary with key-value pairs defining the new TOML configuration file
	filepath : str, optional
		Path to directory with TOML configuration files, by default "$PWD/config/configuration.toml"
	"""
	with open(filepath , "w") as file:
		toml.dump(new_config , file)
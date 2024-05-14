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
import inspect
import numpy as np
import xarray as xr

from Fires._macros.macros import (
	DATA_FPATH,
	CONFIG,
	LOG_DIR,
	DATA_DIR,
	NEW_DS_PATH,
)
from Fires._utilities.decorators import debug, export
from Fires._utilities.logger import Logger as logger

_log = logger(log_dir=LOG_DIR).get_logger("DatasetUtils")

# toml_g_data = config.data
# toml_g_features = toml_g_data['features']		

@export
@debug(log=_log)
class DatasetZarr:
	"""
	Class used to create a xarray Dataset with the 
	features specified in a TOML configuration file
	"""

	def __init__(self) -> None:
		# define logger
		self.logger = _log

		# define path to zarr file
		self.path_to_zarr = DATA_FPATH
		self.path_new_zarr = NEW_DS_PATH

		# define drivers and targets
		self.drivers = sorted(CONFIG.data.features.drivers) + CONFIG.data.features.landsea_mask
		self.targets = sorted(CONFIG.data.features.targets)

		# save the current dataset in a zarr file
		for target in self.targets:
			name = target.split('_')[0].lower()
			path = self.path_new_zarr(name=name)
			if not os.path.exists(path):
				# get the dataset with all features	
				self.logger.info(f"Creating {path.split('/')[1]}")
				self._get_dataset(target=target)
				self.logger.info(f"Saving dataset to {path}")
				self.dataset.to_zarr(path)
			else:
				self.logger.warning(f"File {path} already exists. Skipping")
	
	def _bin_mask(self, data):
		"""
		Create binary mask maps for xarray DataArray

		Parameters
		----------
		data : xarray DataArray
			Variable data that must be masked

		Returns
		-------
		xarray DataArray
			Binary mask for the xarray DataArray in input
		"""
		fn_name = inspect.currentframe().f_code.co_name
		self.logger.info(f"{fn_name} | Creating binary mask")
		temp = data.where(data>0, 0)
		return temp.where(temp==0, 1)


	def _get_dataset(self, target:str):
		"""
		Generates the dataset used to train ML models.

		Parameters
		----------
		target : str
			defines the target feature
		"""
		# 
		
		fn_name = inspect.currentframe().f_code.co_name

		# define initial features
		self.logger.info(f"{fn_name} | Define features that must be used to retrieve data")
		_init_features = self.drivers + [target]
		for feature in _init_features: 
			self.logger.info(f" - {feature.upper()}")

		# define valid mask for current target variable
		_valid_mask = [f'{target}_valid_mask']
		self.logger.info(f"{fn_name} | Target: {target}, Valid Mask: {_valid_mask[0]}")

		# define path to .zarr dataset file and load it
		_ds = xr.open_zarr(self.path_to_zarr)[_init_features + _valid_mask]
		self.logger.info(f"{fn_name} | Loaded dataset from: {self.path_to_zarr}")

		# create binary masks
		self.logger.info(f"{fn_name} | Creating binary masks for {self.drivers[-1]}")
		_ds[self.drivers[-1]] = self._bin_mask(_ds[self.drivers[-1]]).expand_dims(time=_ds.time)

		# turn target's burned areas from hectares to percentage of hectares
		max_hectares = pow((111/4), 2)*100
		min_trg =  _ds[target].min(dim=['time', 'latitude', 'longitude'], skipna=True).load().data
		max_trg =  _ds[target].max(dim=['time', 'latitude', 'longitude'], skipna=True).load().data
		print(f"MIN: {min_trg} - MAX: {max_trg} - MAX HECT: {max_hectares} - IS MAX GREATHER THAN MAX HECT: {max_trg > max_hectares} \n Dataset target {target}: \n {_ds[[target]]} \n")
		max_pxl_value = max_hectares if max_hectares > max_trg else max_trg
		_ds[target] = _ds[target] / max_pxl_value

		# get valid data from '(target)_valid_mask' variable
		_valid_ds = _ds[_valid_mask]
		_valid_dates = [time for time in _valid_ds.time.data if _valid_ds.sel(time=str(time)) == 1]

		# define the final dataset that must be saved
		self.dataset = _ds[_init_features].sel(time = slice(str(_valid_dates[0]), str(_valid_dates[-1])))



@export
@debug(log=_log)
def load_zarr(name:str):
	"""
	Load the preprocessed `.zarr` dataset with xarray.

	Returns
	-------
	data: xarray.core.Data.Dataset
		Preprocessed Xarray Dataset with all the features needed
	"""
	path = NEW_DS_PATH(name=name.lower())
	# create dataset if not exists
	if not os.path.exists(path): DatasetZarr()
	# open dataset zarr file
	data = xr.open_zarr(path) #.load()
	return data



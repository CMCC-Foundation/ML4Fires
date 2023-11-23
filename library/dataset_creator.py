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

import os
import inspect
import xarray as xr
import numpy as np
from .dataset_builder_wf import WildFiresDatasetWriter
from .tfr_io import DriverInfo
from .decorators import debug, export
from .macros import (LOG_DIR, DATA_DIR, AGGREGATION, CREATOR_NOTES, DESCRIPTION, DOWNLOADED_FROM, LONG_NAME, PROVIDER)

from .logger import Logger as logger
_log = logger(log_dir=LOG_DIR).get_logger("DatasetCreator")

from .configuration import load_global_config
toml_general = load_global_config()
toml_model = eval(toml_general['toml_configuration_files']['toml_model'])

@export
@debug(log=_log)
class DatasetCreator():

	def __init__(self, years:list=[], target_source:str=None, shift_list:list=[]) -> None:
		
		self.logger = _log
		
		if target_source is None:
			raise ValueError('The target source must not be None')
		
		if target_source.upper() not in ['FCCI', 'GWIS', 'MERGE']:
			raise ValueError(f'Specify a valid target source between FCCI, GWIS or MERGE. Current value is {target_source}')

		if not shift_list:
			raise ValueError(f'List of shifts must not be empty')
		
		self.logger.info(f'Target source: {target_source.upper()}')
		self.target_source=target_source.upper()
		
		self.logger.info(f'Shift list: {shift_list}')
		self.shift_list = shift_list

		self._get_years(years)
		self._get_dataset()
	
	def build(self):
		# merge GWIS and FCCI burned areas data
		fn_name = inspect.currentframe().f_code.co_name

		# combine data sources
		self.logger.info(f"{fn_name} | Combining burned areas data sources")
		self._combine_ba_sources()

		# select feature data
		ds = self.dataset[self.features]

		# shifting dataset
		self.logger.info(f"{fn_name} | Shifting Dataset")
		self._shift_dataset(data=ds)

		self.logger.info(f"{fn_name} | Dataset successfully built")
		
		return self.folders, self.drivers_info


	def _get_years(self, years:list):
		# check if years list is empty or not
		fn_name = inspect.currentframe().f_code.co_name
		self.logger.info(f"{fn_name} | List of selected years is {'empty' if not years else years}")
		self.years = list(range(2001, 2022, 1)) if not years else years

	
	def _bin_mask(self, data):
		fn_name = inspect.currentframe().f_code.co_name
		self.logger.info(f"{fn_name} | Creating binary mask")
		temp = data.where(data>0, 0)
		return temp.where(temp==0, 1)
	

	def _longitude_fillnan(self, var_name, data):
		# fill NaN values for Greenwich longitude (it must be a problem due to the dataset creation methodology)
		fn_name = inspect.currentframe().f_code.co_name
		self.logger.info(f"{fn_name} | Filling longitude values for variable {var_name}")
		var_ds = data[var_name]
		var_ds[:, :, 719] = (var_ds[:, :, 718] + var_ds[:, :, 720])/2
		return var_ds


	def _setup_dataset(self):
		fn_name = inspect.currentframe().f_code.co_name
		self.logger.info(f"{fn_name} | Setting up dataset")
		
		# create binary masks
		self.logger.info(f"{fn_name} | Creating binary masks for {self.target_masks[0]}, {self.target_masks[1]} and {self.land_sea_mask[0]}")
		self.dataset[self.target_masks[0]] = self._bin_mask(self.dataset[self.targets[0]])
		self.dataset[self.target_masks[1]] = self._bin_mask(self.dataset[self.targets[1]])
		self.dataset[self.land_sea_mask] = self._bin_mask(self.dataset[self.land_sea_mask[0]]).expand_dims(time=self.dataset.time)

		# create drivers and targets features
		self.logger.info(f"{fn_name} | Creating drivers and targets features")
		self.driver_features = sorted(self.drivers) + self.land_sea_mask
		
		fcci_targets = [self.targets[0], self.target_masks[0]]
		gwis_targets = [self.targets[1], self.target_masks[1]]
		merged_targets = self.merged_ba+self.merged_ba_mask

		if self.target_source=='FCCI': self.target_features=fcci_targets
		elif self.target_source=='GWIS': self.target_features=gwis_targets
		elif self.target_source=='MERGE': self.target_features=merged_targets

		# choose which subset of targets
		self.features = self.driver_features + self.target_features


	def _get_dataset(self):
		fn_name = inspect.currentframe().f_code.co_name

		_features_dict = toml_general['data']['features']
		# define path to data
		self.path = eval(toml_general['data']['seasfirecube_path'])
		self.logger.info(f"{fn_name} | Path to data: {self.path}")

		# define drivers and targets
		self.logger.info(f"{fn_name} | Defining drivers, targets and their masks")
		self.drivers = sorted(_features_dict['drivers'])
		self.targets = sorted(_features_dict['targets'])
		self.target_masks = sorted(_features_dict['target_masks'])
		self.land_sea_mask = _features_dict['landsea_mask']

		# define features for merged burned areas data
		self.logger.info(f"{fn_name} | Defining merged burned area feature and its mask labels")
		self.merged_ba = _features_dict['merged_ba']
		self.merged_ba_mask = _features_dict['merged_ba_mask']

		# merge drivers and targets to
		self.init_features = self.drivers + self.targets + self.target_masks + self.land_sea_mask
		self.logger.info(f"{fn_name} | Defined features that must be used to retrieve data")
		for feature in self.init_features: self.logger.info(f" - {feature.upper()}")
		
		# load dataset from .zarr file
		self.dataset = xr.open_zarr(self.path)[self.init_features]
		self.logger.info(f"{fn_name} | Loaded dataset")

		# setup dataset
		self._setup_dataset()


	def _shift_dataset(self, data):
		fn_name = inspect.currentframe().f_code.co_name
		self.logger.info(f"{fn_name} | Starting creation of shifted dataset...")

		# define drivers and targets features, and list of years
		drivers = self.driver_features
		targets = self.target_features
		years = self.years
		self.logger.info(f"{fn_name} | \n Drivers: {drivers} \n Targets: {targets} \n Years: {years}")

		# define target source dir and create if not exists
		# $PWD/Data/Target source 
		target_src_dir = os.path.join(DATA_DIR, self.target_source)
		os.makedirs(target_src_dir, exist_ok=True)

		# define lambda function to get folder name to store in DATA_DIR and an empty folder list
		# $PWD/Data/Target source/NN_days_dataset
		get_dir_name = lambda days : os.path.join(target_src_dir, f'{days}_days_dataset')
		self.folders = []

		_model_config = toml_general['data']['selected_configuration']
		shape = eval(toml_model[_model_config]['base_shape'])
		shard_size = toml_model[_model_config]['shard_size']

		# create DriverInfo() list for drivers and targets
		self.logger.info(f"{fn_name} | Defining list of DriverInfo() objects")
		driver_info = DriverInfo(vars=drivers, shape=shape)
		target_info = DriverInfo(vars=targets, shape=shape)
		self.drivers_info = [driver_info, target_info]

		# create folders with .tfrecord files
		self.logger.info(f"{fn_name} | Iterate over: {self.shift_list}")
		for shift in self.shift_list:
			self.logger.info("-------------------------------------------------------")

			# create folder for current shift value
			shift_days = shift*8
			name = '0'+str(shift_days) if shift_days in list(range(10)) else str(shift_days)
			folder = get_dir_name(days=name)
			os.makedirs(name=folder, exist_ok=True)
			self.folders.append(folder)
			self.logger.info(f"{fn_name} | Store {shift_days} days shifted data in {folder}")

			# create shifted dataset for drivers and targets
			self.logger.info(f"{fn_name} | Create Drivers and Targets dataset shifted by {shift_days} days")
			ds_1 = data[drivers].isel(time = slice(0, 966-shift))
			ds_2 = data[targets].isel(time = slice(shift, 966))

			# align targets time coordinate to drivers one
			self.logger.info(f"{fn_name} | Align Targets time coordinate to Drivers")
			if shift != 0: ds_2['time'] = ds_1['time']

			# create merged xarray shifted dataset
			self.logger.info(f"{fn_name} | Creating merged xArray shifted dataset")
			merged = xr.merge([ds_1, ds_2], join='outer')

			self.logger.info(f"{fn_name} | Creating Data Writer...")
			writer = WildFiresDatasetWriter(drivers_info=self.drivers_info, shard_size=shard_size)
			
			self.logger.info(f"{fn_name} | Adding sources to the Data Writer")
			writer = writer.source(years=years)
			writer = writer.source(data=merged)

			self.logger.info(f"{fn_name} | Processing data in the Data Writer")
			writer.process(dst=folder)
			
		self.logger.info("-------------------------------------------------------")
	
	def _update_data_info(self, feature):
		fn_name = inspect.currentframe().f_code.co_name
		self.logger.info(f"{fn_name} | Dataset infos: {feature.upper()} feature > Updating ")
		data_feature = self.dataset[feature]
		data_feature.attrs['aggregation'] = AGGREGATION
		data_feature.attrs['creator_notes'] = CREATOR_NOTES
		data_feature.attrs['description'] = DESCRIPTION
		data_feature.attrs['downloaded_from'] = DOWNLOADED_FROM
		data_feature.attrs['long_name'] = LONG_NAME
		data_feature.attrs['provider'] = PROVIDER
		self.logger.info(f"{fn_name} | Dataset infos: {feature.upper()} feature > Updated ")
		return data_feature
	
	def _combine_ba_sources(self):
		fn_name = inspect.currentframe().f_code.co_name

		self.logger.info(f"{fn_name} | Combining datasets Burned Area data sources")

		map1 = self.dataset.gwis_ba
		map2 = self.dataset.fcci_ba

		# Step 1: Check whether the points in the first map have null values (NaN)
		self.logger.info(f"{fn_name} | Check whether the points in the first map have NaN values")
		mask = np.isnan(map1)

		# Step 2: Add values from the second map only at points where the first map has null values
		self.logger.info(f"{fn_name} | Add values from the second map only at points where the first map has NaN values")
		combined_map = map1.where(~mask, other=map2)

		# Step 3: Average the values at common points
		self.logger.info(f"{fn_name} | Average the values at common points")
		mean_map = (map1 + map2) / 2

		# Step 4: Create a new DataArray containing the resulting values
		# Use combined_map where values from the second map were added only at the null points of the first map
		# And use mean_map where the average values at the common points have been calculated
		self.logger.info(f"{fn_name} | Create a new DataArray containing the resulting values")
		final_map = combined_map.where(mask, other=mean_map)
		
		# Step 5: Get the burned area mask map with the classic method
		self.logger.info(f"{fn_name} | Get the burned area mask map with the classic method")
		final_map_mask = self._bin_mask(final_map)

		# Step 6: Add map and mask data to the existing dataset
		self.logger.info(f"{fn_name} | Add map and mask data to the existing dataset")
		self.dataset[self.merged_ba[0]] = final_map
		self.dataset[self.merged_ba_mask[0]] = final_map_mask
		
		# Step 7: Update dataset informations
		self.logger.info(f"{fn_name} | Update dataset informations for the merged data source")
		self.dataset[self.merged_ba[0]] = self._update_data_info(feature=self.merged_ba[0])
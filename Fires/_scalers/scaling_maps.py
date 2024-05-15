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
import xarray as xr
from typing import List

from Fires._utilities.decorators import export


class Map():
	"""
	Base class for handling maps.

	Attributes:
		features (List[str]):
			List of feature names.
		years (List[int]):
			List of years to select data from.
		data_filepath (str):
			Path to the Zarr data file.
		store_dir (str):
			Directory to store generated maps.

	Methods:
		get_maps():
			Abstract method for retrieving maps.
		save():
			Abstract method for saving maps.
	"""
	def __init__(self, features: List[str], years: List[int], data_filepath: str, store_dir: str) -> None:
		if not features:
			raise ValueError("Provide a list of features. It must not be empty!")
		if not years:
			raise ValueError("Provide a list of years. It must not be empty!")
		if not os.path.exists(data_filepath):
			raise ValueError(f"Path {data_filepath} doesn't exist. Please provide a valid path to zarr file!")
		if not os.path.exists(store_dir):
			raise ValueError(f"Path {data_filepath} doesn't exist. Please provide a valid path!")
		pass

	def get_maps(self):
		raise NotImplementedError
	
	def save(self):
		raise NotImplementedError



@export
class StandardMaps(Map):
	"""
	Calculates and saves standard deviation and mean maps for specified features and years.

	Attributes:
		mean_map (xr.DataArray):
			Mean values across time for each feature.
		stdv_map (xr.DataArray):
			Standard deviation values across time for each feature.
		name (str):
			Name derived from the last feature.

	Methods:
		__init__(features, years, data_filepath, store_dir):
			Initializes the class.
		get_maps():
			Calculates mean and standard deviation maps.
		save():
			Saves the maps to NetCDF files.
	"""
	def __init__(self, features: List[str], years: List[int], data_filepath: str, store_dir: str) -> None:
		super().__init__(features, years, data_filepath, store_dir)
		
		self.features = features
		self.training_years = years
		self.data_filepath = data_filepath
		self.store_dir = store_dir
		self.name = self.features[-1].split('_')[0]

		self.get_maps()
		self.save()
	
	def get_maps(self):
		ds = xr.open_zarr(self.data_filepath)[self.features]
		ds = ds.sel(time=ds.time.dt.year.isin(self.training_years))

		self.mean_map = ds.mean(dim='time', skipna=True)
		self.stdv_map = ds.std(dim='time', skipna=True)

	def save(self):
		mean_map_fname = os.path.join(self.store_dir, f'{self.name}_mean_map.nc')
		stdv_map_fname = os.path.join(self.store_dir, f'{self.name}_stdv_map.nc')

		try:
			self.mean_map.to_netcdf(mean_map_fname)
			self.stdv_map.to_netcdf(stdv_map_fname)
			print("Maps saved correctly")
		except:
			print("Error: maps can't be saved")



@export
class StandardMapsPointWise(Map):
	"""
	Calculates and saves mean and standard deviation maps point-wise across time, latitude and longitude,
	for specified features and years from a Zarr dataset.

	Attributes:
		features (List[str]):
			List of feature names to process.
		years (List[int]):
			List of years to select data from.
		data_filepath (str):
			Path to the Zarr dataset file.
		store_dir (str):
			Directory to save the calculated maps.
		name (str):
			Name of the variable (derived from the last feature name).
		mean_map (xarray.DataArray):
			Mean values across time, latitude, and longitude for each feature.
		stdv_map (xarray.DataArray):
			Standard deviation values across time, latitude, and longitude for each feature.

	Methods:
		__init__(features, years, data_filepath, store_dir):
			Initializes the StandardMapsPointWise object.
		get_maps():
			Calculates and returns the mean and standard deviation maps.
		save():
			Saves the mean and standard deviation maps to NetCDF files in the store directory.
	"""
	def __init__(self, features: List[str], years: List[int], data_filepath: str, store_dir: str) -> None:
		super().__init__(features, years, data_filepath, store_dir)

		self.features = features
		self.training_years = years
		self.data_filepath = data_filepath
		self.store_dir = store_dir
		self.name = features[-1].split('_')[0]
		
	def get_maps(self):
		ds = xr.open_zarr(self.data_filepath)[self.features]
		ds = ds.sel(time=ds.time.dt.year.isin(self.training_years))

		self.mean_map = ds.mean(dim=['time', 'latitude', 'longitude'], skipna=True).load()
		self.stdv_map = ds.std(dim=['time', 'latitude', 'longitude'], skipna=True).load()

		return self.mean_map, self.stdv_map

	def save(self):
		mean_map_fname = os.path.join(self.store_dir, f'{self.name}_mean_point_map.nc')
		stdv_map_fname = os.path.join(self.store_dir, f'{self.name}_stdv_point_map.nc')
		
		try:
			self.mean_map.to_netcdf(mean_map_fname)
			print(" Mean data \n Maps saved correctly")
		except:
			print(" Mean data \n Error: maps can't be saved")
		
		try:
			self.stdv_map.to_netcdf(stdv_map_fname)
			print(" Stdv data \n Maps saved correctly")
		except:
			print(" Stdv data \n Error: maps can't be saved")


@export
class MinMaxMaps(Map):
	"""
	Calculates and saves minimum and maximum maps for specified features and years.

	Attributes:
		min_map (xr.DataArray):
			Minimum values across time for each feature.
		max_map (xr.DataArray):
			Maximum values across time for each feature.
		name (str):
			Name derived from the last feature.

	Methods:
		__init__(features, years, data_filepath, store_dir):
			Initializes the class.
		get_maps():
			Calculates minimum and maximum maps.
		save():
			Saves the maps to NetCDF files.
	"""
	def __init__(self, features: List[str], years: List[int], data_filepath: str, store_dir: str) -> None:
		super().__init__(features, years, data_filepath, store_dir)
		
		self.features = features
		self.training_years = years
		self.data_filepath = data_filepath
		self.store_dir = store_dir
		self.name = self.features[-1].split('_')[0]

		self.get_maps()
		self.save()

	def get_maps(self):
		ds = xr.open_zarr(self.data_filepath)[self.features]
		ds = ds.sel(time=ds.time.dt.year.isin(self.training_years))

		self.min_map = ds.min(dim='time', skipna=True)
		self.max_map = ds.max(dim='time', skipna=True)
	
	def save(self):
		min_map_fname = os.path.join(self.store_dir, f'{self.name}_min_map.nc')
		max_map_fname = os.path.join(self.store_dir, f'{self.name}_max_map.nc')

		try:
			self.min_map.to_netcdf(min_map_fname)
			self.max_map.to_netcdf(max_map_fname)
			print("Maps saved correctly")
		except:
			print("Error: maps can't be saved")



@export
class MinMaxMapsPointWise(Map):
	"""
	Calculates and saves minimum and maximum maps point-wise across time, latitude and longitude,
	for specified features and years from a Zarr dataset.

	Attributes:
		features (List[str]):
			List of feature names to process.
		years (List[int]):
			List of years to select data from.
		data_filepath (str):
			Path to the Zarr dataset file.
		store_dir (str):
			Directory to save the calculated maps.
		name (str):
			Name of the variable (derived from the last feature name).
		min_map (xarray.DataArray):
			Minimum values across time, latitude, and longitude for each feature.
		max_map (xarray.DataArray):
			Maximum values across time, latitude, and longitude for each feature.

	Methods:
		__init__(features, years, data_filepath, store_dir):
			Initializes the MinMaxMapsPointWise object.
		get_maps():
			Calculates and returns the minimum and maximum maps.
		save():
			Saves the minimum and maximum maps to NetCDF files in the store directory.
	"""
	def __init__(self, features: List[str], years: List[int], data_filepath: str, store_dir: str) -> None:
		super().__init__(features, years, data_filepath, store_dir)
		
		self.features = features
		self.training_years = years
		self.data_filepath = data_filepath
		self.store_dir = store_dir
		self.name = self.features[-1].split('_')[0]

	def get_maps(self):
		ds = xr.open_zarr(self.data_filepath)[self.features]
		ds = ds.sel(time=ds.time.dt.year.isin(self.training_years))

		self.min_map = ds.min(dim=['time', 'latitude', 'longitude'], skipna=True).load()
		self.max_map = ds.max(dim=['time', 'latitude', 'longitude'], skipna=True).load()

		return self.min_map, self.max_map
	
	def save(self):
		min_map_fname = os.path.join(self.store_dir, f'{self.name}_min_point_map.nc')
		max_map_fname = os.path.join(self.store_dir, f'{self.name}_max_point_map.nc')

		try:
			self.min_map.to_netcdf(min_map_fname)
			print(" Min data \n Maps saved correctly")
		except:
			print(" Min data \n Error: maps can't be saved")

		try:
			self.max_map.to_netcdf(max_map_fname)
			print(" Max data \n Maps saved correctly")
		except:
			print(" Max data \n Error: maps can't be saved")

		


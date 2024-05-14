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
import torch
import xarray as xr
from typing import List, Any
from torch.utils.data import Dataset

from Fires._scalers.base import Scaler

class FireDataset(Dataset):
	"""
	A PyTorch Dataset class for loading and preprocessing fire data from a Zarr archive.

	This class loads driver and target features from a Zarr archive, selects data for specific years, 
	performs scaling (if scalers are provided), handles missing values, and returns tensors 
	suitable for training a PyTorch model.

	Attributes
	----------
	src : str
		Path to the Zarr archive containing the dataset.
	drivers : List[str])
		List of driver feature names to load from the Zarr archive.
	targets : List[str]
		List of target feature names to load from the Zarr archive.
	years : List[int]
		List of years for which to select data.
	scalers : List[Scaler], optional
		List of two scalers, one for drivers and one for targets. Defaults to [None, None].
	fill_value : float, optional
		Value to use for replacing missing values (NaN). Defaults to 0.0.
	dtype : torch.dtype, optional
		Data type of the returned tensors. Defaults to torch.float32.
	X : torch.Tensor
		Tensor containing the driver features.
	Y : torch.Tensor
		Tensor containing the target features.
	x_scaler : Scaler, optional
		Scaler used for driver features (if provided).
	y_scaler : Scaler, optional
		Scaler used for target features (if provided).

	Raises
	------
	ValueError
		If the provided path is invalid, any of the required lists are empty, 
		or the scalers list doesn't contain exactly two elements.
	"""
	def __init__(self, src: str, drivers: List[str], targets: List[str], years: List[int], scalers: List[Scaler] = [None,None], fill_value: float = 0.0, dtype=torch.float32) -> None:
		super().__init__()

		if not os.path.exists(src):
			raise ValueError(f"Path {src} doesn't exist. Please provide a valid path for zarr file!")
		if not drivers:
			raise ValueError("Please provide a list of driver features!")
		if not targets:
			raise ValueError("Please provide a list of target features!")
		if not years:
			raise ValueError("Please provide a list of years!")
		if not scalers:
			raise ValueError("Please provide a list of 2 scalers, one for drivers and one for targets!")
		if len(scalers) != 2:
			raise ValueError(f"Please provide a list of 2 scalers, one for drivers and one for targets! (Received: {len(scalers)})")
		
		# open zarr dataset
		ds = xr.open_zarr(src)[drivers+targets]
		# select only dataset years
		ds = ds.sel(time=ds.time.dt.year.isin(years))
		# get torch dataset (X and Y)
		self.X = torch.as_tensor(data=ds[drivers].to_array().load().values, dtype=dtype)
		self.Y = torch.as_tensor(data=ds[targets].to_array().load().values, dtype=dtype)
		# permute to compatible shape N x C x H x W
		self.X = self.X.permute(dims=(1, 0, 2, 3))
		self.Y = self.Y.permute(dims=(1, 0, 2, 3))
		# store scalers, if provided
		self.x_scaler = scalers[0]
		self.y_scaler = scalers[1]
		self.n = ds.time.data.shape[0]
		self.fill_value = fill_value

	def __len__(self):
		"""
		Returns the length of the dataset.
		"""
		return self.n

	def __getitem__(self, index) -> Any:
		"""
		Retrieves a single data point from the dataset.

		Parameters
		----------
		index : int
			Index of the data point to retrieve.

		Returns
		-------
		x, y : Tuple[torch.Tensor, torch.Tensor]
			A tuple containing the driver features (X) 
			and target features (Y) scaled data for the given index.
		"""
		# get the data from dataset
		x, y = self.X[index], self.Y[index]
		# scale the data
		if self.x_scaler:
			x = self.x_scaler.transform(x)
		if self.y_scaler:
			y = self.y_scaler.transform(y)
		# remove possible nan values
		x = torch.nan_to_num(x, nan=self.fill_value)
		y = torch.nan_to_num(y, nan=self.fill_value)
		# return x and y
		return x, y
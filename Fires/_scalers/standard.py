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
from typing import List

from Fires._scalers.base import Scaler
from Fires._utilities.decorators import export

@export
class StandardScaler(Scaler):
	"""
	Scales data using mean and standard deviation.

	Attributes:
		mean_ds (xr.Dataset):
			xarray Dataset containing mean values for features.
		stdv_ds (xr.Dataset):
			xarray Dataset containing standard deviation values for features.
		coeff_ (torch.Tensor):
			Scaling coefficients derived from mean and stdv values.
		element_ (torch.Tensor):
			Scaling offset derived from mean and stdv values.

	Methods:
		__init__(mean_ds, stdv_ds, features, dtype):
			Initializes the scaler.
		transform(tensor):
			Standardizes the input tensor.
		inverse_transform(tensor):
			Reverses the standardization.
	"""
	def __init__(self, mean_ds, stdv_ds, features:List[str], dtype=torch.float32) -> None:
		super().__init__(features, dtype)

		if not features:
			raise ValueError("Features list must be non-empty!")
		
		if not mean_ds:
			raise ValueError("Data for min values must not be empty!")
		
		if not stdv_ds:
			raise ValueError("Data for min values must not be empty!")
		
		mean_ds = mean_ds[features]
		stdv_ds = stdv_ds[features]

		coeff = 1 / stdv_ds
		element = mean_ds * coeff
		
		self.coeff_ = torch.as_tensor(coeff.to_array().data, dtype=self.dtype).view(8, 1, 1)
		self.element_ = torch.as_tensor(element.to_array().data, dtype=self.dtype).view(8, 1, 1)
		
	def transform(self, tensor: torch.Tensor):
		scaled_tensor = tensor * self.coeff_ - self.element_
		return scaled_tensor

	def inverse_transform(self, tensor: torch.Tensor):
		rescaled_tensor = (tensor + self.element_) / self.coeff_
		return rescaled_tensor
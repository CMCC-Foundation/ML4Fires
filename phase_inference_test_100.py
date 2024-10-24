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
import sys
import toml
import inspect
import numpy as np
import xarray as xr

from tqdm import tqdm
from typing import Any, List
from itertools import islice

import torch
import torch.nn as nn

from Fires._datasets.torch_dataset import FireDataset
from Fires._macros.macros import (
	CONFIG,
	LOGS_DIR,
	EXPS_PTH,
	DRIVERS,
	TARGETS,
	DATA_PATH_100KM,
	MAX_HECTARES_100KM,
)
from Fires._models.unet import Unet
from Fires._models.unetpp import UnetPlusPlus
from Fires._scalers.standard import StandardScaler
from Fires._plots.plot_utils import plot_dataset_map
from Fires._utilities.decorators import debug
from Fires._utilities.logger import Logger as logger

# define logger
_log = logger(log_dir=LOGS_DIR).get_logger("Inference_on_100km")


@debug(log=_log)
def compute_aggregated_data(data, other_data=None, operation="mean") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Compute the mean or difference between data, and aggregate along latitudes and longitudes

	Parameters
	----------
	data : numpy.ndarray
	 	Input data, can be unscaled or already scaled and masked depending on the operation to be performed
	other_data : numpy.ndarray, optional
	 	Optional input data for calculating the difference, also assumed to be scaled and masked
		Required if `operation` is 'diff'.
	operation : str
		Operation to perform ("mean" for mean, "diff" for difference)

	Returns
	-------
	tuple of np.ndarray
		A tuple containing:
			- data : np.ndarray
				Scaled and masked data after the operation.
			- descaled_on_lats : np.ndarray
				Mean of data along latitudes.
			- descaled_on_lons : np.ndarray
				Mean of data along longitudes.
	
	Raises
	------
	ValueError
		If `operation` is 'diff' and `other_data` is not provided.

	"""

	# define function name
	fn_name = inspect.currentframe().f_code.co_name

	data = data.copy()

	if operation == "diff":
		if other_data is None:
			raise ValueError("other_data must be provided when operation is 'diff'")
		# difference between data that has been masked and rescaled to the original size
		data -= other_data
	else:
		# mask data with the land sea mask and rescale to original size
		data *= MAX_HECT_LSM_MAP

	descaled_on_lats = np.nanmean(data, axis=1)
	descaled_on_lons = np.nanmean(data, axis=0)

	_log.info(f"{fn_name} | {operation.capitalize()} of data: {data.shape}")
	_log.info(f"{fn_name} | Max: {round(np.nanmax(data), 2)} \t Min: {round(np.nanmin(data), 2)}")
	_log.info(f"{fn_name} | Lats Max: {round(np.nanmax(descaled_on_lats), 2)} \t Lons Max: {round(np.nanmax(descaled_on_lons), 2)}")

	return data, descaled_on_lats, descaled_on_lons


@debug(log=_log)
def load_model(model_path: str) -> nn.Module:
	"""
	Load the stored model from the given path.

	Parameters
	----------
	model_path : str
		Path to the saved model file.

	Returns
	-------
	nn.Module
		The loaded model ready for inference.

	"""

	# define model
	# model = Unet(
	# 	input_shape=(180, 360, 7),
	# 	base_filter_dim=128, #32 64 128 192
	# 	activation=torch.nn.modules.activation.Sigmoid()
	# )

	model = UnetPlusPlus(
		input_shape=(180, 360, 7),
		base_filter_dim=128, #32 64 128 192
		activation=torch.nn.modules.activation.Sigmoid(),
		depth=2
	)

	# define model loss
	model.loss = nn.BCELoss()
	# deifne model metrics
	model.metrics = []
	# load model from path
	load_model_state = torch.load(model_path, map_location=torch.device('cpu'))['model']
	# load weights
	model.load_state_dict(load_model_state)
	# evaluate model
	model.eval()
	return model


@debug(log=_log)
def make_predictions(model: nn.Module, data_loader: torch.utils.data.DataLoader) -> np.ndarray:
	"""
	Make predictions using the loaded model and the PyTorch data loader.

	Parameters
	----------
	model : nn.Module
		The loaded PyTorch model to use for predictions.
	data_loader : torch.utils.data.DataLoader
		The PyTorch DataLoader providing the data.

	Returns
	-------
	np.ndarray
		An array containing the predictions.

	"""
	
	preds = []
	with torch.no_grad():
		for data, _ in tqdm(data_loader):
			preds.append(model(data))
	preds_array = np.vstack(preds)
	return preds_array


@debug(log=_log)
def up_and_lower_bounds(avg_value, std_value):
	"""
	Compute upper and lower bound values.

	Parameters
	----------
	avg_value : np.ndarray or float
		The average values.
	std_value : np.ndarray or float
		The standard deviation values.

	Returns
	-------
	tuple
		A tuple containing the upper bound and lower bound values.

	"""

	_upper = avg_value + std_value
	_lower = avg_value - std_value
	return _upper, _lower


@debug(log=_log)
def prepare_data_loader(path_to_dataset:str, drivers_list:List[str], targets_list:List[str], list_of_years:List[int], scalers:List[StandardScaler|None], batch_size:int=1):
	"""
	Prepare a PyTorch DataLoader for the test data.

	Parameters
	----------
	path_to_dataset : str
		Absolute path to the stored dataset that must be loaded.
	drivers_list : List[str]
		List of driver features.
	targets_list : List[str]
		List of target features.
	list_of_years : List[int]
		List of years related to the test set.
	scalers : List[StandardScaler or None]
		List of scalers; the first one is for drivers data, the second one is for target data.
	batch_size : int, optional
		Size of the batch that must be loaded when the DataLoader is called, by default 1.

	Returns
	-------
	torch.utils.data.DataLoader
		PyTorch DataLoader for test data.

	"""

	torch_dataset = FireDataset(
		src=path_to_dataset,
		drivers=drivers_list,
		targets=targets_list,
		years=list_of_years,
		scalers=scalers
	)

	torch_data_loader = torch.utils.data.DataLoader(
		torch_dataset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True
	)
	
	return torch_data_loader


@debug(log=_log)
def process_and_plot_data(data, label, lats, lons, model_name):
	"""
	Process the data and generate plots.

	Parameters
	----------
	data : xarray.DataArray or np.ndarray
		Data to process; can be an xarray.DataArray for real data or a numpy.ndarray for predictions.
	label : str
		Label to use in the plot title.
	lats : np.ndarray
		Array of latitudes.
	lons : np.ndarray
		Array of longitudes.
	model_name : str
		Name of the model, used in the plot title.

	"""
	
	# Verify data type and compute mean and standard deviation along time axis
	if isinstance(data, xr.DataArray):
		avg_on_time = data.mean(dim='time', skipna=True).data
		std_on_time = data.std(dim='time', skipna=True).data
		print(f"Is DataArray - AVG: {avg_on_time.shape} STD: {std_on_time.shape}")
	else:
		avg_on_time = np.nanmean(data, axis=0)[0, ...]
		std_on_time = np.nanstd(data, axis=0)[0, ...]
		print(f"NOT DataArray - AVG: {avg_on_time.shape} STD: {std_on_time.shape}")

	# Aggregate data
	avg_descaled, avg_on_lats, _ = compute_aggregated_data(data=avg_on_time)
	_, std_on_lats, _ = compute_aggregated_data(data=std_on_time)

	# Compute upper and lower boundaries
	upperbound, lowerbound = up_and_lower_bounds(avg_value=avg_on_lats, std_value=std_on_lats)

	# Plot data
	plot_dataset_map(
		avg_target_data=avg_descaled,
		avg_data_on_lats=avg_on_lats,
		lowerbound_data=lowerbound,
		upperbound_data=upperbound,
		lats=lats,
		lons=lons,
		title=f'{label} ({model_name.upper()})',
		cmap='nipy_spectral_r'
	)


@debug(log=_log)
def main():
	# load features
	drivers, targets = DRIVERS, TARGETS

	# define path to complete dataset
	DS_PATH = DATA_PATH_100KM

	# open the dataset and choose a subset
	dataset = xr.open_zarr(DS_PATH)[drivers + targets].load()
	test_data = dataset.sel(time=slice('2019', '2020'))

	# load the land sea mask and substitute zeros with NaN values
	lsm = test_data.lsm.mean(dim='time', skipna=True).values
	lsm[lsm == 0] = np.nan
	print(lsm.shape)

	# define MAX_HECT_LSM_MAP as global
	global MAX_HECT_LSM_MAP
	MAX_HECT_LSM_MAP = lsm * MAX_HECTARES_100KM

	# define latitudes and longitudes
	lats = dataset.latitude.values
	lons = dataset.longitude.values
	_log.info(f"Latitude count: {len(lats)} \t Longitude count: {len(lons)}")

	# path to the experiments folder with last model
	PATH_TO_EXP_FOLDER = EXPS_PTH(dirname='20240920_upp/20240920_172117')
	_log.info(f"Path to the experiment folder: {PATH_TO_EXP_FOLDER}")

	# load the model
	model_path = os.path.join(PATH_TO_EXP_FOLDER, 'last_model.pt')
	model = load_model(model_path=model_path)

	# define trianing dataset
	ds_trn = dataset.sel(time=slice('2001', '2016'))
	# compute mean along time, latitude and longitude axes
	mean_ds = ds_trn.mean(dim=['time','latitude', 'longitude'], skipna=True)
	# compute standard deviation along time, latitude and longitude axes
	stdv_ds = ds_trn.std(dim=['time','latitude', 'longitude'], skipna=True)
	# define scaler
	x_scaler = StandardScaler(mean_ds=mean_ds, stdv_ds=stdv_ds, features=drivers)
	
	# define data loader for test data
	test_loader = prepare_data_loader(
		path_to_dataset=DS_PATH,
		drivers_list=drivers,
		targets_list=targets,
		list_of_years=list(range(2019, 2021)),
		scalers=[x_scaler, None]
	)

	_log.info("\n BEFORE PREDICTIONS \n")
	# perform predictions
	preds_array = make_predictions(model=model, data_loader=test_loader)
	_log.info("\n AFTER PREDICTIONS \n")


	model_name = "Unet ++" # "Unet"

	# Process and plot real data
	process_and_plot_data(
		data=test_data.fcci_ba,
		label='FCCI Burned Areas - Real',
		lats=lats,
		lons=lons,
		model_name=model_name
	)

	# Process and plot predicted data
	process_and_plot_data(
		data=preds_array,
		label='Predicted Burned Areas',
		lats=lats,
		lons=lons,
		model_name=model_name
	)



if __name__ == '__main__':
	main()
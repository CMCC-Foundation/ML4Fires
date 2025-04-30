import numpy as np
import xarray as xr
import os
import joblib
import torch
import pydot

from Fires._datasets.torch_dataset import FireDataset
from Fires._macros.macros import DRIVERS, TARGETS, MAX_HECTARES_100KM, LOGS_DIR, CONFIG
from Fires._plots.plot_utils import plot_dataset_map
from Fires._scalers.standard import StandardScaler
from Fires._utilities.logger import Logger as logger
from Fires._utilities.decorators import debug, export

# define logger
_log = logger(log_dir=LOGS_DIR).get_logger("Inference Utilities")


@export
@debug(log=_log)
def get_prov_image(run_name):
	#prov_doc = os.path.join(os.getcwd(), 'MLFLOW', f"{run_name}/provgraph_{CONFIG.mlflow.EXPERIMENT_NAME}.dot")
	#prov_img = os.path.join(os.getcwd(), 'MLFLOW', f"{run_name}/provgraph_{CONFIG.mlflow.EXPERIMENT_NAME}.png")
	#(graph,) = pydot.graph_from_dot_file(prov_doc)
	#graph.write_png(prov_img)
	prov_img = os.path.join(os.getcwd(), 'MLFLOW', f"{run_name}/provgraph_{CONFIG.mlflow.EXPERIMENT_NAME}.svg")
	return prov_img

@export
@debug(log=_log)
def load_input_data(data_path, time_start, time_end):
    drivers, targets = DRIVERS, TARGETS
    print(drivers, "\n", targets)

    # open the dataset and choose a subset
    dataset = xr.open_zarr(data_path)[drivers + targets].load()

    test_data = dataset.sel(time=slice(time_start, time_end))

    # load the land sea mask and substitute zeros with NaN values
    lsm = test_data.lsm.mean(dim='time', skipna=True).values
    lsm[lsm == 0] = np.nan
    print(lsm.shape)

    # define MAX_HECT_LSM_MAP as global
    global MAX_HECT_LSM_MAP, lats, lons
    lats = dataset.latitude.values
    lons = dataset.longitude.values
    MAX_HECT_LSM_MAP = lsm * MAX_HECTARES_100KM

    return test_data

def get_scaler(run_name:str):
    # define scaler
    local_path = os.path.join(os.getcwd(), 'MLFLOW', f"{run_name}/scaler/scaler.dump")
    return joblib.load(local_path)
    
@export
@debug(log=_log)
def create_data_loader(data_path, run_name):
    # get scaler
    x_scaler = get_scaler(run_name=run_name)

    # define torch dataset
    drivers, targets = DRIVERS, TARGETS
    torch_dataset = FireDataset(
        src=data_path,
        drivers=drivers,
        targets=targets,
        years=list(range(2019,2021)),
        scalers=[x_scaler, None]
    )

    torch_data_loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True
    )
    
    return torch_data_loader


@export
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

	print(f" {operation.capitalize()} of data: {data.shape}")
	print(f" Max: {round(np.nanmax(data), 2)} \t Min: {round(np.nanmin(data), 2)}")
	print(f" Lats Max: {round(np.nanmax(descaled_on_lats), 2)} \t Lons Max: {round(np.nanmax(descaled_on_lons), 2)}")

	return data, descaled_on_lats, descaled_on_lons


@export
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


@export
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
    
@export
@debug(log=_log)
def process_and_plot_cmip6infer(data: xr.Dataset, temporal_aggregate_scheme: list|str, label, lats, lons, model_name):
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
		if temporal_aggregate_scheme == "mean":
            # In case when simply wants to compute the average on the whole dataset 
			avg_on_time = data.mean(dim='time', skipna=True).data
			std_on_time = data.std(dim='time', skipna=True).data
			print(f"Is DataArray - AVG: {avg_on_time.shape} STD: {std_on_time.shape}")
		else:
            # Check that parameter temporal_aggregate_scheme is not an str and is not an empty list
			assert isinstance(temporal_aggregate_scheme, list), "For multi-scale aggregate, a list of aggregation type is required."
			assert len(temporal_aggregate_scheme) > 0, "For multi-scale aggregate, more than one aggregate methods should be provided."
			years_in_ds = np.unique(data.time.dt.year)
            # The first item in temporal_aggregate_scheme should not be a None or empty.
			assert temporal_aggregate_scheme[0].lower() != "none", "None cannot be given as an aggregate method for monthly scale."
			assert temporal_aggregate_scheme[0] != "", "'' cannot be given as an aggregate method."
            # Get aggreagated on monthly scale for the years in the dataset.
            # Default date is in start of the month.
			monthly_aggregate = eval(f"data.resample(time='1M',skipna=True).{temporal_aggregate_scheme[0]}()")

			# Check that a second entry is given for the yealy aggregate and is not a none or empty
			assert temporal_aggregate_scheme[1].lower() != "none", "None cannot be given as an aggregate method for yearly scale."
			assert len(temporal_aggregate_scheme) > 1, "There should be more than 1 aggregate method for yearly aggregation."
            # Aggregate on the year
			yearly_aggregate = eval(f"monthly_aggregate.resample(time='1Y',skipna=True).{temporal_aggregate_scheme[1]}()")
            
			if len(years_in_ds) > 1: # check if there are more than one year in the dataset - decadal scale prediction
				# If there are more than on year, then there should be an aggregate method in the list which is not none or empty
				assert len(temporal_aggregate_scheme) > 2 , "There should be more than aggregate method for multi-scale aggregation."
				assert temporal_aggregate_scheme[2].lower() != "none", "None cannot be given as an aggregate method for decadal scale."
				assert temporal_aggregate_scheme[2] != "", "'' cannot be given as an aggregate method for decadal scale."
                # Do decadal aggregate
				avg_on_time = eval(f"yearly_aggregate.{temporal_aggregate_scheme[2]}(dim='time',skipna=True)")
				std_on_time = yearly_aggregate.std(dim='time', skipna=True).data
			else:
                # No decadal aggregate in case you only 
				avg_on_time = yearly_aggregate 
				std_on_time = yearly_aggregate.std(dim='time', skipna=True).data
	else:
		if temporal_aggregate_scheme == ["mean","mean","mean"]:
			avg_on_time = np.nanmean(data, axis=0)
			std_on_time = np.nanstd(data, axis=0)
			print(f"NOT DataArray - AVG: {avg_on_time.shape} STD: {std_on_time.shape}")
		else:
			raise Exception("Different averaging on different time scales on works if prediction is provided in xr.DataArray format.")

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
    
def aggregate_var(dataset: xr.Dataset, method:str, dim:str='time'):
    eval_str = f"dataset.{method}(dim='{dim}',skipna=True,keep_attrs=True)"
    output = eval(eval_str)
    return output
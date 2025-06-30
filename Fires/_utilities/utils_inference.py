import numpy as np
import xarray as xr
import os
import joblib
import torch
import pydot
import datetime
from cftime import num2date, date2num

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
def process_and_plot_data(data,
                          label,
                          lats,
                          lons,
                          model_name,
                          scale_min: int=None,
                          scale_max: int=None):
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
		scale_min=scale_min if scale_min else np.nanmin(avg_descaled),
		scale_max=scale_max if scale_max else np.nanmax(avg_descaled),
		lats=lats,
		lons=lons,
		title=f'{label} ({model_name.upper()})',
		cmap='nipy_spectral_r'
	)
    
@export
@debug(log=_log)
def process_and_plot_cmip6infer(data: xr.Dataset,
                                temporal_aggregate_scheme: list|str,
                                label,
                                lats,
                                lons,
                                model_name,
                                scale_min: int=None,
                                scale_max: int=None):
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
			assert isinstance(temporal_aggregate_scheme, dict), "For multi-scale aggregate, a dictionary with keys 'monthly,  'year' and 'decadal' is required."
			assert len(temporal_aggregate_scheme) > 0, "For multi-scale aggregate, more than one aggregate methods should be provided."
			years_in_ds = np.unique(data.time.dt.year)
            
            # Monthly aggregate
            
            # Get aggreagated on monthly scale for the years in the dataset.
            # Default date is in start of the month.
			monthly_aggregate = eval(f"data.resample(time='{temporal_aggregate_scheme['monthly'][1]}',skipna=True).{temporal_aggregate_scheme['monthly'][0]}()")
            
            # Aggregate over the year
            # Aggregate on the year
			yearly_aggregate = eval(f"monthly_aggregate.resample(time='1Y',skipna=True).{temporal_aggregate_scheme['yearly']}()")
            
			if len(years_in_ds) > 1: # check if there are more than one year in the dataset - decadal scale prediction
				# If there are more than on year, then there should be an aggregate method in the list which is not none or empty
                # Do decadal aggregate
				avg_on_time = eval(f"yearly_aggregate.{temporal_aggregate_scheme['decadal']}(dim='time',skipna=True)")
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
        scale_max=scale_max,
        scale_min=scale_min,
		lats=lats,
		lons=lons,
		title=f'{label} ({model_name.upper()})',
		cmap='nipy_spectral_r'
	)
    
def aggregate_var(dataarray: xr.DataArray, method:str, dim:str='time'):
    eval_str = f"dataarray.{method}(dim='{dim}',skipna=True,keep_attrs=True)"
    output = eval(eval_str)
    return output


def _get_list_of_dates(year_range):
    n_prior_days = 8
    str_dates = [f'{year_range.value[0]}-01-08',f'{year_range.value[1]}-12-24'] #[start_date, end_date] yyyy-mm-dd

    np_dates = [np.datetime64(f"{str_date}T12:00:00.00") for str_date in str_dates]
    np_all_dates = [np_dates[0]]
    date = np_dates[0]
    while date < np_dates[1]:
        date += np.timedelta64(n_prior_days, "D")
        np_all_dates.append(date)
    date_range_np = [[np_date - np.timedelta64(n_prior_days-1, "D"), np_date] for np_date in np_all_dates]
    
    return date_range_np


def _get_cft_times_list(year_range):
    dates_range_np = _get_list_of_dates(year_range=year_range)
    
    dates_range_cfttime = []
    for single_date_range in dates_range_np:
        py_date = []
        for single_np_date in single_date_range:
            if single_np_date.astype(datetime.datetime).month != 2 and single_np_date.astype(datetime.datetime).day != 29:
                py_date.append(single_np_date.astype('datetime64[ms]').astype(object))
            else:
                single_np_date -= np.timedelta64(1, "D")
                py_date.append(single_np_date.astype('datetime64[ms]').astype(object))
        units = "days since 0000-01-01"
        # calendar_type = "standard" if calendar.isleap(int(str(single_date_range[0])[0:4])) else "noleap"
        calendar_type = "noleap"
        numeric_time = date2num(py_date, units, calendar_type, has_year_zero=True)
        dates_range_cfttime.append(num2date(numeric_time, units, calendar_type, has_year_zero=True))
    
    return dates_range_cfttime


def _get_file_list(scenario, config, year_range):
    import os
    dates_range_np = _get_list_of_dates(year_range=year_range)
    
    cmip6_var_filename = {}
    for var_key, var_value in config.data.drivers.items():
        cmip6_var_filename[var_key] = []
        if var_value.type == "dynamic":
            if "[scenario]" in var_value.cmip6_path:
                path_to_files = os.path.join(config.config.base_dir,var_value.cmip6_path.replace("[scenario]", scenario.value))
            else:
                path_to_files = os.path.join(config.config.base_dir,var_value.cmip6_path)
            list_of_files = [file for file in os.listdir(path_to_files) if file.endswith(".nc")]
            for file in list_of_files:
                date = file.split("_")[-1].split(".")[0].split("-")
                start_date = np.datetime64(f"{date[0][0:4]}-{date[0][4:6]}-{date[0][6:8]}")
                end_date = np.datetime64(f"{date[1][0:4]}-{date[1][4:6]}-{date[1][6:8]}")
                for date_range in dates_range_np:
                    if min(date_range) >= start_date and max(date_range)<=end_date and os.path.join(path_to_files,file) not in cmip6_var_filename[var_key]:
                        cmip6_var_filename[var_key].append(os.path.join(path_to_files,file))
        else:
            path_to_file = os.path.join(config.config.base_dir,var_value.cmip6_path.replace("[scenario]", scenario.value))
            file = [file for file in os.listdir(path_to_file) if file.endswith(".nc")]
            cmip6_var_filename[var_key] = os.path.join(path_to_file,file[0])
    
    print("Loading the following CMIP6 data files...")
    for key,value in cmip6_var_filename.items():
        print(f"{key}: {value}")
    
    return cmip6_var_filename, dates_range_np
            
def _read_and_aggregate_cmip6_data(seafire_ds, scenario, config, year_range):
    
    cmip6_var_filename, dates_range_np = _get_file_list(scenario=scenario, config=config,year_range=year_range)
    dates_range_cfttime = _get_cft_times_list(year_range=year_range)
    
    var_ds_list = []
    for var_name, var_file in cmip6_var_filename.items():
        print(f"Reading variable {var_name} and aggregating with method {config.data.drivers[var_name].aggregation}...")
        if config.data.drivers[var_name].aggregation.lower() != "none":
                ds_var = xr.open_mfdataset(var_file)[var_name]
                ds_var_time_slices = []
                for single_range_cfttime in dates_range_cfttime:
                    var_time_slice = ds_var.sel(time=slice(single_range_cfttime[0],single_range_cfttime[1]))
                    ds_var_time_slices.append(var_time_slice)
        else:
            ds_var = xr.open_dataset(var_file)[var_name]/100 # divide by 100 to get the same units as training dataset
        if config.data.drivers[var_name].aggregation.lower() != "none":
            for slice_idx, single_time_slice in enumerate(ds_var_time_slices):
                if var_name == "pr":
                    single_time_slice = single_time_slice*3600*24 # Convering flux to total precipitation
                ds_var_time_slices[slice_idx] = aggregate_var(dataarray=single_time_slice,method=config.data.drivers[var_name].aggregation,dim='time')
                ds_var_time_slices[slice_idx] = ds_var_time_slices[slice_idx].expand_dims({"time":[dates_range_np[slice_idx][-1]]})

            ds_var = xr.concat(ds_var_time_slices, dim="time")
        ds_var = ds_var.assign_coords({"lon": ((ds_var.lon + 180) % 360) - 180}).sortby("lon") # translating the longitude values
        ds_var = ds_var.rename({"lon":"longitude", "lat":"latitude"}) # renaming longitude and latitude for regridding
        ds_var = ds_var.interp_like(seafire_ds[["longitude","latitude"]])
        var_ds_list.append(ds_var)

    merged_ds_var = xr.merge(var_ds_list)
    merged_ds_var = merged_ds_var.isel(plev=0) # This is selected for the variable which also dependso on the pressure - this selects the sea level pressure
    
    ds_array = merged_ds_var.to_array().transpose("time", "variable", "latitude", "longitude").values
    return ds_array, merged_ds_var.time.values

def get_cmip6_inference(seafire_ds, run_name, scenario, year_range, config, model):
    
    print(f"Reading CMIP6 data for scenario {scenario.value} for year range {year_range.value[0]}-{year_range.value[1]}")
    ds_array, np_dates = _read_and_aggregate_cmip6_data(seafire_ds=seafire_ds,scenario=scenario, config=config, year_range=year_range)
    print(f"Dimensions of the input: ", ds_array.shape)
                                                               
    scaler = get_scaler(run_name=run_name)
    transformed_ds = scaler.transform(torch.as_tensor(ds_array))
    transformed_ds = transformed_ds.float()
    transformed_ds = torch.nan_to_num(transformed_ds,nan=0)
    
    print("Passing the processed CMIP6 data to the ML model for inference...")
    prediction_cpu = []
    with torch.no_grad():
        for idx in range(transformed_ds.shape[0]):
            prediction = model(transformed_ds[idx].unsqueeze(0).to('cuda:0'))
            prediction_cpu.append(prediction.cpu().detach().numpy())
    predictions = np.vstack(prediction_cpu).squeeze()
    
    print("Creating predicton dataset...")
    ds_predictions = xr.Dataset(data_vars={"global_burned_areas":(("time","latitude","longitude"),predictions)},
                            coords={"time":("time", np_dates),
                                    "longitude":("longitude", seafire_ds.longitude.values),
                                    "latitude":("latitude", seafire_ds.latitude.values)},
                           attrs={"Details": f"CMIP6 prediction for the scenario {scenario.value} from {year_range.value[0]} to {year_range.value[0]}. Prediction is made for the InterTwin Project.",
                                  "Processing": "No information",
                                 "Source": "CMCC Foundation."})
    ds_predictions = ds_predictions.sortby(variables="time")    
    
    return ds_predictions
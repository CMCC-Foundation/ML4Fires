import numpy as np
import xarray as xr
import os
import joblib
import torch
import pydot
import datetime
import re
from cftime import num2date, date2num
from Fires._datasets.torch_dataset import FireDataset
from Fires._macros.macros import DRIVERS, TARGETS, MAX_HECTARES_100KM, LOGS_DIR, CONFIG
from Fires._plots.plot_utils import plot_dataset_map
from Fires._scalers.standard import StandardScaler
from Fires._utilities.logger import Logger as logger
from Fires._utilities.decorators import debug, export
from Fires._utilities.utils_general import check_backend


import toml
import munch            
from typing import Any

# define logger
_log = logger(log_dir=LOGS_DIR).get_logger("Inference Utilities")


@export
@debug(log=_log)
def get_prov_image(run_name):

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

def get_scaler_from_path(scaler_path):
    return joblib.load(scaler_path)

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
def compute_aggregated_data(data, other_data=None, operation="mean", verbose=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
	descaled_max = np.nanmax(data)

	if verbose:
		print(f" {operation.capitalize()} of data: {data.shape}")
		print(f" Max: {round(np.nanmax(data), 2)} \t Min: {round(np.nanmin(data), 2)}")
		print(f" Lats Max: {round(np.nanmax(descaled_on_lats), 2)} \t Lons Max: {round(np.nanmax(descaled_on_lons), 2)}")

	return data, descaled_on_lats, descaled_on_lons, descaled_max


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
		#print(f"Is DataArray - AVG: {avg_on_time.shape} STD: {std_on_time.shape}")
	else:
		avg_on_time = np.nanmean(data, axis=0)[0, ...]
		std_on_time = np.nanstd(data, axis=0)[0, ...]
		#print(f"NOT DataArray - AVG: {avg_on_time.shape} STD: {std_on_time.shape}")

	# Aggregate data
	avg_descaled, avg_on_lats, _, avg_max = compute_aggregated_data(data=avg_on_time)
	_, std_on_lats, _, _ = compute_aggregated_data(data=std_on_time)

	# Compute upper and lower boundaries
	upperbound, lowerbound = up_and_lower_bounds(avg_value=avg_on_lats, std_value=std_on_lats)
	
	# Plot data
	plot_dataset_map(
		avg_target_data=avg_descaled,
		scale_min=scale_min if scale_min else 0,
		scale_max=scale_max if scale_max else avg_max,
		lats=lats,
		lons=lons,
		title=f'{label} ({model_name.upper()})',
		cmap='nipy_spectral_r',
		plot_lat = False,
		avg_data_on_lats=avg_on_lats,
		lowerbound_data=lowerbound,
		upperbound_data=upperbound
	)
	
def process_and_plot_difference_map(
                          obs_data, 
                          pred_data, 
                          label,
                          lats,
                          lons,
                          scale_min: int=None,
                          scale_max: int=None):
	"""
	Process the data and generate plots with difference between 2 datasets.

	Parameters
	----------
	obs_data : xarray.DataArray or np.ndarray
		Ground truth data to process; can be an xarray.DataArray for real data or a numpy.ndarray for predictions.
	pred_data : xarray.DataArray or np.ndarray
		Predicted data to process; can be an xarray.DataArray for real data or a numpy.ndarray for predictions.
	label : str
		Label to use in the plot title.
	lats : np.ndarray
		Array of latitudes.
	lons : np.ndarray
		Array of longitudes.

	"""
	# Verify data type and compute mean and standard deviation along time axis
	if isinstance(obs_data, xr.DataArray):
		obs_data = obs_data.data
	else:
		obs_data = obs_data.squeeze(1)
	if isinstance(pred_data, xr.DataArray):
		pred_data = pred_data.data
	else:
		pred_data = pred_data.squeeze(1)

	obs_data = np.nan_to_num(obs_data)
	pred_data = np.nan_to_num(pred_data)

	# compute the difference between real and predicted data
	difference = pred_data - obs_data  # Shape: (time, lat, lon)
	avg_difference = np.mean(difference, axis=0)  # Sum across time dimension
	avg_diff_max = np.max(np.abs(avg_difference))

	plot_dataset_map(
		avg_target_data=avg_difference,
		scale_min=scale_min if scale_min else -avg_diff_max,
		scale_max=scale_max if scale_max else avg_diff_max,
        lats=lats,
		lons=lons,
		title=f'{label}',
		cmap="RdBu",
		plot_lat = False
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
                                scale_max: int=None,
                                sea_poles_mask: xr.DataArray=None):
	"""
	Process the data and generate plots for CMIP6 data.

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
    scale_min : int
        The minimum of the scale used for the plotting.
    scale_max: int
        The maximum of the scale used for the plotting.
    sea_poles_mask: xr.DataArray=None
        xr.DataArray object to mask the poles and sea in the final map.

	"""
	# TODO: Alot of redundant code here. CLEAN UP!
 
	# Verify data type and compute mean and standard deviation along time axis
	if isinstance(data, xr.DataArray):
		print("Data type xr.DataArray...")
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
			avg_on_time = avg_on_time.values
			std_on_time = yearly_aggregate.std(dim='time', skipna=True).values
		else:
            # No decadal aggregate in case you only 
			avg_on_time = yearly_aggregate.values.squeeze(0) 
			std_on_time = yearly_aggregate.std(dim='time', skipna=True).data
	else:
		avg_on_time = np.nanmean(data, axis=0)
		std_on_time = np.nanstd(data, axis=0)
		print(f"NOT DataArray - AVG: {avg_on_time.shape} STD: {std_on_time.shape}")

	# Aggregate data
	avg_descaled, avg_on_lats, _, _ = compute_aggregated_data(data=avg_on_time)
	_, std_on_lats, _, _ = compute_aggregated_data(data=std_on_time)

	if isinstance(sea_poles_mask,xr.DataArray):
		sea_poles_idxs = np.where(~(sea_poles_mask == 0))
		lat_lon_idx_pairs = list(zip(sea_poles_idxs[0], sea_poles_idxs[1]))
		for pair_x, pair_y in lat_lon_idx_pairs:
			avg_descaled[pair_x, pair_y] = 0
	# Compute upper and lower boundaries
	upperbound, lowerbound = up_and_lower_bounds(avg_value=avg_on_lats, std_value=std_on_lats)
    
	# Plot data
	plot_dataset_map(
		avg_target_data=avg_descaled,
        scale_max=scale_max,
        scale_min=scale_min,
		lats=lats,
		lons=lons,
		title=f'{label} ({model_name.upper()})',
		cmap='nipy_spectral_r',
		plot_lat = False,
        avg_data_on_lats=avg_on_lats,
		lowerbound_data=lowerbound,
		upperbound_data=upperbound,
	)


def aggregate_var(dataarray: xr.DataArray, method:str, dim:str='time'):
    eval_str = f"dataarray.{method}(dim='{dim}',skipna=True,keep_attrs=True)"
    output = eval(eval_str)
    return output


def _get_list_of_dates(year_range):
    n_prior_days = 8
    str_dates = [f'{year_range.value[0]}-01-08', f'{year_range.value[1]}-12-24']  # ✅ fixed here

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
        calendar_type = "noleap"
        numeric_time = date2num(py_date, units, calendar_type, has_year_zero=True)
        dates_range_cfttime.append(num2date(numeric_time, units, calendar_type, has_year_zero=True))
    
    return dates_range_cfttime


def _get_file_list_directly(scenario, climate_model, infer_config, year_range):
    
    dates_range_np = make_8day_windows(year_range)
    cmip6_var_filename = {}

    for var_key, var_value in infer_config.data.drivers.items():
        cmip6_var_filename[var_key] = []
        if var_value.type == "dynamic":
            cmip6_path = var_value.cmip6_path.replace("[scenario]", scenario.value)
            path_to_files = os.path.join(infer_config.config.base_dir, cmip6_path)
            list_of_files = [file for file in os.listdir(path_to_files) if file.endswith(".nc")]
            for file in list_of_files:
                date = file.split("_")[-1].split(".")[0].split("-")
                start_date = np.datetime64(f"{date[0][:4]}-{date[0][4:6]}-{date[0][6:]}")
                end_date = np.datetime64(f"{date[1][:4]}-{date[1][4:6]}-{date[1][6:]}")
                for date_range in dates_range_np:
                    if min(date_range) >= start_date and max(date_range) <= end_date:
                        full_path = os.path.join(path_to_files, file)
                        if full_path not in cmip6_var_filename[var_key]:
                            cmip6_var_filename[var_key].append(full_path)
        else:
            path_to_file = os.path.join(infer_config.config.base_dir, var_value.cmip6_path.replace("[scenario]", scenario.value))
            file = [f for f in os.listdir(path_to_file) if f.endswith(".nc")]
            cmip6_var_filename[var_key] = [os.path.join(path_to_file, file[0])]

    print("Loading the following CMIP6 data files from LOCAL:")
    for key, value in cmip6_var_filename.items():
        print(f"{key}: {value}")

    return cmip6_var_filename, dates_range_np



def make_8day_windows(year_range: tuple[int, int]) -> np.ndarray:
    start = np.datetime64(f"{year_range.value[0]}-01-01")
    end   = np.datetime64(f"{year_range.value[1]}-12-31")
    windows = []
    current = start
    one_day = np.timedelta64(1, "D")
    eight_days = np.timedelta64(8, "D")

    while current <= end:
        window_end = min(current + eight_days - one_day, end)
        windows.append((current, window_end))
        current = current + eight_days

    return np.array(windows, dtype="datetime64[ns]")


def _get_cmip6_files_rucio(scope,
                          rse,
                          dataset,
                          scenario,
                          climate_model,
                          year_range,
                          infer_config):
    
    """
    Returns:
      - cmip6_var_filename: dict[var_name, list[file_paths]]
      - np_dates:       list of numpy.datetime64 stamps (one per 8-day window)
    """
    
    windows = make_8day_windows(year_range)     

    from rucio.client.client import Client
    rucio = Client()
    cmip6_var_filename: dict[str, list[str]] = {}

    for var, cfg in infer_config.data.drivers.items():
        cmip6_var_filename[var] = []
        replicas = rucio.list_replicas(
            dids=[{"scope": scope, "name": dataset}],
            schemes=["file"],
            rse_expression=rse
        )

        # find replicas at your chosen RSE
        paths = []
        for replica in replicas:
            if (var in replica["name"] and climate_model.value in replica["name"] and scenario.value in replica["name"]) and rse in replica["rses"]:
                lfilepath=replica["rses"][rse][0]
                filepath = lfilepath.replace('file://localhost', '')
                paths.append(filepath)
        
        assert paths, "No files were found for the query to RUCIO"
        
        # TODO: Finish this code do it in a nicer way
        
        if cfg.type == "dynamic":
            # only keep those files that cover *any* of our windows
            for p in sorted(paths):
                fn = os.path.basename(p)
                datestr = fn.rsplit("_", 1)[-1].removesuffix(".nc")
                start_s, end_s = datestr.split("-")
                start = np.datetime64(f"{start_s[:4]}-{start_s[4:6]}-{start_s[6:]}")
                end   = np.datetime64(f"{end_s[:4]}-{end_s[4:6]}-{end_s[6:]}")
                # if this file fully covers at least one 8-day window, keep it
                if any((w[0] >= start and w[1] <= end) for w in windows):
                    cmip6_var_filename[var].append(p)
        else:
            # static driver → only need the first match
            if paths:
                cmip6_var_filename[var] = [paths[0]]
            else:
                raise FileNotFoundError(
                    f"No static file for '{var}' (pattern {pattern})"
                )

    print("Loading the following CMIP6 data files from RUCIO:")
    for var, flist in cmip6_var_filename.items():
        print(f"  {var}: {flist}")

    # **Key change**: return np_dates (1-D list of window ends), not the (N,2) windows
    return cmip6_var_filename, windows


def _read_and_aggregate_cmip6_data(seafire_ds, scenario, climate_model, infer_config, year_range):

    if CONFIG.rucio.rse:
        try:
            #from rucio.client.uploadclient import UploadClient
            from rucio.client.client import Client
            rucio = Client()
        except:
            raise Exception("Rucio client could not be found. Make sure Rucio library is installed.")
        cmip6_var_filename, dates_range_np = _get_cmip6_files_rucio(scope=CONFIG.rucio.scope,
                                                                    rse=CONFIG.rucio.rse,
                                                                    dataset=CONFIG.rucio.dataset,
                                                                    scenario=scenario,
                                                                    climate_model=climate_model,
                                                                    year_range=year_range,
                                                                    infer_config=infer_config)
    else:
        cmip6_var_filename, dates_range_np = _get_file_list_directly(scenario=scenario,
                                                                     climate_model=climate_model,
                                                                     infer_config=infer_config,
                                                                     year_range=year_range)

    if "lon" in seafire_ds.dims:
        seafire_ds = seafire_ds.rename({"lon":"longitude"})
    if "lat" in seafire_ds.dims:
        seafire_ds = seafire_ds.rename({"lat":"latitude"})

    if infer_config.config.temp_dir:
        from cdo import Cdo
        cdo_obj = Cdo()
        grid_spec = infer_config.config.temp_dir + "/grid.grid"

    dates_range_cftime = _get_cft_times_list(year_range=year_range)
    var_ds_list = []
    for var_name, var_cfg in infer_config.data.drivers.items():
        print(f"Reading variable {var_name} and aggregating with method {infer_config.data.drivers[var_name].aggregation}...")
        cmip6_path = var_cfg.cmip6_path.replace("[scenario]", scenario.value) 
        full_dir = os.path.join(infer_config.config.base_dir, cmip6_path)
        files = sorted(cmip6_var_filename[var_name])
        assert files, f"There were no files found for {var_name}. Please double check configuration."
        agg_method = var_cfg.aggregation.lower()
        if agg_method != "none":
            # Open all files along time
            ds = xr.open_mfdataset(
                files,
                concat_dim="time",
                combine="nested"
            )[var_name]

            slices = []
            for idx, win in enumerate(dates_range_cftime):
                chunk = ds.sel(time=slice(win[0], win[1]))
                if chunk.time.size == 0:
                    continue
                if var_name == "pr":
                    chunk = chunk * 3600 * 24

                agg = aggregate_var(chunk, method=agg_method, dim="time")
                stamp = np.datetime64(dates_range_np[idx][1])
                
                # Build a 1-element DataArray
                agg_da = xr.DataArray(
                    data=agg.values[np.newaxis, ...],
                    dims=("time",) + agg.dims,
                    coords={"time": [stamp], **{d: agg.coords[d] for d in agg.dims}},
                    name=var_name,
                )
                slices.append(agg_da)

            ds_var = xr.concat(slices, dim="time")

        else:
            # Static file
            # Changing the 'unit' for the land sea mask 
            ds_var = xr.open_dataset(files[0])[var_name] / 100.0

        # Regrid
        if infer_config.config.temp_dir:
            tmp_input = infer_config.config.temp_dir + var_name + "_tmp_in.nc"
            output_path = infer_config.config.temp_dir + var_name + "_tmp_out.nc"
            ds_var.to_netcdf(tmp_input)
            getattr(cdo_obj, "remapcon")(
                grid_spec,
                input=tmp_input,
                output=output_path
            )
            ds_var = xr.open_dataset(output_path)

        # Rename
        ds_var = (
            ds_var
            .assign_coords(lon=((ds_var.lon + 180) % 360) - 180)
            .sortby("lon").sortby("lat", False)
            .rename(lon="longitude", lat="latitude")
        )
        if not infer_config.config.temp_dir:
            ds_var = ds_var.interp_like(seafire_ds[["longitude", "latitude"]])

        var_ds_list.append(ds_var)

    assert var_ds_list, "No local variables found or processed."
    merged_data = xr.merge(var_ds_list)
    if "plev" in merged_data.dims:
        merged_data = merged_data.isel(plev=0).drop_vars("plev", errors="ignore")

    # Sort and extract array + time vector
    merged_data = merged_data.sortby("time")
    time_vec = merged_data.time.values
   
    return merged_data, time_vec

def _make_xr_ds_of_prediction(np_prediction: np.ndarray,
                              org_ds: xr.Dataset,
                              coords = None,
                              attrs = None,
                              var_name = "global_burned_areas"):
    if coords is None:
        latitude = "latitude"
        longitude = "longitude"
        xr_coords = {
            "time": org_ds.time,
            latitude: org_ds.latitude,
            longitude: org_ds.longitude,
        }
    else:
        latitude = coords[0]
        longitude = coords[1]
        xr_coords = {
            "time": org_ds.time,
            latitude: org_ds.lat,
            longitude: org_ds.lon,
        }
    if np_prediction.ndim < 3:
        np_prediction = np.expand_dims(a=np_prediction, axis=0)
    xr_dataset = xr.Dataset(
        data_vars={
            var_name: (("time", latitude, longitude), np_prediction)
        },
        coords=xr_coords
    ).sortby("time")

    if attrs:
        xr_dataset.attrs.update(attrs)
    
    return xr_dataset
    
def do_inference_from_ds(dataset: xr.Dataset,
                         model,
                         scaler,
                         var_name = "global_burned_areas",
                         move_latlon = True):

    if "plev" in dataset.dims:
        dataset = dataset.isel(plev=0).drop_vars("plev", errors="ignore")

    if "longitude" in dataset.dims:
        dataset = dataset.rename({"longitude":"lon"})
    if "latitude" in dataset.dims:
        dataset = dataset.rename({"latitude":"lat"})

    if move_latlon:
        dataset = dataset.assign_coords({"lon": (((dataset.lon + 180) % 360) - 180)}).sortby("lon").sortby("lat", False)

    dataset = dataset[['lai', 'lst_day', 'rel_hum', 't2m_min', 'pr', 'lsm']]

    X = torch.tensor(dataset.to_array().transpose("time", "variable", "lat", "lon").values)
    X = scaler.transform(X).float()
    X = torch.nan_to_num(X, nan=0)

    preds = []
    model.eval()
    with torch.inference_mode():
        for t in range(X.shape[0]):
            out = model(X[t : t + 1].to(check_backend()))
            preds.append(out.cpu().numpy())
    predictions = np.vstack(preds).squeeze()

    ds_attrs={
            "Source": "CMCC Foundation",
            "Processed_by": "ML4Fires",
        }
    
    ds_pred = _make_xr_ds_of_prediction(np_prediction=predictions, 
                                        org_ds=dataset,
                                        coords=["lat", "lon"],
                                        attrs=ds_attrs,
                                        var_name=var_name)
    
    return ds_pred


def get_cmip6_inference(
    seafire_ds,
    run_name,
    scenario,
    climate_model,
    year_range,
    model,
    infer_config
):
 
    print(f"📘 Running inference for scenario: {scenario.value}, years: {year_range.value[0]}–{year_range.value[1]}")

    xr_ds, _ = _read_and_aggregate_cmip6_data(
        seafire_ds=seafire_ds,
        scenario=scenario,
        climate_model=climate_model,
        infer_config=infer_config,
        year_range=year_range
    )
    
    if "longitude" in xr_ds.dims and "latitude" in xr_ds.dims:
        ds_array = xr_ds.to_array().transpose("time", "variable", "latitude", "longitude").values
    else:
        ds_array = xr_ds.to_array().transpose("time", "variable", "lat", "lon").values    
    
    print("🧮 Input shape:", ds_array.shape)

    # ── Run the model ──
    scaler = get_scaler(run_name=run_name)
    
    X = torch.tensor(ds_array)
    X = scaler.transform(X).float()
    X = torch.nan_to_num(X, nan=0)

    print("⚙️  Running model inference...")

    preds = []
    model.eval()
    with torch.inference_mode():
        for t in range(X.shape[0]):
            out = model(X[t : t + 1].to(check_backend()))
            preds.append(out.cpu().numpy())
    predictions = np.vstack(preds).squeeze()

    print("📦 Building prediction dataset...")
    
    attrs={
            "Details": f"Inference for {scenario.value}, {year_range.value[0]}–{year_range.value[1]}",
            "Source": "CMCC Foundation",
            "Processed_by": "ML4Fires",
        }
    
    ds_pred = _make_xr_ds_of_prediction(np_prediction=predictions, org_ds=xr_ds, attrs=attrs)

    return ds_pred


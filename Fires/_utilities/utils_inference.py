import numpy as np
import xarray as xr
import os
import joblib
import torch
import pydot
import datetime
import re
from cftime import num2date, date2num
import toml
from Fires._datasets.torch_dataset import FireDataset
from Fires._macros.macros import DRIVERS, TARGETS, MAX_HECTARES_100KM, LOGS_DIR, CONFIG
from Fires._plots.plot_utils import plot_dataset_map
from Fires._scalers.standard import StandardScaler
from Fires._utilities.logger import Logger as logger
from Fires._utilities.decorators import debug, export
import munch

#os.environ['RUCIO_CONFIG'] = '/ceph/hpc/home/ciangottinid/ML4Fires/rucio.cfg'

import toml
import munch            
from types import SimpleNamespace
from typing import Any
from types import SimpleNamespace

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
		scale_min=scale_min,
		scale_max=scale_max,
		lats=lats,
		lons=lons,
		title=f'{label} ({model_name.upper()})',
		cmap='nipy_spectral_r'
	)

@export
@debug(log=_log)
def process_and_plot_cmip6infer(data: xr.Dataset,
                                temporal_aggregate_scheme: list | str,
                                label,
                                lats,
                                lons,
                                model_name,
                                scale_min: int = None,
                                scale_max: int = None,
                                lat_min: float = -60):
    """
    Process the data and generate plots, excluding latitudes below lat_min.

    Parameters
    ----------
    data : xarray.DataArray or np.ndarray
        Data to process.
    temporal_aggregate_scheme : list | str
        Aggregation strategy.
    label : str
        Label for the plot title.
    lats : np.ndarray
        Latitude array.
    lons : np.ndarray
        Longitude array.
    model_name : str
        Name of the model for the title.
    scale_min : int
        Plot color scale minimum.
    scale_max : int
        Plot color scale maximum.
    lat_min : float
        Minimum latitude to include (default -60 to remove South Pole).
    """

  
    # Temporal aggregation
    if isinstance(data, xr.DataArray):
        if temporal_aggregate_scheme == "mean":
            avg_on_time = data.mean(dim='time', skipna=True).data
            std_on_time = data.std(dim='time', skipna=True).data
            print(f"Is DataArray - AVG: {avg_on_time.shape} STD: {std_on_time.shape}")
        else:
            assert isinstance(temporal_aggregate_scheme, dict), "For multi-scale aggregate, a dictionary with keys 'monthly', 'yearly', and 'decadal' is required."
            assert len(temporal_aggregate_scheme) > 0, "Provide at least one aggregate method."

            years_in_ds = np.unique(data.time.dt.year)

            # Monthly aggregate
            monthly_aggregate = eval(
                f"data.resample(time='{temporal_aggregate_scheme['monthly'][1]}', skipna=True).{temporal_aggregate_scheme['monthly'][0]}()"
            )

            # Yearly aggregate
            yearly_aggregate = eval(
                f"monthly_aggregate.resample(time='1Y', skipna=True).{temporal_aggregate_scheme['yearly']}()"
            )

            if len(years_in_ds) > 1:
                avg_on_time = eval(
                    f"yearly_aggregate.{temporal_aggregate_scheme['decadal']}(dim='time', skipna=True)"
                )
                std_on_time = yearly_aggregate.std(dim='time', skipna=True).data
            else:
                avg_on_time = yearly_aggregate
                std_on_time = yearly_aggregate.std(dim='time', skipna=True).data
    else:
        if temporal_aggregate_scheme == ["mean", "mean", "mean"]:
            avg_on_time = np.nanmean(data, axis=0)
            std_on_time = np.nanstd(data, axis=0)
            print(f"NOT DataArray - AVG: {avg_on_time.shape} STD: {std_on_time.shape}")
        else:
            raise Exception("Different averaging works only if predictions are in xr.DataArray format.")

   
    # Spatial aggregation
    avg_descaled, avg_on_lats, _ = compute_aggregated_data(data=avg_on_time)
    _, std_on_lats, _ = compute_aggregated_data(data=std_on_time)

    # Compute upper and lower bounds
    upperbound, lowerbound = up_and_lower_bounds(avg_value=avg_on_lats, std_value=std_on_lats)

    
    # Latitude filtering
    lat_mask = lats >= lat_min
    #print(f"Filtering latitudes below {lat_min} degrees.")

    lats = lats[lat_mask]

    # Helper function to apply the latitude mask safely
    def apply_lat_mask(arr):
        if arr.ndim == 2:
            return arr[lat_mask, :]
        elif arr.ndim == 1:
            return arr[lat_mask]
        else:
            raise ValueError(f"Unexpected array dimension: {arr.ndim}")

    avg_on_lats = apply_lat_mask(avg_on_lats)
    upperbound = apply_lat_mask(upperbound)
    lowerbound = apply_lat_mask(lowerbound)
    avg_descaled = apply_lat_mask(avg_descaled)

    
    # Plot
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
        # calendar_type = "standard" if calendar.isleap(int(str(single_date_range[0])[0:4])) else "noleap"
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
    
    # 1) build the 8-day windows
    windows = make_8day_windows(year_range)      # array of shape (N,2)
    # 2) our “time” stamps are simply the window-end dates:
    np_dates = [w[1] for w in windows]          # list of length N

    # 3) now, exactly as before, discover your file paths via Rucio…
    from rucio.client.client import Client
    rucio = Client()
    cmip6_var_filename: dict[str, list[str]] = {}

    for var, cfg in infer_config.data.drivers.items():
        cmip6_var_filename[var] = []
        replicas = rucio.list_replicas(
            dids=[{"scope": scope, "name": dataset}],
            schemes=["https"],
            rse_expression=rse
        )

        # find replicas at your chosen RSE
        paths = []
        for replica in replicas:
            if (var in replica["name"] and climate_model in replica["name"] and scenario.value in replica["name"]) and rse in replica["rses"]:
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
    return cmip6_var_filename, np_dates


# def _get_file_list_rucio(
#     scenario: str,
#     year_range: tuple[int, int],
#     infer_config
# ):

#     base_dir = infer_config.config.base_dir
#     drivers_cfg = infer_config.data.drivers

#     return _get_cmip6_files_rucio(scope=CONFIG.rucio.scope,
#                                  rse=CONFIG.rucio.rse,
#                                  model_name=CONFIG.rucio.model,
#                                  scenario=scenario,
#                                  year_range=year_range,
#                                  infer_config=infer_config)


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

        # Regrid & rename
        ds_var = (
            ds_var
            .assign_coords(lon=((ds_var.lon + 180) % 360) - 180)
            .sortby("lon")
            .rename(lon="longitude", lat="latitude")
            .interp_like(seafire_ds[["longitude", "latitude"]])
        )

        var_ds_list.append(ds_var)

    assert var_ds_list, "No local variables found or processed."
    merged = xr.merge(var_ds_list)
    if "plev" in merged.dims:
        merged = merged.isel(plev=0)

    # Sort and extract array + time vector
    merged = merged.sortby("time")
    time_vec = merged.time.values
    data = merged.to_array().transpose("time", "variable", "latitude", "longitude").values
   
    return data, time_vec



# def _read_and_aggregate_rucio_cmip6(files_dict, local_config, seafire_ds):
    
#     def extract_years_from_filenames(filenames):
#         years = []
#         for fn in filenames:
#             try:
#                 part = fn.rsplit("_", 1)[-1].replace(".nc", "")
#                 start, end = part.split("-")
#                 years += [int(start[:4]), int(end[:4])]
#             except:
#                 continue
#         if not years:
#             raise ValueError("Cannot parse years from filenames.")
#         return min(years), max(years)

#     # 1) Infer date range
#     all_files = [f for flist in files_dict.values() for f in flist]
#     min_year, max_year = extract_years_from_filenames(all_files)

#     # 2) Build windows
#     windows_np = make_8day_windows((min_year, max_year))
#     windows_cftime = _get_cft_times_list((min_year, max_year))

#     drivers_cfg = local_config.data.drivers
#     var_ds_list = []

#     for var_name, file_list in files_dict.items():
#         if not file_list:
#             continue

#         agg_method = drivers_cfg[var_name].aggregation.lower()

#         if agg_method != "none":
#             ds = xr.open_mfdataset(file_list, combine="by_coords")[var_name]
#             slices = []
#             for idx, win in enumerate(windows_cftime):
#                 chunk = ds.sel(time=slice(win[0], win[1]))
#                 if chunk.time.size == 0:
#                     continue
#                 if var_name == "pr":
#                     chunk *= 3600 * 24

#                 agg = aggregate_var(chunk, method=agg_method, dim="time")
#                 stamp = np.datetime64(windows_np[idx][1])

#                 # build 1-element time‐indexed DataArray
#                 agg_da = xr.DataArray(
#                     data=agg.values[np.newaxis, ...],
#                     dims=("time",) + agg.dims,
#                     coords={"time": [stamp], **{d: agg.coords[d] for d in agg.dims}},
#                     name=var_name
#                 )
#                 slices.append(agg_da)

#             if not slices:
#                 continue

#             ds_var = xr.concat(slices, dim="time")
#             # remove any duplicate times
#             _, uniq = np.unique(ds_var.time.values, return_index=True)
#             ds_var = ds_var.isel(time=uniq)

#         else:
#             ds_var = xr.open_dataset(file_list[0])[var_name] / 100.0

#         # regrid & rename
#         ds_var = (
#             ds_var
#             .assign_coords(lon=((ds_var.lon + 180) % 360) - 180)
#             .sortby("lon")
#             .rename(lon="longitude", lat="latitude")
#             .interp_like(seafire_ds[["longitude", "latitude"]])
#         )
#         var_ds_list.append(ds_var)

#     if not var_ds_list:
#         raise ValueError("No valid variables to process.")

#     merged = xr.merge(var_ds_list)
#     if "plev" in merged.dims:
#         merged = merged.isel(plev=0)

#     # **Here** we extract the time vector that actually matches merged.time
#     merged = merged.sortby("time")
#     time_vec = merged.time.values  # 1D array of length nt

#     data = merged.to_array().transpose("time", "variable", "latitude", "longitude").values
#     return data, time_vec


def get_cmip6_inference(
    seafire_ds,
    run_name,
    scenario,
    climate_model,
    year_range,
    model,
    infer_config
):
    # Load configs
    # server_cfg = munch.munchify(toml.load(server_config_path))
    # local_cfg = munch.munchify(toml.load(local_config_path))
    # rse = CONFIG.rucio.get("rse", "")

    print(f"📘 Running inference for scenario: {scenario.value}, years: {year_range.value[0]}–{year_range.value[1]}")

    ds_array, time_vec = _read_and_aggregate_cmip6_data(
        seafire_ds=seafire_ds,
        scenario=scenario,
        climate_model=climate_model,
        infer_config=infer_config,
        year_range=year_range
    )

    print("🧮 Input shape:", ds_array.shape)

    # ── Run the model ──
    scaler = get_scaler(run_name=run_name)
    X = torch.tensor(ds_array)
    X = scaler.transform(X).float()
    X = torch.nan_to_num(X, nan=0)

    print("⚙️  Running model inference...")
    preds = []
    with torch.no_grad():
        for t in range(X.shape[0]):
            out = model(X[t : t + 1].to("cuda:0"))
            preds.append(out.cpu().numpy())
    predictions = np.vstack(preds).squeeze()

    print("📦 Building prediction dataset...")
    ds_pred = xr.Dataset(
        data_vars={
            "global_burned_areas": (("time", "latitude", "longitude"), predictions)
        },
        coords={
            "time": ("time", time_vec),
            "latitude": seafire_ds.latitude,
            "longitude": seafire_ds.longitude,
        },
        attrs={
            "Details": f"Inference for {scenario.value}, {year_range.value[0]}–{year_range.value[1]}",
            "Source": "CMCC Foundation",
            "Processed_by": "ML4Fires",
        },
    ).sortby("time")

    return ds_pred


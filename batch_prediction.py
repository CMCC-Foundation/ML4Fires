import numpy as np
import xarray as xr
import toml
import munch
from tqdm import tqdm
import torch
import datetime

import warnings
warnings.filterwarnings("ignore")

from Fires._utilities.utils_mlflow import load_model_from_mlflow
from Fires._utilities.utils_inference import get_cmip6_inference, aggregate_var

def get_prediction_for_data(dataset: xr.Dataset):
    temporal_aggregate_scheme = {}
    agg_inference = aggregate_cmip6infer(dataset=dataset,
                                         temporal_aggregate_scheme=temporal_aggregate_scheme,
                                         label=label,
                                         lats=dataset.lat.values,
                                         lons=dataset.lon.values,
                                         scale_min=0,
                                         scale_max=8000,
                                         sea_poles_mask=None)
                                         
    # Questions:
    # Config files to load here?
    # seafire dataset to load here? - May be this can be done in the data that Cosimo creates. 
    # Model to load in this part?
    # Configuration for processing of the data?
    # Sea pole mask?
    pass


def aggregate_cmip6infer(data: xr.Dataset,
                                temporal_aggregate_scheme,
                                lats,
                                lons,
                                model_name,
                                scale_min: int=None,
                                scale_max: int=None,
                                sea_poles_mask: xr.DataArray=None):	
	
 # Verify data type and compute mean and standard deviation along time axis
    print("Data type xr.DataArray...")
    # Check that parameter temporal_aggregate_scheme is not an str and is not an empty list
    assert isinstance(temporal_aggregate_scheme, dict), "For multi-scale aggregate, a dictionary with keys 'monthly,  'year' and 'decadal' is required."
    assert len(temporal_aggregate_scheme) > 0, "For multi-scale aggregate, more than one aggregate methods should be provided."
    years_in_ds = np.unique(data.time.dt.year)

    # Monthly aggregate
    # Get aggreagated on monthly scale for the years in the dataset.
    # Default date is in start of the month.
    monthly_aggregate = eval(f"data.resample(time='{temporal_aggregate_scheme['monthly'][1]}',skipna=True).{temporal_aggregate_scheme['monthly'][0]}()")
    # Aggregate over the year (s)
    # Aggregate on the year (s)
    yearly_aggregate = eval(f"monthly_aggregate.resample(time='1Y',skipna=True).{temporal_aggregate_scheme['yearly']}()")

    if len(years_in_ds) > 1: # check if there are more than one year in the dataset - decadal scale prediction
        # If there are more than on year, then there should be an aggregate method in the list which is not none or empty
        # Do decadal aggregate
        avg_on_time = eval(f"yearly_aggregate.{temporal_aggregate_scheme['decadal']}(dim='time',skipna=True)")
        avg_on_time = avg_on_time.values
        std_on_time = yearly_aggregate.std(dim='time', skipna=True).values
    else:
        # No decadal aggregate in case you only 
        avg_on_time = yearly_aggregate 
        std_on_time = yearly_aggregate.std(dim='time', skipna=True).data

    if isinstance(sea_poles_mask,xr.DataArray):
        sea_poles_idxs = np.where(~(sea_poles_mask == 0))
        lat_lon_idx_pairs = list(zip(sea_poles_idxs[0], sea_poles_idxs[1]))
        for pair_x, pair_y in lat_lon_idx_pairs:
            avg_on_time[pair_x, pair_y] = 0

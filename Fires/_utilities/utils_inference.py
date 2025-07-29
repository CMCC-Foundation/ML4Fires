import numpy as np
import xarray as xr
import os
import joblib
import torch
import pydot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from skimage.metrics import structural_similarity as ssim
import matplotlib.colors as mcolors




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


@export
@debug(log=_log)
def create_data_loader(data_path, run_name):
    # define scaler
    local_path = os.path.join(os.getcwd(), 'MLFLOW', f"{run_name}/scaler/scaler.dump")
    x_scaler = joblib.load(local_path)

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
def process_and_plot_data_all(data, label, lats, lons, model_name):
	"""
	Process the data and generate plots with the average of 20 years.

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

	if isinstance(data, xr.DataArray):
		avg_on_time = data.mean(dim='time', skipna=True).data
		std_on_time = data.std(dim='time', skipna=True).data
	else:
		avg_on_time = np.nanmean(data, axis=0)[0, ...]
		std_on_time = np.nanstd(data, axis=0)[0, ...]

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
def process_and_plot_data_all_sum(data, label, lats, lons, model_name):
    """
    Process the data and generate plots (sum of all the 20 years).

    Parameters
    ----------
    data : xarray.DataArray or np.ndarray
        Data to process.
    label : str
        Plot title label.
    lats : np.ndarray
        Latitudes.
    lons : np.ndarray
        Longitudes.
    model_name : str
        Model name for the plot title.
    """

    if isinstance(data, xr.DataArray):
        sum_on_time = data.sum(dim='time', skipna=True).data
        std_on_time = data.std(dim='time', skipna=True).data
    else:
        sum_on_time = np.nansum(data, axis=0)[0, ...]
        std_on_time = np.nanstd(data, axis=0)[0, ...]

    # Aggregate data (sum instead of mean)
    sum_descaled, sum_on_lats, _ = compute_aggregated_data(data=sum_on_time)
    _, std_on_lats, _ = compute_aggregated_data(data=std_on_time)

    # Compute upper and lower boundaries
    upperbound, lowerbound = up_and_lower_bounds(avg_value=sum_on_lats, std_value=std_on_lats)

    # Plot
    plot_dataset_map(
        avg_target_data=sum_descaled,
        avg_data_on_lats=sum_on_lats,
        lowerbound_data=lowerbound,
        upperbound_data=upperbound,
        lats=lats,
        lons=lons,
        title=f'{label} ({model_name.upper()})',
        cmap='nipy_spectral_r'
    )


    
    
# input_tensor and preds_tensor are already:
# - dtype=torch.float32
# - shaped (time, lat, lon)
# - NaNs already masked out using nan_mask
# If you skipped singleton dimension on preds_tensor, both should now be shape (T, H, W)

def compute_aggregated_mape_smape(input_tensor, preds_tensor):
    """
    Compute aggregated MAPE and sMAPE after summing over time (per-pixel totals).
    Args:
        input_tensor: torch.Tensor of shape (T, lat, lon)
        preds_tensor: torch.Tensor of shape (T, lat, lon)
    Returns:
        mape, smape: float
    """
    eps = 1e-6  # small constant to avoid division by zero

    # Aggregate predictions and inputs over time (axis=0)
    agg_input = input_tensor.sum(dim=0)   # shape: (lat, lon)
    agg_preds = preds_tensor.sum(dim=0)   # shape: (lat, lon)

    # Compute Mean Absolute Percentage Error (MAPE)
    mape = torch.mean(torch.abs((agg_input - agg_preds) / (agg_input + eps))) * 100

    # Compute Symmetric MAPE (sMAPE)
    smape = 100 * torch.mean(
        2 * torch.abs(agg_preds - agg_input) / (torch.abs(agg_input) + torch.abs(agg_preds) + eps)
    )

    return mape.item(), smape.item()




def plot_burned_area_difference_map(input_data, preds_array, save_path: str = None):
    """
    Plots (and optionally saves) the global burned‐area difference map.

    Args:
        input_data: xarray Dataset with .latitude, .longitude, and "fcci_ba"
        preds_array: 3D numpy/torch array (time, lat, lon) of predictions
        save_path:    if provided, the full filepath to save the PNG
    """
    # Extract latitude and longitude
    lats = input_data.latitude.values
    lons = input_data.longitude.values

    # Get actual and predicted burned area data
    actual = input_data["fcci_ba"].values
    predicted = preds_array.squeeze(1)  # make sure shape is (time, lat, lon)

    # Replace NaNs with 0 for clean visualization
    actual = np.nan_to_num(actual)
    predicted = np.nan_to_num(predicted)

    # Compute the difference over time
    diff = predicted - actual            # shape: (time, lat, lon)
    diff_sum = np.mean(diff, axis=0)     # mean across time → (lat, lon)

    # Set up the map
    fig = plt.figure(figsize=(15, 7))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    # Add features
    ax.coastlines()  
    ax.add_feature(cfeature.LAND, alpha=0.3)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Plot the difference map
    vmax = np.max(np.abs(diff_sum))
    mesh = ax.pcolormesh(
        lons, lats, diff_sum,
        cmap="RdBu", 
        transform=ccrs.PlateCarree(),
        shading="auto",
        vmin=-vmax, vmax=vmax
    )

    # Add title & colorbar
    plt.title("Global Burned Area Difference (Predicted - Actual)", fontsize=14)
    cbar = plt.colorbar(mesh, orientation="vertical", pad=0.02,
                        aspect=30, shrink=0.8)
    cbar.set_label("Difference in Burned Area")

    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    
    

def compute_values(input_data, preds_array):
    # Extract only fcci_ba for comparison
    input_tensor = torch.tensor(input_data["fcci_ba"].values, dtype=torch.float32)
    preds_tensor = torch.tensor(preds_array, dtype=torch.float32)

    # Ensure preds_tensor has the correct shape by squeezing singleton dimension
    preds_tensor = preds_tensor.squeeze(1)  # Removes the extra dimension

    # Create a mask for non-NaN values in fcci_ba
    nan_mask = ~torch.isnan(input_tensor)  # True for valid values, False for NaNs

    # Apply the mask to remove NaN locations from both input and predictions
    input_tensor = input_tensor[nan_mask]
    preds_tensor = preds_tensor[nan_mask]

    # Ensure shape consistency after masking
    if input_tensor.shape != preds_tensor.shape:
        raise ValueError(f"Shape mismatch after masking: input {input_tensor.shape} vs preds {preds_tensor.shape}")

    # Apply Min-Max Normalization
    #input_tensor = normalize_data(input_tensor)
    #preds_tensor = normalize_data(preds_tensor)

    # Compute Metrics
   
    mae_value = compute_mae(input_tensor, preds_tensor)    
    ssim_value = compute_ssim(input_tensor, preds_tensor)

    return {
       
        "MAE": mae_value,
        "SSIM": ssim_value
        
    }    
    
def compute_mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()


def compute_ssim(input_tensor, preds_tensor):
    return ssim(input_tensor.numpy(), preds_tensor.numpy(), data_range=preds_tensor.max().item() - preds_tensor.min().item())




def plot_accumulated_weighted_map(data_path, mask_path, save_path: str = None):
    """
    Loads a monthly 3D field from Zarr, applies basis‐region weights from mask.nc,
    accumulates to 2D, masks out ocean & class 0, then plots with a rainbow colormap.
    If save_path is provided, saves the figure to that file.

    Parameters
    ----------
    data_path : str
        Path to your Zarr dataset (3D: time × lat × lon).
    mask_path : str
        Path to mask.nc containing `basis_regions` (cl × lat × lon).
    save_path : str, optional
        If given, the full filepath to save the PNG (or other supported format).
    """
    # ─── Load / Recompute your 2D result ─────────────────────────────────────────
    df_mask = xr.open_dataset(mask_path)
    df1     = xr.open_zarr(data_path)
    if isinstance(df1, xr.Dataset):
        df1 = df1[list(df1.data_vars)[0]]

    # accumulate weighted sums
    k = xr.zeros_like(df1)
    for c in df_mask.cl.values:
        msl = df_mask["basis_regions"].isel(cl=c)
        a1, m1 = xr.align(df1, msl, join="inner")
        w = (a1 * m1).sum().compute().item()
        k += w * m1
    k = k.compute()
    k2d = k.isel(time=0)

    # ─── Build masks ──────────────────────────────────────────────────────────────
    land_mask = (df_mask["basis_regions"].sum(dim="cl") > 0)
    _, land_mask = xr.align(k2d, land_mask, join="inner")
    land_mask_arr = land_mask.values

    cls0 = df_mask["basis_regions"].isel(cl=0) > 0
    _, cls0 = xr.align(k2d, cls0, join="inner")
    cls0_arr = cls0.values

    # ─── Prepare the 2D data array ────────────────────────────────────────────────
    data2d = k2d.values.astype(float)
    vmin = 0.0
    vmax = float(np.nanmax(data2d[land_mask_arr]))

    data2d_masked = np.full_like(data2d, np.nan, dtype=float)
    data2d_masked[land_mask_arr] = data2d[land_mask_arr]
    data2d_masked[cls0_arr]     = np.nan

    # ─── Set up rainbow colormap & linear norm ────────────────────────────────────
    cmap = plt.cm.rainbow.copy()
    cmap.set_bad("white")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ─── Plot on a world map ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 6))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    lon2d, lat2d = np.meshgrid(k2d.longitude, k2d.latitude)
    im = ax.pcolormesh(
        lon2d, lat2d, data2d_masked,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto"
    )

    ticks = np.linspace(vmin, vmax, num=6)
    cbar = plt.colorbar(
        im, ax=ax,
        orientation="vertical",
        shrink=0.6, pad=0.02,
        ticks=ticks
    )
    cbar.set_label("Accumulated weighted value (linear scale)")

    ax.set_title("Accumulated Result – rainbow colormap,\nlinear scale, class 0/ocean white")
    plt.tight_layout()

    # ─── Save if requested ────────────────────────────────────────────────────────
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


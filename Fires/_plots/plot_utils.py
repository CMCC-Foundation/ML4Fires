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

import inspect
import numpy as np
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.markers as markers
from mpl_toolkits.axes_grid1 import make_axes_locatable    

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)

from Fires._macros.macros import LOGS_DIR
from Fires._utilities.logger import Logger as logger
from Fires._utilities.decorators import debug, export

# define logger
_log = logger(log_dir=LOGS_DIR).get_logger("Plots")

# define projection
datacrs = ccrs.PlateCarree()

# define map extent
extent_args=dict(extents = [-180, 180, -90, 90], crs=datacrs)

# define latitudes and longitudes array that must be used for axes
latitudes = np.arange(-60, 90, 30)
longitudes = np.arange(-160, 180, 80)

@export
@debug(log=_log)
def draw_features(ax:Any):
	"""
	This function adds several geographical features to the map using Cartopy features:

    * Political borders: outlines country borders with a solid black line style (':') and a linewidth of 0.5.
    * Oceans: outlines the ocean regions with a solid black line style ('-') and a linewidth of 0.8.
    * Lakes: outlines lakes with a solid black line style ('-') and a linewidth of 0.8.
    * Rivers: outlines rivers with a solid black line style ('-') and a linewidth of 0.8.
    * Coastlines: adds high-resolution coastlines (50 meters) to the map with a higher zorder (3) for better visibility.

    **Note:** 
        * Land is not explicitly added in this function. 
        * Adding a background image using `ax.stock_img()` is not implemented. 

	Parameters
	----------
	ax : Any
		The matplotlib axis object to add the features to.

	Returns
	-------
	ax : Any
		The modified matplotlib axis object.

	
    
	"""
	# define function name
	fn_name = inspect.currentframe().f_code.co_name

	# political borders
	ax.add_feature(cfeature.BORDERS, linestyle=':',linewidth=0.5, edgecolor='k')
	_log.info(f"{fn_name} | Added BORDERS feature")
	# add ocean
	ax.add_feature(cfeature.OCEAN, linestyle='-',linewidth=0.8, edgecolor='k')
	_log.info(f"{fn_name} | Added OCEAN feature")
	# add lakes
	ax.add_feature(cfeature.LAKES, linestyle='-',linewidth=0.8, edgecolor='k')
	_log.info(f"{fn_name} | Added LAKES feature")
	# add rivers
	ax.add_feature(cfeature.RIVERS, linestyle='-',linewidth=0.8, edgecolor='k')
	_log.info(f"{fn_name} | Added RIVERS feature")
	# add land
	# ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k')
	# add coastlines
	ax.coastlines(resolution='50m', zorder=3)
	_log.info(f"{fn_name} | Added COASTLINES")
	
	
	# add stock image
	# ax.stock_img()

	return ax



@export
@debug(log=_log)
def highlight_ba(ax:Any, y:float, x:float, color:str):
	"""
	Plots lines and a circle to highlight a specific point on a map.

	Parameters
	----------
	ax : Any
		The matplotlib axis object where to plot the elements.
	y : float
		The latitude value of the point to highlight.
	x : float
		The longitude value of the point to highlight.
	color : str
		The color to use for the lines and circle.

	Returns
	-------
	ax : Any 
		The modified matplotlib axis object.
	"""
	
	# define function name
	fn_name = inspect.currentframe().f_code.co_name

	# plot lines corresponding to latitude and longitude value of burned areas
	ax.axhline(y=y, color=color, linewidth=3, zorder=3, linestyle=':')
	ax.text(-181.5, np.round(y, 2), f'{np.round(y, 2)}', color=color, fontweight='bold', size=50, ha='center', va='center', rotation=90)
	
	ax.axvline(x=x, color=color, linewidth=3, zorder=3, linestyle=':')
	ax.text(np.round(x, 2), -90.5, f'{np.round(x, 2)}', color=color, fontweight='bold', size=50, ha='center', va='top')
	
	_log.info(f"{fn_name} | Added lines corresponding to specific latitude and longitude values of burned areas")
	
	# plot cicle around the pixel with the value of burned areas
	circle = plt.Circle((x, y), 1, color=color, linewidth=5, fill=False, zorder=3)
	ax.add_patch(circle)
	
	_log.info(f"{fn_name} | Added circles corresponding to specific latitude and longitude values of burned areas")
	
	return ax



@export
@debug(log=_log)
def set_axis(ax, is_y:bool, latlon_vals, gl):
	"""
	Sets the axis labels, ticks, and formatters for latitude or longitude on a map.

	Parameters
	----------
	ax : Any
		The matplotlib axis object to modify.
	is_y : bool
		True if setting the y-axis, False for x-axis.
	latlon_vals : np.array
		The list of latitude or longitude values for the axis.
	gl : Any
		The gridlines object for the map.

	Returns
	-------
	ax : Any
		The modified axis object.
	"""

	# define function name
	fn_name = inspect.currentframe().f_code.co_name

	values = latlon_vals
	if is_y:
		lat_formatter = LatitudeFormatter()
		ax.yaxis.set_major_formatter(lat_formatter)
		ax.yaxis.set_major_locator(mticker.FixedLocator(values))
		ax.set_yticklabels(values, fontweight='bold', size=50, rotation=90)
		ax.set_yticks(values)
		gl.xlocator = mticker.FixedLocator(values)
		gl.xlines = False

	else:
		lon_formatter = LongitudeFormatter(zero_direction_label=True)
		ax.xaxis.set_major_formatter(lon_formatter)
		ax.xaxis.set_major_locator(mticker.FixedLocator(values))
		ax.set_xticklabels(values, fontweight='bold', size=50)
		ax.set_xticks(values)
		gl.ylocator = mticker.FixedLocator(values)
		gl.ylines = False
	
	_log.info(f"{fn_name} | Axis labels, ticks, and formatters for latitude or longitude has been set on the map")

	return ax



@export
@debug(log=_log)
def draw_tropics_and_equator(ax):
	"""
	Plots lines representing the Tropic of Cancer, Equator, and Tropic of Capricorn on a map.

	Parameters
	----------
	ax : Any
		The matplotlib axis object where to plot the lines.

	Returns
	-------
	ax : Any: 
		The modified matplotlib axis object.
	"""

	# define function name
	fn_name = inspect.currentframe().f_code.co_name

	ax.axhline(23.5, linestyle=':', color='blue', linewidth=0.7, label='Tropic of Cancer')
	ax.axhline(0.00, linestyle=':', color='black', linewidth=0.7, label='Equator')
	ax.axhline(-23.5, linestyle=':', color='blue', linewidth=0.7, label='Tropic of Capricorn')
	
	_log.info(f"{fn_name} | Added tropics and equator")

	return ax



@export
@debug(log=_log)
def plot_dataset_map(
	avg_target_data:np.array,
	avg_data_on_lats:np.array,
	lowerbound_data:np.array,
	upperbound_data:np.array,
	lats:list,
	lons:list,
	title:str,
	cmap:str) -> None:
	"""
	Generates a comprehensive map visualization of a dataset, highlighting 
	minimum and maximum values alongside their confidence intervals.

	Parameters
	----------
	avg_target_data : np.array
		2D array containing the core data to be visualized as color intensity on the map.
		Missing values (NaN) are handled by setting the color to transparent.

	avg_data_on_lats : np.array
		1D array containing the average of the target data for each latitude value.
		This data is plotted as a line in a secondary subplot.

	lowerbound_data : np.array
		2D array containing the lower bound of the data (e.g., standard deviation or confidence interval) for each latitude and longitude.
		This data is used to shade the area around the average line in the secondary subplot.

	upperbound_data : np.array
		2D array containing the upper bound of the data for each latitude and longitude.
		Similar to `lowerbound_data`, it's used for shading the confidence interval in the secondary subplot.

	lats : list
		List containing the latitude values corresponding to the data.

	lons : list
		List containing the longitude values corresponding to the data.

	title : str
		The title to be displayed at the top of the plot.

	cmap : str
		The name of the colormap to use for visualizing the data on the map.
	
	Returns
	-------
	None
		Saves the figure as a high-resolution PNG image (300 dpi) but does not return anything.
	"""

	# define function name
	fn_name = inspect.currentframe().f_code.co_name

	# define color
	color = 'darkred' #fc6742 #4296fc #990e0e
	
	# compute maximum along latitudes and longitudes and find index
	maximum_val = np.nanmax(avg_target_data)
	lat_idx_max, lon_idx_max  = np.where(avg_target_data==maximum_val)
	max_val_latitude = lats[lat_idx_max][0]
	max_val_longitude = lons[lon_idx_max][0]
	_log.info(f"{fn_name} | Computed max value and index found")
	
	# compute minimum along latitudes and longitudes and find index
	minimum_val = np.nanmin(avg_target_data)
	lat_idx_min, lon_idx_min  = np.where(avg_target_data==minimum_val)
	min_val_latitude = lats[lat_idx_min][0]
	min_val_longitude = lons[lon_idx_min][0]
	_log.info(f"{fn_name} | Computed min value and index found")
	
	# define figure and subplots
	_, ax1 = plt.subplots(figsize=(90, 80), subplot_kw=dict(projection=datacrs), sharey=True)
	
	# set title of the plot
	ax1.set_title(title, fontweight='bold', size=80)
	_log.info(f"{fn_name} | Set title")
	
	# set x and y labels
	ax1.set_xlabel('Longitude [deg]', fontweight='bold', size=50)
	ax1.set_ylabel('Latitude [deg]', fontweight='bold', size=50)
	_log.info(f"{fn_name} | Set axes titles")
	
	# set map extent
	ax1.set_extent(**extent_args)
	
	# plot map features such as borders, sea, lakes, rivers and background image
	ax1 = draw_features(ax=ax1)
	_log.info(f"{fn_name} | Plot features")
	
	# plot data on the map
	cmap = plt.get_cmap(cmap)
	cmap.set_under((0, 0, 0, 0))
	h = ax1.pcolormesh(lons, lats, avg_target_data, transform=datacrs, cmap=cmap, zorder=3, alpha=0.5)
	
	# highlight pixel where the maximum value of burned areas has been found and put a circle around it
	ax1 = highlight_ba(ax=ax1, y=max_val_latitude, x=max_val_longitude, color=color)
	_log.info(f"{fn_name} | Highlighted pixel with maximum and minimum values of burned areas")
	
	# highlight pixel where the minimum value of burned areas has been found and put a circle around it
	ax1 = highlight_ba(ax=ax1, y=min_val_latitude, x=min_val_longitude, color='green')

	# add grid lines for latitude and longitude
	gl = ax1.gridlines(crs=datacrs, draw_labels=False, linewidth=1.5, color='gray', alpha=0.5, linestyle='-', zorder=3)
	_log.info(f"{fn_name} | Added gridlines for latitudes and longitudes")
	
	# define longitudes and set x ticks
	ax1 = set_axis(ax=ax1, is_y=False, latlon_vals=longitudes, gl=gl)
	_log.info(f"{fn_name} | Set x ticks for longitudes")

	# define latitudes and set y ticks
	ax1 = set_axis(ax=ax1, is_y=True, latlon_vals=latitudes, gl=gl)
	_log.info(f"{fn_name} | Set y ticks for latitudes")
	
	# define latitudes for tropics and equator
	ax1 = draw_tropics_and_equator(ax=ax1)
	_log.info(f"{fn_name} | Drew tropics and equator")

	# add subplot
	divider = make_axes_locatable(ax1)
	ax2 = divider.append_axes("right", size="10%", pad=0.5, axes_class=plt.Axes)
	_log.info(f"{fn_name} | Added divider between the main plot and the subplot")

	# plot data
	ax2.plot(avg_data_on_lats, lats, color='red', linewidth=1)
	ax2.plot(upperbound_data, lats, alpha=0.3, color='black', linewidth=0.5)
	ax2.plot(lowerbound_data, lats, alpha=0.3, color='black', linewidth=0.5)
	_log.info(f"{fn_name} | Plotted fires distribution along latitudes")
	
	# fill space between lines
	ax2.fill_betweenx(y=lats, x1=avg_data_on_lats, x2=upperbound_data, color='gray', alpha=0.15)
	ax2.fill_betweenx(y=lats, x1=avg_data_on_lats, x2=lowerbound_data, color='gray', alpha=0.15)
	
	# define latitudes for tropics (in degrees) and equator
	ax2 = draw_tropics_and_equator(ax=ax2)
	
	# plot max position
	ax2.axhline(max_val_latitude, color=color, linewidth=3)
	
	# plot min position
	ax2.axhline(min_val_latitude, color='green', linewidth=3)
	
	# set x label	
	ax2.set_xlabel(' Mean ', fontweight='bold', size=50, labelpad=50)
	
	# create list of max values
	ax2_vals = np.around([np.nanmin(lowerbound_data, axis=0), np.nanmax(avg_data_on_lats, axis=0), np.nanmax(upperbound_data, axis=0)], 2)
	
	# plot axes tick lines§
	for tick in ax2_vals:
		ax2.axvline(x=tick, color='blue', alpha=1, linewidth=1, linestyle=':')
		ax2.text(round(tick), -.005, f'{round(tick)}', color='blue', fontweight='bold', size=50, transform=ax2.get_xaxis_transform(), ha='center', va='top')

	ax2.text(0, -.005, '0', color='black', fontweight='bold', size=50, transform=ax2.get_xaxis_transform(), ha='center', va='top') 
	
	ax2.set_xticks([])
	ax2.set_yticks([])
	ax2.set_ylim(bottom=-90, top=90)
	ax2.margins(y=0)
	# ax2.autoscale_view(scaley=True)
		
	# add colorbar plot
	ax_cb = divider.append_axes("right", size="2%", pad=0.3, axes_class=plt.Axes)
	cbar = plt.colorbar(h, ax_cb)
	cbar.ax.tick_params(labelsize=30)
	cbar.ax.set_ylabel('Hectares', color='black', fontweight='bold', size=50, labelpad=50, rotation=270)
	_log.info(f"{fn_name} | Added colorbar")
	
	plt.tight_layout()
	plt.savefig(f"./images/fcci {title}.png")#, dpi=300)
	# plt.clf()
	_log.info(f"{fn_name} | Saved plot in ./images/fcci {title}.png")


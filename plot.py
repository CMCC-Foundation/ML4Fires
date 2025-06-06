import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap

def plot_white_background_maps(actual, predicted, lats, lons):
	# Mask NaNs – ensure they're masked, not just replaced
	actual_masked = np.ma.masked_invalid(actual)
	predicted_masked = np.ma.masked_invalid(predicted)

	# Color levels for discrete colorbar
	levels = [0.0000, 0.0001, 0.0005, 0.0010, 0.0050, 0.0130, 0.0250, 0.0400, 0.0600, 0.0800]

	# Jet color map (discrete) with white for NaNs
	base_cmap = plt.cm.get_cmap("jet", len(levels) - 1)
	colors = base_cmap(np.arange(base_cmap.N))
	new_colors = np.vstack([[1, 1, 1, 1], colors])  # prepend white
	cmap = ListedColormap(new_colors)
	norm = BoundaryNorm(levels, cmap.N)

	fig, axs = plt.subplots(2, 1, figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
	fig.patch.set_facecolor('white')

	for ax, data, title in zip(
		axs,
		[actual_masked, predicted_masked],
		["(a) Actual burned area", "(b) Prediction burned area"]
	):
		ax.set_title(title, fontsize=13, weight='bold', loc='left')
		ax.set_global()
		ax.set_facecolor('white')
		ax.coastlines(color='black', linewidth=0.6)
		ax.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.4)
		ax.add_feature(cfeature.LAND, facecolor='white')
		ax.add_feature(cfeature.OCEAN, facecolor='white')
		ax.gridlines(draw_labels=False, color='gray', linewidth=0.3)

		# pcolormesh with NaN masking respected
		mesh = ax.pcolormesh(
			lons,
			lats,
			data,
			cmap=cmap,
			norm=norm,
			shading='auto',
			transform=ccrs.PlateCarree()
		)

	# Colorbar on the right
	cbar = fig.colorbar(mesh, ax=axs, orientation='vertical', shrink=0.7, pad=0.02, aspect=30)
	cbar.set_label("Burned Area (Mha)", fontsize=12)
	cbar.set_ticks(levels)
	cbar.ax.tick_params(labelsize=10)

	plt.show()

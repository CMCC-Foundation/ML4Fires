## Input Data Preparation

#### Choosing the variables

<p align="justify"> SeasFire Cube v3 provides a list of 59 climate variables on a regular grid that are related to wildfires in an xArray dataset stored in <code>.zarr</code> format. As described previously in the <a href="../docs/data.md">Data</a> section, a subset of 12 climate variables was chosen. Most of them have <i>time</i>, <i>latitude</i> and <i>longitude</i> as their coordinates. Other variables, such as <code>fcci_ba_valid_mask</code> and <code>gwis_ba_valid_mask</code>, have only <i>time</i> coordinates. Finally, the variable <code>lsm</code> has <i>latitude</i> and <i>longitude</i> as its coordinates; this is because the Land-Sea Mask is a binary map of water bodies. </p>

#### Preprocessing data

<p align="justify"> In order to have a dataset with <i>time</i>, <i>latitude</i> and <i>longitude</i> coordinates, some of the chosen variables must be adjusted, in particular <code>fcci_ba_valid_mask</code> and <code>gwis_ba_valid_mask</code> must be regenerated as maps with the <i>latitude</i> and <i>longitude</i> coordinates for each timestamp present in the _time_ coordinate. For this purpose, for each timestamp, the burned area map (<code>fcci_ba</code> and <code>gwis_ba</code>) is taken and the binary mask is calculated by placing 1 in pixels where the burned area map has hectare values greater than 0, otherwise 0. Whereas, the variable <code>lsm</code> has <i>latitude</i> and <i>longitude</i> as coordinates, since it is a binary map of the globe in which pixels are equal to 1 in land areas and 0 in sea areas. To add the <i>time</i> coordinate to the variable <code>lsm</code>, the map was repeated for each timestamp in the dataset.</p>

<p align="justify"> Following these operations, <code>merged_ba</code> and <code>merged_ba_valid_mask</code> variable maps were composed by merging FCCI and GWIS data. It was intended to construct these maps because burned areas present in FCCI were missing in the GWIS data and vice versa. The pixel values of the common burned areas were averaged. Conversely, pixels where burned areas were 0 in one source (FCCI or GWIS) were replaced with values from the other. With this operation, the complete burned area maps were obtained.</p>

#### Scalers Creation

<p align="justify"> Once the dataset has been created, it is necessary to choose a scaler to scale data. Tipically, the data are scaled using Standard or MinMax scalers. To achieve this, several maps have been created: </p>

- Standard Scaler: Mean and Standard Deviation maps;
- MinMax Scaler: Maximum and Minimum maps.

> Why create maps to scale data?

<p align="justify"> A pixel-wise scaling is needed because the climate variables under consideration vary not only in time but also have different values by spatial domain. Therefore, it is necessary to compute for each variable, along the time axis, maps of Mean, Standard Deviation, Minimum and Maximum. Thus, we obtain bidimensional maps containing only the <code>latitude</code> and <code>longitude</code> coordinates for each variable in the dataset. These maps can be used to build Standard and MinMax scalers and scale data. </p>

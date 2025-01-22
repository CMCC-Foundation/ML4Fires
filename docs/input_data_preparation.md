# [&larr;](../README.md) Input Data Preparation

## Choosing the variables

<p align="justify"> SeasFire Cube v3 provides a list of 59 climate variables on a regular grid that are related to wildfires in an xArray dataset stored in <code>.zarr</code> format. As described previously in the <a href="../docs/data.md" style="text-decoration:none;">Data</a> section, a subset of 8 climate variables was chosen. Most of them have <i>time</i>, <i>latitude</i> and <i>longitude</i> as their coordinates. The variable <code>lsm</code> is the only variable having <i>latitude</i> and <i>longitude</i> as its coordinates; this is because the Land-Sea Mask is a binary map of water bodies. </p>

## Preprocessing data

<p align="justify"> In order to have a dataset with <i>time</i>, <i>latitude</i> and <i>longitude</i> coordinates, the variable <code>lsm</code> must be adjusted. The variable <code>lsm</code> has <i>latitude</i> and <i>longitude</i> as coordinates, since it is a binary map of the globe in which pixels are equal to 1 in land areas and 0 in sea areas. To add the <i>time</i> coordinate to the variable <code>lsm</code>, the map was repeated for each timestamp in the dataset.</p>

## Scalers Creation

<p align="justify"> Once the dataset has been created, it is necessary to choose a scaler to scale data. Tipically, the data are scaled using Standard or MinMax scalers. Two different approaches can be used to achieve this, one based on maps and one based on arrays containing individual values of the reference statistics for the chosen scaler: </p>

- Standard Scaler: Mean and Standard Deviation maps or arrays;
- MinMax Scaler: Maximum and Minimum maps or arrays.

<p align="justify"> The climatic variables considered not only change over time, but also geographically. Although it is more intuitive to proceed with the first approach and use maps, the second approach is used because the variations that the climatic variables can undergo in the time frame considered are not significant and, consequently, it is conceivable to use a single representative value for the Mean, Standard Deviation, Maximum and Minimum statistics. </p>

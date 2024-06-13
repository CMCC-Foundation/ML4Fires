# Data

## Sources

<p align="justify"> The first data source is the <a href="https://zenodo.org/records/8055879"> SeasFire Cube v3 </a>, a data cube of pre-processed data provided by SeasFire project integrating multiple sources. The <b>SeasFire Cube Dataset</b> contains a set of 59 climate variables aligned on a <b>regular grid</b>.</p>

All variables have:

1. **spatial resolution** of  **$\small{0.25° \times 0.25°}$**
1. **temporal resolution** of **8 days**

|                Feature             |            Value             |
|                :--                 |             :--              |
|          Spatial Coverage          |            Global            |
|         Spatial Resolution         | $\small{0.25° \times 0.25°}$ |
|         Temporal Coverage          | 20 years (2001 &rarr; 2021)  |
|        Temporal Resolution         |            8 days            |
|        Number of variables         |              59              |
|          Overall size              |           ~ 44 Gb            |

<p align="justify"> The second data source is <a href="https://esgf-node.llnl.gov/projects/cmip6/">CMIP6</a> data. <b>CMIP6</b> data are climate model projections produced on long time frames (up to 2100) by different climate modeling centers. Data is provided (mainly) on <b>regular grids</b> in NetCDF format and made available through the ESGF infrastructure. Various scenarios are implemented according to different <i>Shared Socioeconomic Pathways (SSP)</i>, representative of socio-economic and greenhouse gas emissions projected changes. Such data will be used for model inference.</p>

|                Feature             |            Value             |
|                :--                 |             :--              |
|          Spatial Coverage          |            Global            |
|         Spatial Resolution         | $\small{0.25° \times 0.25°}$ |
|         Temporal Coverage          |      2020 &rarr; 2100        |
|        Temporal Resolution         |            Daily             |
|     Scenarios to be considered     | SSP2-4.5, SSP3-7.0, SSP5-8.5 |

## Overview

<p align="justify"> The dataset used to carry on the development of the code is composed by 14 climate variables: </p>

- 12 climate variables belong to SeasFire Cube v3;
- 2 (`merged_ba` and `merged_ba_valid_mask`) are obtained by merging _FCCI_ (`fcci_ba` and `fcci_ba_valid_mask`) and _GWIS_ (`gwis_ba` and `gwis_ba_valid_mask`) variables on burned areas.

### Drivers

<p align="justify"> The table below provides additional details regarding the list of all used variables. As can be seen, for each SeasFire Cube v3 climate variable is associated the CMIP6 climate variable short name in order to provide a match between them.</p>

| Variable | CMIP6 | SeasFire Cube | Shape | Units |  Origin |
| :--- | :--- | :--- | :--- | :--- |  :--- |
| _Leaf Area Index_ | `lai` | `lai` | (**time**, lat, lon) | $\textit{m}^2 \textit{m}^{-2}$ | Nasa MODIS MOD11C1, MOD13C1, MCD15A2 |
| _Land-Sea Mask_ | `sftlf` | `lsm` | (lat, lon) | 0-1 | ERA5 |
| _Land Surface Temperature at Day_ | `ts` | `lst_day` | (**time**, lat, lon) | K | Nasa MODIS MOD11C1, MOD13C1, MCD15A2 |
| _Relative Humidity_ | `hur` | `rel_hum` | (**time**, lat, lon) | % | ERA5 |
| _Surface net Solar Radiation_ | `rss` | `ssr` | (**time**, lat, lon) | $\small{MJm^{-2}}$  |ERA5 |
| _Sea Surface Temperature_ | `tos` | `sst` | (**time**, lat, lon) | K | ERA5 |
| _Temperature at 2 meters - Min_ | `tasmin` | `t2m_min` | (**time**, lat, lon) | K | ERA5 |
| _Total Precipitation_ | `pr` | `tp` | (**time**, lat, lon) | m | ERA5 |

### Targets

<p align="justify"> The table below contains all the targets used to train the Deep Neural Network model. To train the model, a data source must be chosen among FCCI, GWIS and MERGE.</p>

| Variable | CMIP6 | SeasFire Cube | Shape | Units | Origin |
| :--- | :---: | :--- | :--- | :--- | :--- |
| _Burned Areas - FCCI_ | - | `fcci_ba` | (**time**, lat, lon) | ha | Fire Climate Change Initiative (FCCI) |
| _Valid mask (B.A.) - FCCI_ | - | `fcci_ba_valid_mask` | (**time**) | 0-1 | Fire Climate Change Initiative (FCCI) |
| _Burned Areas - GWIS_ | - | `gwis_ba` | (**time**, lat, lon) | ha | Global Wildfire Information System  (GWIS) |
| _Valid mask (B.A.) - GWIS_ | - | `gwis_ba_valid_mask` | (**time**) | 0-1 | Global Wildfire Information System  (GWIS) |
| _Burned Areas - MERGE_ | - | `merge_ba` | (**time**, lat, lon) | ha | _Computed_ |
| _Valid mask (B.A.) - MERGE_ | - | `merge_ba_valid_mask` | (**time**) | 0-1| _Computed_ |

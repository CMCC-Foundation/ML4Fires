ML4Fires
========

This repository includes the code to develop and test data-driven models for Wildfire Burned Areas Prediction and Projection case studies<!--in InterTwin Project on Wildfires-->.
The main aim is to provide ML pipelines and models to investigate the Wildfires spread and propagation around the world focusing on Burned Areas.

<!-- The main aim is to provide ML pipelines and models to investigate the Wildfires spread and propagation around the world focusing on Burned Areas.-->

## Overview

Fire plays an important role in shaping ecosystems, but its long-term carbon-neutral status can be altered by climate change; in fact, climate change is the main cause of worsening fire-related weather factors and increasing global fire activity.

The expansion of fires in evergreen forest regions could weaken their ability to act as carbon sinks, releasing stored carbon into the atmosphere and providing feedback to climate change.

Therefore, it is crucial to improve understanding and anticipation of fires in the Earth system, particularly by assessing the likelihood of occurrence of large events, as climate change and human activity continue to influence fire regimes globally.

Detecting and predicting wildfires is challenging and recent advances in Machine Learning (ML) provide cutting-edge modeling techniques to deal with detection and prediction tasks, offering also cost-effective and fast-computing solutions.

## The backbone architecture

Data-driven models can be used to understand how climate change is affecting the frequency and location of wildfire events, according to different future projection scenarios based on different socioeconomic changes.

Different DNN architectures must be evaluated to address the various use cases, including Convolutional and Graph networks. The trained models will be used as a base for understanding how different climate scenarios can affect extreme events frequency, locations, probability, etc. Future projection data from large experiments, like the Climate Model Intercomparison Project 6th phase (CMIP6), will be used as input for the trained models.

The [UNet++](https://arxiv.org/abs/1807.10165) was chosen as the backbone architecture of the **first version** of our data-driven model to predict fire danger (burned area maps) based on future climate change projections.

Contributors
============

[CMCC - Institute for Earth System Predictions (IESP) - Advanced Scientific Computing (ASC) Division](https://www.cmcc.it)

| Contributor | Contact | Role |
| :-- | :-- | :-- |
| Donatello Elia | <donatello.elia@cmcc.it> | Project Manager
| Emanuele Donno | <emanuele.donno@cmcc.it> | Main Contributor
| Davide Donno | <davide.donno@cmcc.it> | Project Contributor
| Gabriele Accarino | <gabriele.accarino@cmcc.it> | Project Contributor
| Giovanni Aloisio | <giovanni.aloisio@cmcc.it> | Group Coordinator

## Data

### Sources

The first data source is the [SeasFire Cube v3](https://zenodo.org/records/8055879), a data cube of pre-processed data provided by SeasFire project integrating multiple sources. The **SeasFire Cube Dataset** contains a set of 59 climate variables aligned on a **regular grid**. All variables have a **spatial resolution** of **$\small{0.25° \times 0.25°}$** and a **temporal resolution** of **8 days**.

| Feature | Value |
| :-- | :-- |
| Spatial Coverage | Global |
| Spatial Resolution | $\small{0.25° \times 0.25°}$ |
| Temporal Coverage  | 20 years (2001 &rarr; 2021) |
| Temporal Resolution  | 8 days |
| Number of variables | 59 |
| Overall size     | ~ 44 Gb |

The second data source is [CMIP6](https://esgf-node.llnl.gov/projects/cmip6/) data. **CMIP6** data are climate model projections produced on long time frames (up to 2100) by different climate modeling centers. Data is provided (mainly) on **regular grids** in NetCDF format and made available through the ESGF infrastructure. Various scenarios are implemented according to different _Shared Socioeconomic Pathways (SSP)_, representative of socio-economic and greenhouse gas emissions projected changes. Such data will be used for model inference.

| Feature | Value |
| :-- | :-- |
| Spatial Coverage | Global |
| Spatial Resolution | $\small{0.25° \times 0.25°}$ |
| Temporal Coverage  | 2020 &rarr; 2100 |
| Temporal Resolution  | Daily |
| Scenarios to be considered | SSP2-4.5, SSP3-7.0, SSP5-8.5 |

### Overview

The dataset used to carry on the development of the code is composed by 14 climate variables:

- 12 climate variables belong to SeasFire Cube v3;
- 2 (`merged_ba` and `merged_ba_valid_mask`) are obtained by merging _FCCI_ (`fcci_ba` and `fcci_ba_valid_mask`) and _GWIS_ (`gwis_ba` and `gwis_ba_valid_mask`) variables on burned areas.


#### Drivers

The table below provides additional details regarding the list of all used variables. As can be seen, for each SeasFire Cube v3 climate variable is associated the CMIP6 climate variable short name in order to provide a match between them.

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

#### Targets

The table below contains all the targets used to train the Deep Neural Network model. To train the model, a data source must be chosen among FCCI, GWIS and MERGE.

| Variable | CMIP6 | SeasFire Cube | Shape | Units | Origin |
| :--- | :---: | :--- | :--- | :--- | :--- |
| _Burned Areas - FCCI_ | - | `fcci_ba` | (**time**, lat, lon) | ha | Fire Climate Change Initiative (FCCI) |
| _Valid mask (B.A.) - FCCI_ | - | `fcci_ba_valid_mask` | (**time**) | 0-1 | Fire Climate Change Initiative (FCCI) |
| _Burned Areas - GWIS_ | - | `gwis_ba` | (**time**, lat, lon) | ha | Global Wildfire Information System  (GWIS) |
| _Valid mask (B.A.) - GWIS_ | - | `gwis_ba_valid_mask` | (**time**) | 0-1 | Global Wildfire Information System  (GWIS) |
| _Burned Areas - MERGE_ | - | `merge_ba` | (**time**, lat, lon) | ha | _Computed_ |
| _Valid mask (B.A.) - MERGE_ | - | `merge_ba_valid_mask` | (**time**) | 0-1| _Computed_ |

## Input Data Preparation

#### Choosing the variables

SeasFire Cube v3 provides a list of 59 climate variables on a regular grid that are related to wildfires in an xArray dataset stored in `.zarr` format. As described previously in the [Data](#overview-1) section, a subset of 12 climate variables was chosen. Most of them have _time_, _latitude_ and _longitude_ as their coordinates. Other variables, such as `fcci_ba_valid_mask` and `gwis_ba_valid_mask`, have only _time_ coordinates. Finally, the variable `lsm` has _latitude_ and _longitude_ as its coordinates; this is because the Land-Sea Mask is a binary map of water bodies.

#### Preprocessing data

In order to have a dataset with _time_, _latitude_ and _longitude_ coordinates, some of the chosen variables must be adjusted, in particular `fcci_ba_valid_mask` and `gwis_ba_valid_mask` must be regenerated as maps with the _latitude_ and _longitude_ coordinates for each timestamp present in the _time_ coordinate. For this purpose, for each timestamp, the burned area map (`fcci_ba` and `gwis_ba`) is taken and the binary mask is calculated by placing 1 in pixels where the burned area map has hectare values greater than 0, otherwise 0. Whereas, the variable `lsm` has _latitude_ and _longitude_ as coordinates, since it is a binary map of the globe in which pixels are equal to 1 in land areas and 0 in sea areas. To add the _time_ coordinate to the variable `lsm`, the map was repeated for each timestamp in the dataset.

Following these operations, `merged_ba` and `merged_ba_valid_mask` variable maps were composed by merging FCCI and GWIS data. It was intended to construct these maps because burned areas present in FCCI were missing in the GWIS data and vice versa. The pixel values of the common burned areas were averaged. Conversely, pixels where burned areas were 0 in one source (FCCI or GWIS) were replaced with values from the other. With this operation, the complete burned area maps were obtained.

#### Scalers Creation

Once the dataset has been created, it is necessary to choose a scaler to scale data. Tipically, the data are scaled using Standard or MinMax scalers. To achieve this, several maps have been created:

- Standard Scaler: Mean and Standard Deviation maps;
- MinMax Scaler: Maximum and Minimum maps.

> Why create maps to scale data?

A pixel-wise scaling is needed because the climate variables under consideration vary not only in time but also have different values by spatial domain. Therefore, it is necessary to compute for each variable, along the time axis, maps of Mean, Standard Deviation, Minimum and Maximum. Thus, we obtain bidimensional maps containing only the `latitude` and `longitude` coordinates for each variable in the dataset. These maps can be used to build Standard and MinMax scalers and scale data.

Conda Environment
-----------------

Python version 3.11.2 or higher is needed.

A conda env containing all the packages and versions required to run the scripts can be created by running the following command:

      conda env create --file wildfires.yaml

> [!TIP]
> **Important**: Remember to replace `“YOUR WORK DIRECTORY“` with the path to your work directory in the `wildfires.yaml` file in order to install properly the conda environment.

This makes the installation easy and fast. The Tensorflow v2.12.0 was used to train and test the ML4Fires architecture on the [Juno](https://www.cmcc.it/super-computing-center-scc) supercomputer at CMCC.

Library Structure
-----------------

The library is structured as follows:
```bash
├── README.md
├── config
│   ├── configuration.toml
│   ├── models.toml
├── data
│   ├── seasfire_v03.zarr
│   ├── scaler
│   │   ├── map_trn_max.npy
│   │   ├── map_trn_min.npy
│   │   ├── map_trn_mean.npy
│   │   ├── map_trn_stdv.npy
│   ├── FCCI
│   │   ├── 00_days_dataset
│   ├── GWIS
│   │   ├── 00_days_dataset
│   ├── MERGE
│   │   ├── 00_days_dataset
├── experiments
├── library
│   ├── __init__.py
│   ├── augmentation.py
│   ├── cache.py
│   ├── configuration.py
│   ├── dataset_builder_wf.py
│   ├── dataset_builder.py
│   ├── dataset_creator.py
│   ├── decorators.py
│   ├── hyperparams.py
│   ├── layers.py
│   ├── macros.py
│   ├── models.py
│   ├── scaling.py
│   ├── tfr_io.py
│   ├── timer_class.py
├── main.py
├── launch.sh
```


[**`config`**](./config/) is a directory used to store [TOML](https://toml.io/en/) configuration files.

[**`data`**](./data/) is a directory that is used to store important files (e.g., scalers, symbolic links to the dataset, etc).

[**`experiments`**](./experiments/) directory is created at runtime and is used to store, in a hierarchical way, the outputs of the training executions.

[**`library`**](./library/) directory is used to store, in an organized way, all the code that provides support to the `main.py` script during its execution. Model implementations and training utility functions can be found here.

[**`main.py`**](./main.py) file has all the workflow code.

[**`launch.sh`**](./launch.sh) is the file that must be executed to run the experiments.

>[!WARNING]
Both `data` and `experiments` directories are **empty** and are **not** included in the repository as they contain too heavy files that cannot be stored.

>[!TIP]
Before running the code, remeber to download the [SeasFire Cube v3](https://zenodo.org/records/8055879) and put into `data` folder.
The best model will be saved in the `experiments` folder.




### How to run the code on a LSF cluster (with GPUs)

In order to execute the code, the working directory **must be set** to `src` and a `*.sh` file must be prepared with the following template:

```bash
#!/bin/sh
#BSUB -n total_n_jobs
#BSUB -q queue_name
#BSUB -P project_code
#BSUB -J exec_name
#BSUB -o ./out_file.out
#BSUB -e ./err_file.err
#BSUB -gpu "num=n_gpus_per_node"

python main.py --arguments **cli_arguments
```

For our experiment, we run the code on the Juno LSF cluster, using the following command:

```bash
# locate the current working directory to src
cd src

# e.g. training
bsub < launch.sh
```

## Acknowledgment

This software product is developed in the context of Wildfires use case in the InterTwin Project:
[![InterTwin](images/intertwin.png)](https://www.intertwin.eu/intertwin-use-case-a-digital-twin-for-projecting-wildfire-danger-due-to-climate-change)

<!-- ## Future Investigations -->

# [&larr;](../README.md) Library Structure

<p align="justify">  The library is structured as follows: </p>

    ├── README.mdLR    │
    ├── config
    │    │
    │    ├── experiments
    │    │   ├── UPP_16.toml
    │    │   ├── UPP_32.toml
    │    │   ├── UPP_64.toml
    │    │   ├── UPP_all.toml
    │    │
    │    ├── configuration.toml
    │    ├── credentials.toml
    │    ├── models.toml
    │    ├── torch.toml
    │    ├── train.toml
    │
    ├── data
    │   ├── sfv03_fcci.zarr
    │   ├── data_025km.zarr
    │   ├── data_100km.zarr
    │   │
    │   ├── scaler
    │   │   ├── fcci_max_point_map.nc
    │   │   ├── fcci_min_point_map.nc
    │   │   ├── fcci_mean_point_map.nc
    │   │   ├── fcci_stdv_point_map.nc
    │
    ├── digital_twin_notebooks
    │   │
    │   ├── img
    │   ├── inference_on_test_data.ipynb
    │   ├── inference_with_onnx.ipynb
    │   ├── model_conversion_pytorch_to_onnx.ipynb
    │
    ├── docs
    │
    ├── experiments
    │   │
    │   ├── 20240311_222733
    │   │   ├── exp_5.toml
    │   │   ├── fabric_benchmark.csv
    │   │   ├── last_model.onnx
    │   │   ├── last_model.pt
    │
    ├── Fires
    │   │
    │   ├── _datasets
    │   │   ├── dataset_zarr.py
    │   │   ├── torch_dataset.py
    │   │
    │   ├── _layers
    │   │   ├── unetpp.py
    │   │
    │   ├── _macros
    │   │   ├── macros.py
    │   │
    │   ├── _models
    │   │   ├── base.py
    │   │   ├── unet.py
    │   │   ├── unetpp.py
    │   │   ├── vgg.py
    │   │
    │   ├── _plots
    │   │   ├── plot_utils.py
    │   │
    │   ├── _scalers
    │   │   ├── base.py
    │   │   ├── minmax.py
    │   │   ├── scaling_maps.py
    │   │   ├── standard.py
    │   │
    │   ├── _utilities
    │   │   ├── callbacks.py
    │   │   ├── cli_args_checker.py
    │   │   ├── cli_args_parser.py
    │   │   ├── configuration.py
    │   │   ├── decorators.py
    │   │   ├── logger_itwinai.py
    │   │   ├── logger.py
    │   │   ├── metrics.py
    │   │   ├── utils_general.py
    │   │   ├── utils_inference.py
    │   │   ├── utils_mlflow.py
    │   │   ├── utils_model.py
    │   │   ├── utils_t.pyrainer
    │   │
    │   ├── __init__.py
    │   ├── augmentation.py
    │   ├── datasets.py
    │   ├── layers.py
    │   ├── macros.py
    │   ├── models.py
    │   ├── scalers.py
    │   ├── trainer.py
    │   ├── utils.py
    │
    ├── experiments.py
    ├── main.py
    ├── phase_inference_test_100.py
    ├── phase_training_100.py
    ├── train.sh



| File |      Type     |     Main function     |
| :--  |      :--:      |          :--          |
|[**`config`**](../config/) | $\small{\textcolor{blue}{\texttt{DIR}}}$ | It stores [**`TOML`**](https://toml.io/en/) configuration files|
|[**`data`**](../data/) | $\small{\textcolor{blue}{\texttt{DIR}}}$ | It stores important files (e.g., scalers, symbolic links to the dataset, etc)|
|[**`digital_twin_notebooks`**](../digital_twin_notebooks/) | $\small{\textcolor{blue}{\texttt{DIR}}}$ | It stores Jupyter notebooks that carry on the Digital Twin's tasks on Wildfires use case|
|[**`docs`**](../docs/) | $\small{\textcolor{blue}{\texttt{DIR}}}$ | It stores documentation files|
|[**`experiments`**](../experiments/) | $\small{\textcolor{blue}{\texttt{DIR}}}$ | <p align="justify"> It contains folders named with the current date and time when the experiment took place. The best model, the experiment configuration file and the benchmark file will be saved in this folder after the completion of the experiment. </p> |
|[**`Fires`**](../Fires/) | $\small{\textcolor{blue}{\texttt{DIR}}}$ | <p align="justify"> It is the main library that is used to carry on the training of the Machine Learning model and the inference on the SeasFireCube data. It is used to store, in an organized way, all the code that provides support to the `main.py` script during its execution. Model implementations and training utility functions can be found here. ([details](./fires.md)) </p> |
|[**`main.py`**](../main.py) | $\small{\textcolor{red}{\texttt{FILE}}}$ | <p align="justify"> It contains all the workflow code that must be executed. </p> |
|[**`phase_inference_test_100.py`**](../phase_inference_test_100.py) | $\small{\textcolor{red}{\texttt{FILE}}}$ | <p align="justify"> It contains all the workflow code that must be executed. </p> |
|[**`phase_training_100.py`**](../phase_training_100.py) | $\small{\textcolor{red}{\texttt{FILE}}}$ | <p align="justify"> It contains all the workflow code that must be executed. </p> |
|[**`train.sh`**](../train.sh) | $\small{\textcolor{red}{\texttt{FILE}}}$ | <p align="justify"> It runs the experiments once it has been executed ([details](./run_on_lsf_cluster.md)) </p>|

>[!WARNING]
> <p align="justify"> Both <code>data</code> and <code>experiments</code> directories are <b>empty</b> and are <b>not</b> included in the repository as they contain too heavy files that cannot be stored. </p>

>[!TIP]
> <p align="justify"> Before running the code, remeber to download the <a href="https://zenodo.org/records/8055879" style="text-decoration:none;"> SeasFire Cube v3 </a>  and put into <code>data</code> folder. </p>
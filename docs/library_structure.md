# Library Structure

<p align="justify">  The library is structured as follows: </p>

    ├── README.md
    │
    ├── config
    │    │
    │    ├── experiments
    │    │   ├── UPP_16.toml
    │    │   ├── UPP_32.toml
    │    │   ├── UPP_64.toml
    │    │   ├── UPP_all.toml
    │    │
    │    ├── configuration.toml
    │    ├── discord.toml
    │    ├── models.toml
    │    ├── tflow.toml
    │    ├── torch.toml
    │    ├── train.toml
    │
    ├── data
    │   ├── sfv03_fcci.zarr
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
    │   │   ├── unetpp.py
    │   │   ├── vgg.py
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
    │   │   ├── logger.py
    │   │   ├── swin_model.py
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
    ├── main.py
    ├── launch.sh

<p align="justify">  <b><a href="../config/" style="text-decoration:none;"><code> config </code></a></b> is a directory used to store <a href="https://toml.io/en/" style="text-decoration:none;"> TOML </a> configuration files. </p>


<p align="justify">  <b><a href="../data/" style="text-decoration:none;"><code> data </code></a></b> is a directory that is used to store important files (e.g., scalers, symbolic links to the dataset, etc). </p>

<p align="justify">  <b><a href="../experiments/" style="text-decoration:none;"><code> experiments </code></a> </b> directory is created at runtime and is used to store, in a hierarchical way, the outputs of the training executions. </p>

<p align="justify">  <b><a href="../docs/" style="text-decoration:none;"><code> docs </code></a></b> is a directory used to store documentation files. </p>

<p align="justify"> The <b><a href="../Fires/" style="text-decoration:none;"><code> Fires </code></a></b> folder is the main library that is used to carry on the training of the Machine Learning model and the inference on the SeasFireCube data. It is used to store, in an organized way, all the code that provides support to the `main.py` script during its execution. Model implementations and training utility functions can be found here. </p>

<p align="justify"> <b><a href="../main.py" style="text-decoration:none;"><code> main.py </code></a></b> file has all the workflow code. </p>

<p align="justify"> <b><a href="../launch.sh" style="text-decoration:none;"><code> launch.sh </code></a></b> is the file that must be executed to run the experiments. </p>

>[!WARNING]
> <p align="justify"> Both <code>data</code> and <code>experiments</code> directories are <b>empty</b> and are <b>not</b> included in the repository as they contain too heavy files that cannot be stored. </p>

>[!TIP]
> <p align="justify"> Before running the code, remeber to download the <a href="https://zenodo.org/records/8055879" style="text-decoration:none;"> SeasFire Cube v3 </a>  and put into <code>data</code> folder. </p>

<p align="justify"> The best model will be saved in the <b><a href="../experiments/" style="text-decoration:none;"><code> experiments </code></a></b> folder. </p>
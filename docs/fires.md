# [&larr;](../README.md) Fires



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


### Folders

##### Datasets

<p align="justify"> The <a href="../Fires/_datasets/" target="_blank">_datasets</a> folder contains scripts for handling datasets used in the project: </p>

- <a href="../Fires/_datasets/dataset_zarr.py" target="_blank">dataset_zarr.py</a>

	- **`Dataset025`**
		> Class for handling 0.25° resolution datasets. Creates and saves xarray Datasets in zarr format, processes features and targets, and applies binary masking. Converts burned areas from hectares to percentages for standardization.

	- **`Dataset100`**
		> Class for handling 1° resolution datasets. Converts 25km resolution data to 100km resolution through coarsening and averaging. Handles data scaling and normalization during the conversion process.

	- **`load_zarr`**
		> Function that loads preprocessed zarr datasets. Creates the dataset if it doesn't exist and returns an xarray Dataset object containing all required features.


- <a href="../Fires/_datasets/torch_dataset.py" target="_blank">torch_dataset.py</a>

	- **`FireDataset`** (inherits from **`torch.utils.data.Dataset`**)
		> A PyTorch Dataset class for loading and preprocessing fire data from a Zarr archive. It handles driver and target features, performs data scaling, manages missing values, and yields tensors ready for model training.

##### Layers

<p align="justify"> The <a href="../Fires/_layers/" target="_blank">_layers</a> folder contains the implementation of various neural network layers used in the project: </p>

- <a href="../Fires/_layers/unetpp.py" target="_blank">unetpp.py</a>

	- **`VGGBlock`**: <p align="justify">
		> The `VGGBlock` class implements a fundamental building block of the VGG network architecture, a widely used convolutional neural network (CNN) model. This block consists of two convolutional layers, each followed by batch normalization and a ReLU activation function. Additionally, an optional dropout layer is included in the implementation (currently commented out) to reduce overfitting during training. It is used as a building block in `U-Net++` architecture, which enhances segmentation performance by leveraging nested skip connections. </p>

##### Macros

<p align="justify"> The <a href="../Fires/_macros/" target="_blank">_macros</a> folder contains macro definitions used across the project: </p>

- <a href="../Fires/_macros/macros.py" target="_blank">macros.py</a>: Contains macro definitions.

##### Models

<p align="justify"> The <a href="../Fires/_models/" target="_blank">_models</a> folder contains the implementation of various machine learning models used in the project: </p>

- <a href="../Fires/_models/base.py" target="_blank">base.py</a>: Base class for all models.

	- **`BaseLightningModule`** (inherits from **`pl.LightningModule`**)
		> <p align="justify"> A foundational PyTorch Lightning module that manages training, validation, and testing workflows while logging key metrics and losses. It standardizes model development for efficient training and evaluation.</p>
	- **`BaseUnetPlusPlus`** (inherits from **`BaseLightningModule`**)
		> <p align="justify"> A base class for U-Net++ models, defining key architecture parameters like depth, filters, and deep supervision.</p>
	- **`BaseVGG`** (inherits from **`BaseLightningModule`**)
		> <p align="justify"> A configurable base class for VGG-like convolutional networks, specifying channels, activations, and kernel sizes. It serves as a flexible template for building and experimenting with VGG-based architectures. </p>

- <a href="../Fires/_models/unet.py" target="_blank">unet.py</a>: Implementation of the U-Net model.
	- **`Unet`** (inherits from **`BaseLightningModule`**)
		> <p align="justify"> A U-Net-based model designed using the PyTorch Lightning framework. It defines an encoder-decoder architecture with convolutional blocks, pooling, and upsampling layers. </p>

- <a href="../Fires/_models/unetpp.py" target="_blank">unetpp.py</a>: Implementation of the U-Net++ model.
	- **`UnetPlusPlus`** (inherits from **`BaseLightningModule`**)
		> <p align="justify"> A U-Net++ model designed using the PyTorch Lightning framework. It enhances the U-Net architecture by incorporating nested skip connections and optional deep supervision, improving segmentation accuracy and feature fusion.
	
- <a href="../Fires/_models/vgg.py" target="_blank">vgg.py</a>: Implementation of the VGG model.
	- **`VGG_V1`** (inherits from **`BaseVGG`**)
		> <p align="justify"> A VGG-like model with a predefined convolutional and fully connected layer structure. It provides a balanced architecture for general-purpose feature extraction and classification tasks.</p>
	- **`VGG_V2`** (inherits from **`BaseVGG`**)
		> <p align="justify"> A variant of VGG_V1 with a deeper architecture and additional layers in both convolutional and fully connected parts. It aims to enhance feature extraction and model capacity for more complex tasks.</p>
	- **`VGG_V3`** (inherits from **`BaseVGG`**)
		> <p align="justify"> A further modified version of VGG with an altered arrangement of convolutional and fully connected layers. It is designed for improved performance by optimizing feature extraction and representation learning.</p>

##### Plots

<p align="justify"> The <a href="../Fires/_plots/" target="_blank">_plots</a> folder contains utilities for plotting and visualizing data: </p>

- <a href="../Fires/_plots/plot_utils.py" target="_blank">plot_utils.py</a>: Utility functions for creating plots.


	- **`draw_features(ax)`**
		> Adds geographical features to a map including political borders, oceans, lakes, rivers, and coastlines.

	- **`highlight_ba(ax, y, x, color)`**
		> Draws highlighting elements (lines and circles) on a map to emphasize specific latitude/longitude points.

	- **`set_axis(ax, is_y, latlon_vals, gl)`**
		> Configures axis properties for either latitude or longitude on a map, including labels, ticks, and formatters.

	- **`draw_tropics_and_equator(ax)`**
		> Adds reference lines for the Tropic of Cancer, Equator, and Tropic of Capricorn to a map.

	- **`plot_dataset_map(avg_target_data, avg_data_on_lats, lowerbound_data, upperbound_data, lats, lons, title, cmap)`**
		> Creates a comprehensive visualization combining a geographical map with data overlay and a side panel showing statistical distribution along latitudes.

##### Scalers

<p align="justify"> The <a href="../Fires/_scalers/" target="_blank">_scalers</a> folder contains different scaling techniques: </p>

- <a href="../Fires/_scalers/base.py" target="_blank">base.py</a>: 
	- **`Scaler`** (base class)
		> Base class for data scalers providing interface for transform and inverse_transform operations with PyTorch tensors.

- <a href="../Fires/_scalers/minmax.py" target="_blank">minmax.py</a>
	- **`MinMaxScaler`** (inherits from **`Scaler`**)
		> Scales data to a fixed range [0,1] using minimum and maximum values. Provides transform and inverse_transform methods for scaling tensors.

- <a href="../Fires/_scalers/scaling_maps.py" target="_blank">scaling_maps.py</a>
	- **`Map`** (base class)
		> Base class for handling maps, providing interface for getting and saving maps.
	- **`StandardMaps`** (inherits from **`Map`**)
		> Calculates and saves standard deviation and mean maps across time for features.
	- **`StandardMapsPointWise`** (inherits from **`Map`**)
		> Calculates and saves mean and standard deviation maps point-wise across time, latitude and longitude.
	- **`MinMaxMaps`** (inherits from **`Map`**)
		> Calculates and saves minimum and maximum maps across time for features.
	- **`MinMaxMapsPointWise`** (inherits from **`Map`**)
		> Calculates and saves minimum and maximum maps point-wise across time, latitude and longitude.

- <a href="../Fires/_scalers/standard.py" target="_blank">standard.py</a>
	- **`StandardScaler`** (inherits from **`Scaler`**)
		> Scales data using mean and standard deviation (z-score normalization). Provides transform and inverse_transform methods for standardizing tensors.

##### Utilities

<p align="justify"> The <a href="../Fires/_utilities/" target="_blank">_utilities</a> folder contains various utility scripts: </p>

- <a href="../Fires/_utilities/callbacks.py" target="_blank">callbacks.py</a>: Custom callbacks for model training.

	- **`FabricBenchmark`**
		> A class that tracks training and validation metrics during model training. It saves metrics to a CSV file after each validation epoch on rank 0, maintaining a history of model performance over time.

	- **`FabricCheckpoint`**
		> A class that implements model checkpointing functionality. It monitors a specified metric (default: validation loss) and saves model checkpoints when improvements are detected, helping preserve the best model states during training.

- <a href="../Fires/_utilities/cli_args_checker.py" target="_blank">cli_args_checker.py</a>: Command-line arguments checker.

	- **`checker()`**
		> Function that parses command-line arguments and validates experiment configurations. Takes no parameters but expects a config file (-c) and experiment number (-nexp) from command line. Returns a tuple containing the experiment name and its configuration dictionary, or None if validation fails. Ensures that the specified experiment exists in the configuration file.




- <a href="../Fires/_utilities/cli_args_parser.py" target="_blank">cli_args_parser.py</a>: Command-line arguments parser.

	- **`CLIParser`** (inherits from **`argparse.ArgumentParser`**)
		> A versatile class for creating robust command-line interfaces. It provides methods for initializing parsers, adding arguments with validation, creating mutually exclusive groups, and handling parsing errors.

- <a href="../Fires/_utilities/configuration.py" target="_blank">configuration.py</a>: Configuration management.

	- **`load_global_config(dir_name, config_fname)`**: 
		> Loads configuration settings from a TOML file. Takes a directory path and filename as input, returns a dictionary-like object (munch.Munch) containing the configuration settings.

	- **`save_global_config(new_config, folder, filename)`**: 
		> Saves a new configuration dictionary to a TOML file. Takes the configuration dictionary, target folder, and filename as parameters to store configuration settings.

	- **`save_exp_config(exp_configuration, config_dir, dir_name, filepath)`**: 
		> Saves experiment-specific configurations to a TOML file. Handles both single experiment (dictionary) and multiple experiments (list of dictionaries) configurations. Creates necessary directories and saves configurations with appropriate naming.

- <a href="../Fires/_utilities/decorators.py" target="_blank">decorators.py</a>: Useful decorators.

	- **`export`**: 
		> Registers a function for export by potentially adding its name to the `__all__` attribute of its module. Ensures consistent module exports and public interface management.

	- **`debug`**: 
		> A decorator that logs function signatures, return values, and adds separator lines between calls. Can use a custom logger or default to standard output, making it useful for debugging and monitoring function behavior.

- <a href="../Fires/_utilities/logger_itwinai.py" target="_blank">logger_itwinai.py</a>: Logger for iTwinAI.

	- **`SimpleItwinaiLogger`** (inherits from **`Logger`**):
		> Basic logging implementation that saves artifacts to the filesystem and prints metrics to stdout. Supports different logging frequencies (epoch/batch) and selective logging on specific workers in distributed environments.

	- **`ItwinaiLightningLogger`** (inherits from **`lp.loggers.Logger`**):
		> PyTorch Lightning compatible logger that wraps SimpleItwinaiLogger. Provides integration with Lightning's training framework while maintaining iTwinAI's logging capabilities.

	- **`ItwinaiMLFlowLogger`** (inherits from **`lp.loggers.Logger`**):
		> Extends Lightning logging capabilities with MLflow integration. Enables experiment tracking, artifact storage, and metric logging in MLflow's tracking server.

	- **`ProvenanceLogger`** (inherits from **`lp.loggers.Logger`**):
		> Specialized logger for tracking model provenance information. Captures and stores the relationships between data, code, and results, optionally generating visual representations of the experiment workflow.

- <a href="../Fires/_utilities/logger.py" target="_blank">logger.py</a>: General logging utilities.

	- **`Logger`**:
		> Core logging class that initializes and manages log files. Configures logging with specific formats, levels, and directory structures. Supports multiple loggers with independent configurations for different components of the application.

- <a href="../Fires/_utilities/metrics.py" target="_blank">metrics.py</a>: Metrics calculation.

	- **`DiceLoss`** (inherits from **`nn.Module`**)
		> Loss function that calculates the Dice coefficient for binary segmentation tasks. It measures the overlap between predicted and target segmentation masks, providing a score between 0 (no overlap) and 1 (perfect overlap). Includes a smoothing factor to prevent division by zero.

	- **`TverskyLoss`** (inherits from **`nn.Module`**)
		> A generalization of the Dice Loss that allows for better handling of imbalanced data by introducing alpha and beta parameters to control the importance of false positives and false negatives. This makes it particularly useful for cases where one type of error is more critical than the other.

	- **`FocalLoss`** (inherits from **`nn.Module`**)
		> An improvement over standard binary cross-entropy that helps address class imbalance by down-weighting easy examples and focusing training on hard ones. Uses alpha and gamma parameters to modulate the contribution of each example to the loss based on how well it is already classified.
	
- <a href="../Fires/_utilities/utils_general.py" target="_blank">utils_general.py</a>: General utility functions.
	- **`check_backend()`**:
		> Function that determines the available PyTorch backend engine (MPS, CUDA, or CPU) and configures appropriate precision settings for matrix multiplications. Ensures optimal computation performance based on available hardware.

- <a href="../Fires/_utilities/utils_inference.py" target="_blank">utils_inference.py</a>: Utilities for inference.	

	- **`get_prov_image(run_name)`**: 
		> Retrieves a provenance graph image in SVG format for a specific MLflow run. Takes a run name as input and returns the path to the generated provenance image.

	- **`load_input_data(data_path, time_start, time_end)`**: 
		> Loads and preprocesses input data from a Zarr dataset within a specified time range. Handles land-sea masking and returns a subset of the data containing drivers and targets.

	- **`create_data_loader(data_path, run_name)`**: 
		> Creates a PyTorch DataLoader for model inference. Loads a scaler from MLflow artifacts and sets up a FireDataset with appropriate parameters for data loading.

	- **`compute_aggregated_data(data, other_data=None, operation="mean")`**: 
		> Performs statistical operations (mean or difference) on input data and aggregates results along latitude and longitude dimensions. Returns the processed data along with latitude and longitude aggregations.

	- **`up_and_lower_bounds(avg_value, std_value)`**: 
		> Calculates upper and lower confidence bounds using mean and standard deviation values. Returns a tuple of upper and lower bound arrays.

	- **`process_and_plot_data(data, label, lats, lons, model_name)`**: 
		> Processes data by computing statistics and generates visualization plots. Handles both xarray DataArrays and numpy arrays, creating maps with statistical overlays.

- <a href="../Fires/_utilities/utils_mlflow.py" target="_blank">utils_mlflow.py</a>: Utilities for MLflow integration.
	- **`load_model_from_mlflow_registry(model_name, version=1, tag=None)`**: 
		> Loads a model from MLflow's model registry using either a version number or tag. Downloads the model to a local path and returns it configured for the current device.

	- **`load_model_from_mlflow(run_name, scaler=True, provenance=False)`**: 
		> Retrieves a model from MLflow using a specific run name. Can optionally download associated scaler and provenance artifacts. Returns the model configured for the current device.

	- **`setup_mlflow_experiment()`**: 
		> Initializes MLflow tracking by configuring the tracking URI and experiment name from environment variables. Sets up the experiment for logging model metrics and artifacts.

- <a href="../Fires/_utilities/utils_model.py" target="_blank">utils_model.py</a>: Model utility functions.

	- **`seed_everything(seed)`**:
		> Sets random seeds across Python, NumPy, and PyTorch to ensure reproducible results. Takes a seed integer as input and configures random number generators for consistent behavior across executions.

- <a href="../Fires/_utilities/utils_trainer.py" target="_blank">utils_trainer.py</a>: Trainer utility functions.

	- **`get_trainer_loggers()`**:
		> Function that initializes and returns training loggers, including CSVLogger for metrics tracking. Optionally supports MLFlowLogger for experiment tracking (currently commented out). Returns a list of configured logger instances for use with PyTorch Lightning Trainer.

	- **`get_itwinai_loggers()`**:
		> Function that sets up iTwinAI specific loggers including MLFlowLogger for experiment tracking and Prov4MLLogger for provenance tracking. Returns a LoggersCollection instance containing configured iTwinAI loggers.

	- **`get_callbacks()`**:
		> Function that initializes and returns training callbacks including EarlyStopping and ModelCheckpoint. Optionally supports FabricBenchmark and FabricCheckpoint (currently commented out). Returns a list of callback instances for monitoring and managing the training process.

### Files

##### <a href="../Fires/__init__.py" target="_blank">\_\_init__.py</a>

<p align="justify"> The <a href="../Fires/__init__.py" target="_blank">__init__.py</a> file initializes the Fires module. It sets up the necessary imports and configurations required for the module to function correctly. </p>

##### <a href="../Fires/augmentation.py" target="_blank">augmentation.py</a>

<p align="justify"> The <a href="../Fires/augmentation.py" target="_blank">augmentation.py</a> script contains data augmentation techniques to improve model generalization: </p>

- **`rot180(data)`**
	> Function that rotates both input image (X) and target image (Y) by 90 degrees. Takes a tuple containing input and target images and returns the rotated versions while maintaining the tuple structure.

- **`left_right(data)`**
	> Function that performs horizontal flipping on both input image (X) and target image (Y). Takes a tuple containing input and target images and returns the horizontally flipped versions while maintaining the tuple structure.

- **`up_down(data)`**
	> Function that performs vertical flipping on both input image (X) and target image (Y). Takes a tuple containing input and target images and returns the vertically flipped versions while maintaining the tuple structure.



##### <a href="../Fires/datasets.py" target="_blank">datasets.py</a>

<p align="justify"> The <a href="../Fires/datasets.py" target="_blank">datasets.py</a> script contains functions and classes for handling datasets: </p>

- **`Dataset025`** (inherits from **`torch.utils.data.Dataset`**)
	> Class for handling datasets with 0.25° resolution. Provides methods for data loading, preprocessing, and batch generation optimized for this specific resolution.

- **`Dataset100`** (inherits from **`torch.utils.data.Dataset`**)
	> Class for handling datasets with 1° resolution. Similar to Dataset025 but optimized for lower resolution data handling and preprocessing.

- **`load_zarr`**
	> Function that handles loading data from Zarr format storage. Takes parameters for data path, time range, and specific variables to load. Returns preprocessed data ready for model consumption.

- **`FireDataset`** (inherits from **`torch.utils.data.Dataset`**)
	> Core dataset class that implements PyTorch's Dataset interface. Handles data loading, preprocessing, augmentation, and batch preparation for fire prediction tasks. Supports both training and inference modes.

##### <a href="../Fires/layers.py" target="_blank">layers.py</a>

<p align="justify"> The <a href="../Fires/layers.py" target="_blank">layers.py</a> script contains the implementation of various neural network layers. Below is a description of the functions and classes in the file: </p>

- **`VGGBlock`** (inherits from **`nn.Module`**)
	> VGG-style convolutional block that implements a sequence of two convolutional layers, each followed by batch normalization and ReLU activation. Optionally includes dropout for regularization. Used as a building block in various network architectures, particularly U-Net++.

##### <a href="../Fires/macros.py" target="_blank">macros.py</a>

<p align="justify"> The <a href="../Fires/macros.py" target="_blank">macros.py</a> script contains macro definitions used across the project.

- **`CONFIG`**
	> Path to the main configuration file containing project settings.

- **`TORCH_CFG`**
	> Configuration settings specific to PyTorch operations.

- **`BENCHMARK_HISTORY_CSV`**
	> Path to CSV file storing model benchmark history.

- **`CHECKPOINTS_DIR, CHECKPOINT_FNAME`**
	> Directory and filename for model checkpoints storage.

- **`CREDENTIALS_CFG`**
	> Path to credentials configuration file.

- **`CURR_DIR, DATA_DIR, DATA_PATH_ORIGINAL`**
	> Directory paths for current working directory, data storage, and original data.

- **`EXPS_DIR`**
	> Directory for storing experiment results.

- **`ITWINAI_DIR`**
	> Directory for iTwinAI-related files.

- **`LAST_MODEL`**
	> Path to most recently saved model.

- **`LOGS_DIR, LOG_FILE`**
	> Directory and file for logging outputs.

- **`LOSS_METRICS_HISTORY_CSV`**
	> CSV file tracking loss metrics history.

- **`NEW_DS_PATH`**
	> Path for new dataset storage.

- **`PROVENANCE_DIR`**
	> Directory for model provenance tracking.

- **`RUN_DIR`**
	> Directory for current run outputs.

- **`SAVE_SCALER_PATH`**
	> Path for saving data scalers.

- **`SCALER_DIR`**
	> Directory containing data scaling configurations.

- **`TRAINVAL_TIME_CSV`**
	> CSV file recording training and validation timing metrics.

##### <a href="../Fires/models.py" target="_blank">models.py</a>

<p align="justify"> The <a href="../Fires/models.py" target="_blank">models.py</a> script contains the implementation of various machine learning models. 

- **`BaseLightningModule`** (inherits from **`pl.LightningModule`**)
	> Base PyTorch Lightning module that implements common training functionality. Provides foundation for model implementations with standardized training, validation, and testing workflows.

- **`BaseUnetPlusPlus`** (inherits from **`BaseLightningModule`**)
	> Base class for U-Net++ architectures. Defines core parameters and structure for U-Net++ variants.

- **`BaseVGG`** (inherits from **`BaseLightningModule`**)
	> Base class for VGG-style networks. Provides foundational structure for VGG model variants.

- **`UnetPlusPlus`** (inherits from **`BaseUnetPlusPlus`**)
	> Implementation of the U-Net++ architecture. Enhances standard U-Net with nested skip connections for improved segmentation.

- **`VGG_V1, VGG_V2, VGG_V3`** (inherit from **`BaseVGG`**)
	> Three variants of VGG architecture with different depths and configurations. Each version offers specific modifications for different use cases.



##### <a href="../Fires/scalers.py" target="_blank">scalers.py</a>

<p align="justify"> The <a href="../Fires/scalers.py" target="_blank">scalers.py</a> script contains different scaling techniques. 

- **`Scaler`** (base class)
	> Base class for all data scalers. Defines the interface for scaling operations with methods for transform and inverse_transform.

- **`MinMaxScaler`** (inherits from **`Scaler`**)
	> Scales data to a fixed range [0,1] using minimum and maximum values. Handles both tensor and numpy array inputs.

- **`StandardScaler`** (inherits from **`Scaler`**)
	> Implements standardization by removing the mean and scaling to unit variance. Transforms data to have zero mean and unit standard deviation.

- **`Map`** (base class)
	> Base class for handling scaling maps. Provides methods for saving and loading scaling parameters across different dimensions.

- **`StandardMaps`** (inherits from **`Map`**)
	> Creates and manages maps for standardization across time dimension. Stores mean and standard deviation for each feature.

- **`StandardMapsPointWise`** (inherits from **`Map`**)
	> Creates point-wise standardization maps considering time, latitude, and longitude dimensions for more localized scaling.

- **`MinMaxMaps`** (inherits from **`Map`**)
	> Manages minimum and maximum value maps across time dimension for feature normalization.

- **`MinMaxMapsPointWise`** (inherits from **`Map`**)
	> Creates point-wise min-max maps considering all dimensions for local feature normalization.

##### <a href="../Fires/trainer.py" target="_blank">trainer.py</a>

<p align="justify"> The <a href="../Fires/trainer.py" target="_blank">trainer.py</a> script contains the implementation of a custom Fabric Trainer for distributed training. It provides a flexible training framework with support for gradient accumulation, checkpointing, validation, and MLflow logging. </p>

- **`FabricTrainer`**
	> A custom trainer class built on Lightning Fabric for distributed training. Features include:
	> - Configurable training parameters (epochs, steps, validation frequency)
	> - Gradient accumulation support
	> - Automatic checkpointing
	> - Validation loop integration
	> - MLflow metrics logging
	> - Distributed training support
	> - Progress bar with metrics display
	> - Learning rate scheduler integration
	> - Early stopping capability
	> - Model state management
	> - Batch limiting options for train and validation
	> - Callback system for monitoring training events

The trainer includes several methods for managing the training process:

- **`setup`**
	> Initializes the model, optimizer, and scheduler for training. Handles checkpoint loading and distributed setup.

- **`fit`**
	> Main training loop that manages epochs, validation, checkpoint saving, and MLflow logging.

- **`train_loop`**
	> Executes a single training epoch with gradient accumulation and optimizer steps.

- **`val_loop`**
	> Runs validation on the current model state with metric computation.

- **`training_step`**
	> Performs a single training step with forward pass, loss computation, and backward pass.

- **`step_scheduler`**
	> Handles learning rate scheduling based on monitoring metrics.

- **`save`**, **`load`**
	> Methods for checkpoint management and state restoration.

##### <a href="../Fires/utils.py" target="_blank">utils.py</a>

<p align="justify"> The <a href="../Fires/utils.py" target="_blank">utils.py</a> script contains various utility functions used throughout the project. It serves as a central location for commonly used functions, importing and re-exporting them from specialized utility modules for easier access. </p>

This file imports and exposes functionality from several utility modules:

- From **`plot_utils`**:
	- **`draw_features`**, **`highlight_ba`**, **`set_axis`**, **`draw_tropics_and_equator`**, **`plot_dataset_map`**
		> Functions for creating and enhancing geographical visualizations and plots.

- From **`cli_args_checker`**:
	- **`checker`**
		> Function for validating command-line arguments and experiment configurations.

- From **`cli_args_parser`**:
	- **`CLIParser`**
		> Class for parsing and validating command-line arguments.

- From **`logger`**:
	- **`Logger`**
		> Base logging functionality for the project.

- From **`logger_itwinai`**:
	- **`SimpleItwinaiLogger`**, **`ItwinaiLightningLogger`**, **`ProvenanceLogger`**
		> Specialized logging classes for iTwinAI integration.

- From **`metrics`**:
	- **`DiceLoss`**, **`FocalLoss`**, **`TverskyLoss`**
		> Loss functions for model training.

- From **`callbacks`**:
	- **`FabricBenchmark`**, **`FabricCheckpoint`**
		> Callback classes for monitoring and saving training progress.

- From **`configuration`**:
	- **`load_global_config`**, **`save_global_config`**, **`save_exp_config`**
		> Functions for managing configuration files.

- From **`decorators`**:
	- **`debug`**, **`export`**
		> Utility decorators for debugging and module exports.

- From **`utils_model`**:
	- **`seed_everything`**
		> Function for ensuring reproducibility across runs.

- From **`utils_general`**, **`utils_inference`**, **`utils_mlflow`**, **`utils_trainer`**:
	> Various utility functions for general operations, inference, MLflow integration, and training management.
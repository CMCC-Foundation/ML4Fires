import os
from typing import Dict, Optional, Tuple
import toml
import xarray as xr
import numpy as np
from datetime import datetime as dt

import joblib

# Settings the warnings to be ignored 
import warnings
warnings.filterwarnings('ignore')

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# MLFlow imports
import mlflow

# Itwinai imports
from itwinai.loggers import MLFlowLogger as Itwinai_MLFLogger, Prov4MLLogger, LoggersCollection

# Pytorch imports
import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import F1Score, FBetaScore, MatthewsCorrCoef, Precision, Recall, Accuracy, MeanSquaredError, ConcordanceCorrCoef

# Lightning imports
import lightning as L
import lightning.pytorch as lp
# from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.plugins.environments import MPIEnvironment
from lightning.pytorch.strategies.fsdp import FSDPStrategy
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_only


# ML4Fires imports
import Fires
from Fires._datasets.torch_dataset import FireDataset
from Fires._macros.macros import (
	CONFIG,
	DRIVERS as drivers,
	SEED,
	TARGETS as targets,
	TRN_YEARS as trn_years,
	VAL_YEARS as val_years,
	DATA_PATH_100KM,
	TORCH_CFG,
	CREDENTIALS_CFG,
	DATA_DIR,
	LOGS_DIR,
	NEW_DS_PATH,
	RUN_DIR,
	SCALER_DIR,
	PROVENANCE_DIR,
)
import Fires._models
from Fires._models.unet import Unet
from Fires._models.unetpp import UnetPlusPlus
import Fires._models.unetpp
from Fires._scalers.scaling_maps import StandardMapsPointWise, MinMaxMapsPointWise
from Fires._scalers.standard import StandardScaler
from Fires._scalers.minmax import MinMaxScaler
from Fires._utilities.cli_args_checker import checker
from Fires._utilities.cli_args_parser import CLIParser
from Fires._utilities.configuration import load_global_config
from Fires._utilities.decorators import debug
from Fires._utilities.logger import Logger as logger
from Fires._utilities.metrics import TverskyLoss, FocalLoss
from Fires._utilities.utils_general import check_backend
from Fires._utilities.utils_mlflow import setup_mlflow_experiment
from Fires._utilities.utils_trainer import get_trainer_loggers, get_itwinai_loggers, get_callbacks 
from Fires.trainer import FabricTrainer

# define logger
_log = logger(log_dir=LOGS_DIR).get_logger("Training_on_100km")

@debug(log=_log)
def init_fabric():
	"""
	Initializes the configuration arguments for the Fabric Trainer.

	This function checks the current backend (MPS, CUDA, or CPU) and configures
	the trainer settings based on the available hardware. It also retrieves logger
	and callback instances for monitoring and managing the training process.


	**Configurations**
	- `TORCH_CFG.model.strategy` (str): The training strategy to use (e.g., 'ddp', 'dp').
	- `TORCH_CFG.trainer.devices` (int): Number of devices (e.g., GPUs) to use.
	- `TORCH_CFG.trainer.num_nodes` (int): Number of nodes for distributed training.
	- `TORCH_CFG.trainer.precision` (int): Precision level for training (e.g., 16 for mixed precision).
	- `TORCH_CFG.trainer.plugins` (str): Additional plugins for training.
	
	**Notes**
	- If `backend` is set to 'mps' or 'cuda', the accelerator is set to 'gpu' and the strategy is defined from `TORCH_CFG`.
	- If `backend` is 'cpu', the accelerator is set to 'cpu' and strategy defaults to 'auto'.
	- Ensure that `TORCH_CFG` values are properly configured to match your training environment.

	**Returns**
		`fabric_args` (dict): 
			A dictionary of arguments (`fabric_args`) to be passed to the Fabric Trainer, 
			including settings for accelerator type, strategy, devices, precision, callbacks, 
			and loggers.
	
	**Example**
		```
		fabric_args = init_fabric()
		trainer = Trainer(**fabric_args)
		```


	"""
	# check backend if MPS, CUDA or CPU
	backend = check_backend()

	# get loggers for Fabric Trainer
	_loggers = get_trainer_loggers()

	# get callbacks for Fabric Trainer
	_callbacks = get_callbacks()
	
	# fabric args
	fabric_args = dict(
		accelerator='gpu' if backend in ['mps', 'cuda'] else 'cpu',
		strategy=eval(TORCH_CFG.model.strategy) if backend in ['mps', 'cuda'] else 'auto',
		devices=TORCH_CFG.trainer.devices,
		num_nodes=TORCH_CFG.trainer.num_nodes,
		precision=TORCH_CFG.trainer.precision,
		plugins=eval(TORCH_CFG.trainer.plugins),
		callbacks=_callbacks,
		loggers=_loggers,
	)

	return fabric_args


@debug(log=_log)
def get_trainer(fabric_args: Dict) -> FabricTrainer:
	"""
	Creates and configures a FabricTrainer instance for model training.

	This function checks the backend, sets up various training parameters 
	including callbacks, devices, logging, epochs, and more, and then 
	initializes the FabricTrainer.

	Returns
	-------
	FabricTrainer
		An instance of the FabricTrainer class configured with the specified training parameters.
	"""

	# define trainer args
	trainer_args = {
		# fabric args
		'fabric_args':fabric_args,
		# define number of epochs
		'max_epochs':TORCH_CFG.trainer.epochs,
		# define trainer accumulation steps
		'grad_accum_steps':TORCH_CFG.trainer.accumulation_steps,
		# set distributed sampler
		'use_distributed_sampler':eval(TORCH_CFG.trainer.use_distributed_sampler)
	}

	for k in trainer_args.keys():
		_log.info(f" | {k}:{trainer_args[k]}")

	# initialize trainer and its arguments
	trainer = FabricTrainer(**trainer_args)

	return trainer


@debug(log=_log)
def create_torch_datasets(data_source_path:str) -> Tuple[FireDataset, FireDataset]:
	"""
	Creates PyTorch datasets for training and validation from the provided data source.

	This function checks if the data source path exists, loads the training data, 
	applies standard scaling, and creates PyTorch datasets for training and validation.
 
	Parameters
	----------
	data_source_path : str
		The path to the data source for the 100km dataset.

	Returns
	-------
	Tuple[FireDataset, FireDataset]
		A tuple containing the training and validation datasets as PyTorch FireDataset objects.
	
	Raises
	------
	ValueError
		Check if the data source path exists, if it doesn't exist raise an error.
	"""

	if not os.path.exists(data_source_path):
		raise ValueError(f"Path to 100km dataset doesn't exists: {data_source_path}")
	
	# load training data
	data = xr.open_zarr(data_source_path)[drivers+targets]
	train_data = data.sel(time=data.time.dt.year.isin(trn_years)).load()
	
	# create standard scaler
	mean_std_args = dict(dim=['time','latitude', 'longitude'], skipna=True)
	mean_ds = train_data.mean(**mean_std_args)
	stdv_ds = train_data.std(**mean_std_args)
	x_scaler = StandardScaler(mean_ds=mean_ds, stdv_ds=stdv_ds, features=drivers)

	# define pytorch datasets for training and validation
	fire_ds_args = dict(src=data_source_path, drivers=drivers, targets=targets)
	trn_torch_ds = FireDataset(**fire_ds_args, years=trn_years, scalers=[x_scaler, None])
	val_torch_ds = FireDataset(**fire_ds_args, years=val_years, scalers=[x_scaler, None])

	return trn_torch_ds, val_torch_ds, x_scaler


@debug(log=_log)
def setup_model() -> Optional[Unet | UnetPlusPlus]:
	"""
	Initializes and configures the UNet model for training.

	This function sets up the model configuration, including input shape, 
	number of classes, depth, and activation function, and then creates an 
	instance of the UNet model. It also sets the loss function and initializes 
	the metrics for the model.

	Returns
	-------
	Unet
		An instance of the Unet class configured for training.
	"""
	
	# define model loss
	model.loss = eval(TORCH_CFG.model.loss)   #torch.nn.modules.loss.BCELoss()
	#model.loss = TverskyLoss(alpha=0.5, beta=0.5)

	# define metrics list
	_metrics = []

	# accuracy
	accuracy = Accuracy(task='binary')
	accuracy.name = "accuracy"
	_metrics.append(accuracy)

	# precision
	precision = Precision(task='binary')
	precision.name = "precision"
	_metrics.append(precision)

	# recall
	recall = Recall(task='binary')
	recall.name = "recall"
	_metrics.append(recall)

	# f1 score
	f1_score = F1Score(task='binary')
	f1_score.name = "f1_score"
	_metrics.append(f1_score)

	# f2 score
	f2_score = FBetaScore(task='binary', beta=float(2))
	f2_score.name = "f2_score"
	_metrics.append(f2_score)

	# mcc
	mcc = MatthewsCorrCoef(task='binary')
	mcc.name = "mcc"
	_metrics.append(mcc)

	all_metrics = False
	
	# define model metrics
	model.metrics = _metrics if all_metrics else []

	_log.info(f" | Model: \n\n {model}")
	
	return model


@debug(log=_log)
def get_lightning_trainer():
	"""
	Initializes and returns a PyTorch Lightning `Trainer` 
	configured with necessary loggers, callbacks, and settings.

	This function configures the Lightning Trainer with appropriate hardware 
	settings, loggers, callbacks, and a fixed seed for reproducibility. 
	It also incorporates a custom `itwinai_logger` collection for additional logging.

	**Returns**
		Trainer: A PyTorch Lightning Trainer instance configured for training.

	**Environment/Configuration**
	- `SEED` (int): Seed value for reproducibility.
	- `TORCH_CFG.trainer.precision` (int): Precision level (e.g., 16 for mixed precision).
	- `TORCH_CFG.model.strategy` (str): Model training strategy (e.g., 'ddp', 'dp').

	**Example**
		```
		trainer = get_lightning_trainer()
		trainer.fit(model)
		```
	
	**Notes**
	- Sets `accelerator` to 'gpu' if the backend is MPS or CUDA, otherwise defaults to 'cpu'.
	- For distributed training, uses `ddp` as the strategy when GPU support is detected, otherwise defaults to 'auto'.
	- `_loggers_collection` is assigned to `itwinai_logger` on the trainer for additional logging support.
	"""

	# check backend
	backend = check_backend()

	# get loggers for Pytorch Lightning Trainer
	_loggers = get_trainer_loggers()

	# get itwinai loggers
	_loggers_collection = get_itwinai_loggers()

	# get callbacks for Pytorch Lightning Trainer
	_callbacks = get_callbacks()

	# seed everything for reproducibility
	lp.seed_everything(seed=SEED, workers=True)

	# define lightining.pytorch.Trainer object
	_trainer=lp.Trainer(
		accelerator='gpu' if backend in ['mps', 'cuda'] else 'cpu',
		strategy=eval(TORCH_CFG.model.strategy) if backend in ['mps', 'cuda'] else 'auto',
		devices=TORCH_CFG.trainer.devices,
		num_nodes=TORCH_CFG.trainer.num_nodes,
		precision=TORCH_CFG.trainer.precision,
		logger=_loggers,
		callbacks=_callbacks,
		max_epochs=TORCH_CFG.trainer.epochs,
	)

	_trainer.itwinai_logger = _loggers_collection

	return _trainer


@debug(log=_log)
def main():
	"""
	Main function to execute the model training pipeline.

	This function orchestrates the entire training process by creating datasets, 
	initializing the trainer and model, setting up data loaders, and starting the 
	training process. It also logs the training progress and saves the final model 
	to disk.
	"""
		
	# create pytorch datasets for training and validation
	trn_torch_ds, val_torch_ds, x_scaler = create_torch_datasets(data_source_path=DATA_PATH_100KM)
	
	# define model
	model = setup_model()

	# load dataloader
	dloader_args = dict(batch_size=TORCH_CFG.trainer.batch_size, shuffle=True, drop_last=TORCH_CFG.trainer.drop_reminder, num_workers=TORCH_CFG.trainer.workers)
	train_loader = DataLoader(trn_torch_ds,	**dloader_args)
	valid_loader = DataLoader(val_torch_ds, **dloader_args)

	# get instance of Pytorch Lightning Trainer
	trainer = get_lightning_trainer()
	with trainer.itwinai_logger.start_logging(rank=trainer.global_rank):

		# get global rank
		global_rank = trainer.global_rank
		print(f" | Global rank {global_rank}")

		# fit the model
		trainer.fit(
			model=model,
			train_dataloaders=train_loader,
			val_dataloaders=valid_loader
		)

		# save the model to disk
		last_model = os.path.join(RUN_DIR,'last_model.pt')
		trainer.save_checkpoint(filepath=last_model)
		trainer.itwinai_logger.log(
				item=last_model,
				identifier="model_weights",
				kind='artifact'
		)

		# log model
		original_model = trainer.model # trainer.model.module
		original_model.cpu()
		trainer.itwinai_logger.log(
			item=original_model,
			identifier="last_model",
			kind='model'
		)

		
		#Log scaler
		scaler_file = os.path.join(RUN_DIR,'scaler.dump')
		joblib.dump(x_scaler, scaler_file) 
		trainer.itwinai_logger.log(
			item=scaler_file,
			identifier="scaler",
			kind='artifact'
		)

		#Log provenance
		trainer.itwinai_logger.log(
			item=None,
			identifier=None,
			kind="prov_documents")



@debug(log=_log)
def check_cli_args():
	"""
	Parses and validates command-line arguments for configuring a UNet model for training.

	This function displays the program name and description, then parses command-line 
	arguments related to the UNet model's configuration, such as the base filter dimension 
	and the activation function for the last layer. It ensures that all required arguments 
	are provided and sets default values if necessary. Based on the parsed arguments, it 
	constructs and returns a dictionary containing the model configuration.

	Returns
	-------
	dict
		A dictionary containing the UNet model configuration with the following keys:
		- 'input_shape': The shape of the input data (fixed as (180, 360, 7)).
		- 'base_filter_dim': The base filter dimension for the UNet model, as specified by the user.
		- 'activation': The activation function for the last layer, either Sigmoid or ReLU, 
		  based on the user's choice.
	"""

	PROGRAM_NAME = ""
	PROGRAM_DESCRIPTION = "The following script is designed to perform the training of a ML model that must predict Wildfires Burned Areas on global scale."

	options = [
		[('-bfd', '--base_filter_dim'), dict(type=int, default=32, help='Base filter dimension for Unet (default: 32)')],
		[('-afn', '--activation'), dict(type=str, choices=['S', 'R'], default='S', help='Activation function for the last layer: S - Sigmoid (default) | R - ReLU')],
		[('-mdl', '--model'), dict(type=str, choices=['unet', 'unetpp'], default='unet', help='Name of the model that must be trained: unet (default) | unetpp - UNet++')],
	]
	cli_parser = CLIParser(program_name=PROGRAM_NAME, description=PROGRAM_DESCRIPTION)
	cli_parser.add_arguments(parser=None, options=options)
	cli_args = cli_parser.parse_args()

	activation_fn = torch.nn.Sigmoid() if cli_args.activation == 'S' else torch.nn.ReLU()
	
	cli_base_filter_dim = cli_args.base_filter_dim

	model_config = {
		'input_shape':(180, 360, 7),
		'base_filter_dim':cli_base_filter_dim,
		'activation':activation_fn
	}

	cli_model_name = cli_args.model

	if cli_model_name == 'unet':
		model_class = Fires._models.unet.Unet
	elif cli_model_name == 'unetpp':
		model_class = Fires._models.unetpp.UnetPlusPlus
	else:
		raise ValueError(f"Model not supported: {cli_args.model}")
	
	# define run name as an environment variable
	os.environ['MLFLOW_RUN_NAME'] = f"LOCAL_{cli_model_name.upper()}_BCE_{cli_base_filter_dim}"
		
	return model_class, model_config


if __name__ == '__main__':

	# setup MLFlow experiment
	setup_mlflow_experiment()

	# check cli args
	model_class, model_config = check_cli_args()
	for k in model_config.keys():
		print(f"{k}: {model_config[k]}")
	print("\n\n")

	# get model class and configuration
	global model
	model = model_class(**model_config)
	print(f"Model: {model}")

	main()

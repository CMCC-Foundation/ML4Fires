import os
from typing import Tuple
import mlflow
import toml
import torch
import xarray as xr
import numpy as np
from datetime import datetime as dt

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data.distributed import DistributedSampler

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.plugins.environments import MPIEnvironment

from torchmetrics.regression import MeanSquaredError, ConcordanceCorrCoef
from torchmetrics.classification import Precision, Recall, F1Score, FBetaScore, MatthewsCorrCoef, ConfusionMatrix

import Fires
from Fires._datasets.torch_dataset import FireDataset
from Fires._macros.macros import (
	CONFIG,
	DRIVERS as drivers,
	TARGETS as targets,
	TRN_YEARS as trn_years,
	VAL_YEARS as val_years,
	DATA_PATH_100KM,
	TORCH_CFG,
	DISCORD_CFG,
	CHECKPOINTS_DIR,
	DATA_DIR,
	LOGS_DIR,
	NEW_DS_PATH,
	RUN_DIR,
	SCALER_DIR,
)

from Fires._models.unet import Unet
from Fires._scalers.scaling_maps import StandardMapsPointWise, MinMaxMapsPointWise
from Fires._scalers.standard import StandardScaler
from Fires._scalers.minmax import MinMaxScaler
from Fires._utilities.callbacks import DiscordBenchmark, FabricBenchmark, FabricCheckpoint
from Fires._utilities.cli_args_checker import checker
from Fires._utilities.cli_args_parser import CLIParser
from Fires._utilities.configuration import load_global_config

from Fires._utilities.decorators import debug
from Fires._utilities.logger import Logger as logger
from Fires._utilities.logger_itwinai import ItwinaiLightningLogger

from Fires._utilities.metrics import TverskyLoss, FocalLoss

from Fires.trainer import FabricTrainer

# define logger
_log = logger(log_dir=LOGS_DIR).get_logger("Training_on_100km")


os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'YOUR_EMAIL'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'YOUR_PASSWORD'
TRACKING_URI = 'https://mlflow.intertwin.fedcloud.eu/'


# Step 1: Set up MLflow
mlflow.set_tracking_uri(TRACKING_URI)  # Change this if needed
experiment_name = "ML4Fires_PyTorch_Experiment"
mlflow.set_experiment(experiment_name)

@debug(log=_log)
def check_backend() -> str:
	"""
	Determines the available backend engine for PyTorch computations.

	This function checks if the MPS (Metal Performance Shaders) or CUDA backends
	are available and sets the appropriate backend accordingly. If neither MPS 
	nor CUDA is available, it defaults to the CPU backend.

	Returns
	-------
	str
		The name of the backend to use for PyTorch computations ('mps', 'cuda', or 'cpu').
	"""

	# check MPS availability (Use case: code runs on Mac with M-chip and GPUs)
	if torch.backends.mps.is_available():
		backend: str = 'mps'

	# check CUDA availability (Use case: code runs on Supercomputer with GPUs)
	elif torch.cuda.is_available():
		backend: str = 'cuda'

	else:
		backend: str = 'cpu'

	if backend in ['mps', 'cuda']:
		# set matricial multiplication
		torch.set_float32_matmul_precision(TORCH_CFG.base.matmul_precision)

	_log.info(f" | {backend.upper()} available")

	return backend



@debug(log=_log)
def get_trainer(run) -> FabricTrainer:
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

	itwinai_logger = ItwinaiLightningLogger(savedir=os.path.join(LOGS_DIR, "ITWINAI"))

	mlf_logger = MLFlowLogger(
	 experiment_name=mlflow.get_experiment(run.info.experiment_id).name,
	 tracking_uri=mlflow.get_tracking_uri(),
	 log_model=True,
	)
	mlf_logger._run_id = run.info.run_id

	# check backend if MPS, CUDA or CPU
	# backend = check_backend()
	backend = 'cpu'

	# define today's date
	today = eval(CONFIG.utils.datetime.today)
	_log.info(f" | Today: {today}")

	# define csv log name
	csv_fname = f'{today}_csv_logs'
	_log.info(f" | CSV Filename: {csv_fname}")

	# define trainer args
	trainer_args = {
	 # set accelerator type
	 'accelerator': backend,
	 # define callbacks
	 'callbacks':[
	 DiscordBenchmark(webhook_url=DISCORD_CFG.hooks.webhook_gen, benchmark_csv=os.path.join(RUN_DIR, "fabric_benchmark.csv")),
	 FabricBenchmark(filename=os.path.join(RUN_DIR, "fabric_benchmark.csv")),
	 FabricCheckpoint(dst=CHECKPOINTS_DIR),
	 EarlyStopping('val_loss')
	 ],
	 # define number of devices (GPUs) that must be used
	 'devices': TORCH_CFG.trainer.devices,
	 # define csv logger
	 'loggers':[CSVLogger(root_dir=LOGS_DIR, name=csv_fname), itwinai_logger, mlf_logger],
	 # define number of epochs
	 'max_epochs':10, #TORCH_CFG.trainer.epochs,
	 # define number of nodes used on the cluster
	 'num_nodes':TORCH_CFG.trainer.num_nodes,
	 # define trainer accumulation steps
	 'grad_accum_steps':TORCH_CFG.trainer.accumulation_steps,
	 # define trainer precision
	 'precision':TORCH_CFG.trainer.precision,
	 # define MPI plugin
	 'plugins':eval(TORCH_CFG.trainer.plugins),
	 # init distribution strategy
	 'strategy': eval(TORCH_CFG.model.strategy) if backend in ['mps', 'cuda'] else 'auto',
	 # set distributed sampler
	 'use_distributed_sampler':eval(TORCH_CFG.trainer.use_distributed_sampler),
	}

	_log.info(f" | Trainer arguments: {trainer_args}")

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

	return trn_torch_ds, val_torch_ds



@debug(log=_log)
def get_model():
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

	# define model configuration
	# model_config = {
	# 	'input_shape':(180, 360, 7),
	# 	'base_filter_dim':32,
	# 	'activation':torch.nn.modules.activation.Sigmoid()
	# }

	# define model loss
	# model.loss = torch.nn.modules.loss.BCELoss()
	# model.loss = TverskyLoss(alpha=0.95, beta=0.7)
	model.loss = FocalLoss(alpha=0.25, gamma=2)

	# define model metrics

	precision = Precision(task='binary')
	precision.name = "precision"

	recall = Recall(task='binary')
	recall.name = "recall"

	f1_score = F1Score(task='binary')
	f1_score.name = "f1_score"

	f2_score = FBetaScore(task='binary', beta=float(2))
	f2_score.name = "f2_score"

	mcc = MatthewsCorrCoef(task='binary')
	mcc.name = "mcc"

	model.metrics = [precision, recall, f1_score, f2_score, mcc]

	_log.info(f" | Model: \n\n {model}")

	return model



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
	trn_torch_ds, val_torch_ds = create_torch_datasets(data_source_path=DATA_PATH_100KM)


	# define model
	model = get_model()

	# load dataloader
	dloader_args = dict(batch_size=TORCH_CFG.trainer.batch_size, shuffle=True, drop_last=TORCH_CFG.trainer.drop_reminder)
	train_loader = DataLoader(trn_torch_ds, **dloader_args)
	valid_loader = DataLoader(val_torch_ds, **dloader_args)

	for batch in train_loader:
		_, y = batch
		print(y)
		_log.info(f" | Min: {torch.min(y)} Max: {torch.max(y)}")
		break  # Stampa solo il primo batch

	# Automatically log params, metrics, and model
	mlflow.pytorch.autolog()

	with mlflow.start_run(run_name="ML4Fires") as run:

		# define trainer
		trainer = get_trainer(run=run)

		# setup the model and the optimizer
		trainer.setup(
		 model=model,
		 optimizer_cls=eval(TORCH_CFG.trainer.optim.cls),
		 optimizer_args=eval(TORCH_CFG.trainer.optim.args),
		 scheduler_cls=eval(TORCH_CFG.trainer.scheduler.cls),
		 scheduler_args=eval(TORCH_CFG.trainer.scheduler.args),
		 checkpoint=eval(TORCH_CFG.trainer.checkpoint.ckpt)
		)

		# fit the model
		trainer.fit(train_loader=train_loader, val_loader=valid_loader)

		# save the model to disk
		last_model = os.path.join(RUN_DIR,'last_model.pt')
		trainer.fabric.save(
		 path=last_model,
		 state={
		  'model':trainer.model,
		  'optimizer':trainer.optimizer,
		  'scheduler': trainer.scheduler_cfg
		 }
		)

		# log weights
		mlflow.log_artifact(last_model, artifact_path="model_weights")

		# log model
		original_model = trainer.model.module

		# Remove non-serializable attributes
		if hasattr(original_model, '_fabric'):
			del original_model._fabric
		if hasattr(original_model, 'comm'):
			del original_model.comm

		original_model.cpu()

		mlflow.pytorch.log_model(original_model, "last_model")

	# end MLFlow run
	mlflow.end_run()



@debug(log=_log)
def check_unet_args():
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
		[('-bfd', '--base_filter_dim'),
		 dict(type=int,
			  default=32,
			  help='Base filter dimension for Unet (default: 32)')],
		[('-afn', '--activation'),
		 dict(
			 type=str,
			 choices=['S', 'R'],
			 default='S',
			 help=
			 'Activation function for the last layer: S - Sigmoid (default) | R - ReLU'
		 )],
		[('-mdl', '--model'),
		 dict(
			 type=str,
			 choices=['unet', 'unetpp'],
			 default='unet',
			 help=
			 'Name of the model that must be trained: unet (default) | unetpp - UNet++'
		 )],
	]
	cli_parser = CLIParser(program_name=PROGRAM_NAME, description=PROGRAM_DESCRIPTION)
	cli_parser.add_arguments(parser=None, options=options)
	cli_args = cli_parser.parse_args()

	activation_fn = torch.nn.Sigmoid() if cli_args.activation == 'S' else torch.nn.ReLU()

	model_config = {
		'input_shape': (180, 360, 7),
		'base_filter_dim': cli_args.base_filter_dim,
		'activation': activation_fn
	}

	if cli_args.model == 'unet':
		model_class = Fires._models.unet.Unet
	elif cli_args.model == 'unetpp':
		model_class = Fires._models.unetpp.UnetPlusPlus
	else:
		raise ValueError(f"Model not supported: {cli_args.model}")

	return model_class, model_config



if __name__ == '__main__':

	model_class, model_config = check_unet_args()
	for k in model_config.keys():
		print(f"{k}: {model_config[k]}")
	print("\n\n")

	global model
	model = model_class(**model_config)
	print(f"Model: {model}")

	main()

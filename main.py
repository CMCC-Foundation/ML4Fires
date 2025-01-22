# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 					Copyright 2024 - CMCC Foundation						
#																			
# Site: 			https://www.cmcc.it										
# CMCC Institute:	IESP (Institute for Earth System Predictions)
# CMCC Division:	ASC (Advanced Scientific Computing)						
# Author:			Emanuele Donno											
# Email:			emanuele.donno@cmcc.it									
# 																			
# Licensed under the Apache License, Version 2.0 (the "License");			
# you may not use this file except in compliance with the License.			
# You may obtain a copy of the License at									
#																			
#				https://www.apache.org/licenses/LICENSE-2.0					
#																			
# Unless required by applicable law or agreed to in writing, software		
# distributed under the License is distributed on an "AS IS" BASIS,			
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.	
# See the License for the specific language governing permissions and		
# limitations under the License.											
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import toml
import torch
import xarray as xr
from datetime import datetime as dt

import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data.distributed import DistributedSampler

from lightning.pytorch.callbacks import EarlyStopping
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.plugins.environments import MPIEnvironment

from torchmetrics.regression import MeanSquaredError

import Fires
from Fires._datasets.dataset_zarr import Dataset025, load_zarr
from Fires._datasets.torch_dataset import FireDataset
from Fires._macros.macros import (
	CONFIG,
	TORCH_CFG,
	CHECKPOINTS_DIR,
	DATA_DIR,
	LOGS_DIR,
	NEW_DS_PATH,
	RUN_DIR,
	SCALER_DIR,
)

from Fires._scalers.scaling_maps import StandardMapsPointWise, MinMaxMapsPointWise
from Fires._scalers.standard import StandardScaler
from Fires._scalers.minmax import MinMaxScaler
from Fires._utilities.callbacks import FabricBenchmark, FabricCheckpoint
from Fires._utilities.cli_args_checker import checker
from Fires._utilities.configuration import load_global_config
from Fires._utilities.logger import Logger as logger


from Fires.trainer import FabricTrainer

# define logger
_log = logger(log_dir=LOGS_DIR).get_logger("Workflow")

# current experiment configuration dict
current_experiment = dict(
	features = dict(),
	dataset = dict(),
	model = dict(),
	trainer = dict(),
	scalers = dict(),
)
_log.info(f"Define a dictionary to store current experiment configuration: \n {current_experiment}")

# define features
features = CONFIG.data.configs.config_fcci
drivers = features[:-1]
targets = [features[-1]]
# set drivers and targets into current experiment dictionary
current_experiment['features']['drivers'] = drivers
current_experiment['features']['targets'] = targets

_log.info(f"List of drivers: ")
for d in drivers: _log.info(f" - {d}")

_log.info(f"List of targets: ")
for t in targets: _log.info(f" - {t}")

# define list of years
trn_years = list(eval(CONFIG.data.features.training_years_range))
val_years = list(eval(CONFIG.data.features.validation_years_range))
current_experiment['dataset']['trn_years'] = trn_years
current_experiment['dataset']['val_years'] = val_years

_log.info(f"Training: {trn_years[0]}, ..., {trn_years[-1]}")
_log.info(f"Validation: {val_years[0]}, ..., {val_years[-1]}")

_log.info(f"Creating dataset zarr...")

# create Dataset
Dataset025()

_log.info(f"Dataset zarr has been created")

# define path to new dataset in zarr format
name = targets[0].split('_')[0].lower()
new_path = NEW_DS_PATH(name=name)
current_experiment['dataset']['path_to_zarr'] = new_path
print(f"\n Path to dataset: {new_path} \n")

_log.info(f"Path to new dataset: {new_path}")

# create mean and standard deviation maps
standard_point_maps = StandardMapsPointWise(features=features, years=trn_years, data_filepath=new_path, store_dir=SCALER_DIR)
mean_ds, stdv_ds = standard_point_maps.get_maps()
print(f" Mean data \n {mean_ds} \n Stdv data \n {stdv_ds} \n")

_log.info(f"\n Mean data \n {mean_ds} \n Stdv data \n {stdv_ds} \n")

# create dictionary with all NetCDF4 filepaths
current_experiment['scalers']['paths'] = dict()
print("Maps dict entries:\n")
_log.info(f"Maps dict entries: ")
for file in os.listdir(SCALER_DIR):
	if str(file).endswith('_map.nc'):
		filepath = os.path.join(SCALER_DIR, str(file))
		key = str(file).split('.')[0]
		print(f"- {key}: {filepath}\n")
		_log.info(f" - {key}: {filepath}")

		current_experiment['scalers']['paths'][key] = filepath

# ---- CHECK CLI ARGUMENTS -------------------------------------------------------------------------

_log.info(f"Start checking arguments...")

# check CLI args
checked_args = checker()

_log.info(f"Arguments checked")

_log.info(f"Experiment dir: {RUN_DIR}")

if checked_args:
	exp_name, exp_cfg = checked_args
	print(f"Experiment {exp_name}")

	_log.info(f"Experiment {exp_name}")
	
# ---- DEFINE SCALER -------------------------------------------------------------------------------

# define scaler for drivers
x_scaler = StandardScaler(mean_ds=mean_ds, stdv_ds=stdv_ds, features=drivers)
current_experiment['scalers']['cls'] = str(StandardScaler)

current_experiment['dataset']['torch'] = dict()
current_experiment['dataset']['torch']['args'] = dict()

# ---- DEFINE TORCH DATASET ------------------------------------------------------------------------

# fire dataset arguments
fire_ds_args = dict(src=new_path, drivers=drivers, targets=targets)
current_experiment['dataset']['torch']['args'] = fire_ds_args

# define pytorch datasets for training and validation
trn_torch_ds = FireDataset(**fire_ds_args, years=trn_years, scalers=[x_scaler, None])
val_torch_ds = FireDataset(**fire_ds_args, years=val_years, scalers=[x_scaler, None])
current_experiment['dataset']['torch']['cls'] = str(FireDataset)

# ---- DEFINE TRAINER ARGUMENTS --------------------------------------------------------------------

current_experiment['trainer']['args'] = dict()

# check GPUs availability
cuda_availability:bool = eval(TORCH_CFG.base.cuda_availability)
current_experiment['trainer']['cuda_availability'] = cuda_availability

# set device type
device:str = 'cuda' if cuda_availability else 'cpu'
current_experiment['trainer']['device'] = device

# set matricial multiplication precision
if cuda_availability: torch.set_float32_matmul_precision(TORCH_CFG.base.matmul_precision)
current_experiment['trainer']['matmul_precision'] = TORCH_CFG.base.matmul_precision
print(f" CUDA available: {cuda_availability}\n Device: {device.upper()}")

_log.info(f" CUDA available: {cuda_availability}\t Device: {device.upper()}")

# set accelerator
accelerator:str = 'cuda' if cuda_availability else 'cpu'
current_experiment['trainer']['args']['accelerator'] = accelerator
print(f" Accelerator: {accelerator.upper()}")

_log.info(f" Accelerator: {accelerator.upper()}")

# define trainer accumulation steps
accumulation_steps = TORCH_CFG.trainer.accumulation_steps
current_experiment['trainer']['args']['grad_accum_steps'] = accumulation_steps

# define callbacks
callbacks = [
	FabricBenchmark(filename=os.path.join(RUN_DIR, "fabric_benchmark.csv")),
	FabricCheckpoint(dst=CHECKPOINTS_DIR),
	EarlyStopping('val_loss')
]	

# define number of devices (GPUs) that must be used
devices = TORCH_CFG.trainer.devices
current_experiment['trainer']['args']['devices'] = devices

# define number of epochs
epochs = TORCH_CFG.trainer.epochs
current_experiment['trainer']['args']['max_epochs'] = epochs

# define today's date
today = eval(CONFIG.utils.datetime.today)

# define csv log name
csv_fname = f'{today}_csv_logs'

# define csv logger
loggers = CSVLogger(root_dir=LOGS_DIR, name=csv_fname)
current_experiment['trainer']['args']['loggers_cls'] = str(CSVLogger)
current_experiment['trainer']['args']['loggers_root_dir'] = LOGS_DIR
current_experiment['trainer']['args']['loggers_name'] = csv_fname

# define number of nodes used on the cluster
num_nodes = TORCH_CFG.trainer.num_nodes
current_experiment['trainer']['args']['num_nodes'] = num_nodes

# define trainer precision
precision = TORCH_CFG.trainer.precision
current_experiment['trainer']['args']['precision'] = precision

# define MPI plugin
plugins = eval(TORCH_CFG.trainer.plugins)
current_experiment['trainer']['args']['plugins'] = TORCH_CFG.trainer.plugins

# init distribution strategy
strategy = eval(TORCH_CFG.model.strategy) if accelerator == 'cuda' else 'auto'
print(f" Strategy: {strategy}")

# set distributed sampler
use_distributed_sampler = eval(TORCH_CFG.trainer.use_distributed_sampler)

# ---- DEFINE TRAINER ------------------------------------------------------------------------------

# initialize trainer and its arguments
trainer = FabricTrainer(
	accelerator=accelerator,
	callbacks=callbacks,
	devices=devices,
	loggers=loggers,
	max_epochs=epochs,
	num_nodes=num_nodes,
	grad_accum_steps=accumulation_steps,
	precision=precision,
	plugins=plugins,
	strategy=strategy,
	use_distributed_sampler=use_distributed_sampler
)
current_experiment['trainer']['cls'] = str(FabricTrainer)


# store parallel execution variables
p_variables = dict(
	world_size = trainer.world_size,
	node_rank = trainer.node_rank,
	global_rank = trainer.global_rank,
	local_rank = trainer.local_rank
)

# log
_log.info(f"Logger initialized. Starting the execution")
for key in p_variables:
	_log.info(f"   {key.capitalize().replace('_', ' ')}  :  {p_variables[key]}")

# --------------------------------------------------------------------------------------------------

# define PyTorch model
# TODO implement the model selection from argument parser
chosen_model = 'unetpp'
# get model configuration
model_cfg = TORCH_CFG.model
# get chosen model configuration
chosen_model_cfg = model_cfg[chosen_model]
# get model class
mdl_cls = eval(chosen_model_cfg.cls)

# get model args
mdl_args = chosen_model_cfg.args
print("Model args: \n", mdl_args, "\n")

current_experiment['model']['cls'] = chosen_model_cfg.cls
current_experiment['model']['args'] = dict()
for key in chosen_model_cfg.args.keys():
	current_experiment['model']['args'][key] = chosen_model_cfg.args[key]

if checked_args:
	base_filter_dim = exp_cfg.base_filter_dim
	# update model base_filter_dim argument with the new value
	mdl_args['base_filter_dim'] = base_filter_dim
	# update current experiment dictionary
	current_experiment['model']['args']['base_filter_dim'] = base_filter_dim

	# get activation function
	actv_str = exp_cfg.activation_cls.split("'")[-2]
	activation = eval(actv_str)()

	# update current experiment dictionary
	current_experiment['model']['args']['activation'] = actv_str+'()'
	print("CHECKED \n Model args: \n", mdl_args, "\n")
	# define model
	model = mdl_cls(**mdl_args, activation=activation)

else:
	print("NOT CHECKED \n Model args: \n", mdl_args, "\n")
	# define model
	model = mdl_cls(**mdl_args)
	

# define model
# model = mdl_cls(**mdl_args)

# define model loss
loss_str = TORCH_CFG.model.loss
if checked_args:
	loss_str = exp_cfg.loss_cls.split("'")[-2]+'()'
loss = eval(loss_str)

# add loss to model
model.loss = loss
# update current experiment dictionary
current_experiment['model']['loss'] = loss_str

# define model metrics
model.metrics = eval(TORCH_CFG.model.metrics)
print(model)
# update current experiment dictionary
current_experiment['model']['metrics'] = TORCH_CFG.model.metrics

# load dataloader
batch_size = TORCH_CFG.trainer.batch_size
drop_reminder=TORCH_CFG.trainer.drop_reminder
train_loader = DataLoader(trn_torch_ds,	batch_size=batch_size, shuffle=True, drop_last=drop_reminder)
valid_loader = DataLoader(val_torch_ds, batch_size=batch_size, shuffle=True, drop_last=drop_reminder)

# update current experiment dictionary
current_experiment['trainer']['batch_size'] = batch_size
current_experiment['trainer']['drop_reminder'] = drop_reminder
current_experiment['trainer']['data_loader_cls'] = str(DataLoader)

current_experiment['trainer']['optim'] = dict()
current_experiment['trainer']['optim']['cls'] = TORCH_CFG.trainer.optim.cls
current_experiment['trainer']['optim']['args'] = eval(TORCH_CFG.trainer.optim.args)

current_experiment['trainer']['scheduler'] = dict()
current_experiment['trainer']['scheduler']['cls'] = TORCH_CFG.trainer.scheduler.cls
current_experiment['trainer']['scheduler']['args'] = eval(TORCH_CFG.trainer.scheduler.args)

current_experiment['trainer']['checkpoint'] = dict()
current_experiment['trainer']['checkpoint']['ckpt'] = TORCH_CFG.trainer.checkpoint.ckpt


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

# log
_log.info(f'Model trained')

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

current_experiment['exp_dir'] = RUN_DIR
current_experiment['model']['last_model'] = last_model

filename = exp_name if checked_args else 'experiment'
current_experiment['exp_name'] = filename

print(current_experiment)
_log.info(f"Current experiment dictionary: \n {current_experiment}")

path = os.path.join(RUN_DIR, f'{filename}.toml')
with open(path , "w") as file:
	if type(current_experiment) == dict:
		toml.dump(current_experiment, file)

# log
print(f'Program completed')
_log.info(f'Program completed')

# close program
# exit(1)

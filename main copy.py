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
	DRIVERS,
	TARGETS,
	TRN_YEARS,
	VAL_YEARS,
	TORCH_CFG,
	DISCORD_CFG,
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
from Fires._utilities.callbacks import DiscordBenchmark, FabricBenchmark, FabricCheckpoint
from Fires._utilities.cli_args_checker import checker
from Fires._utilities.configuration import load_global_config
from Fires._utilities.logger import Logger as logger

from Fires.trainer import FabricTrainer

# define logger
_log = logger(log_dir=LOGS_DIR).get_logger("Workflow")

# define features
features = DRIVERS+TARGETS

# _log.info(f"List of drivers: ")
# for d in _drivers: _log.info(f" - {d}")

# _log.info(f"List of targets: ")
# for t in _targets: _log.info(f" - {t}")

# define list of years
trn_years, val_years = TRN_YEARS, VAL_YEARS

# _log.info(f"Training: {trn_years[0]}, ..., {trn_years[-1]}")
# _log.info(f"Validation: {val_years[0]}, ..., {val_years[-1]}")

_log.info(f"Creating dataset zarr...")
# create Dataset
Dataset025()
_log.info(f"Dataset zarr has been created")

# define path to new dataset in zarr format
name = TARGETS[0].split('_')[0].lower()
new_path = NEW_DS_PATH(name=name)

# create mean and standard deviation maps
standard_point_maps = StandardMapsPointWise(features=features, years=trn_years, data_filepath=new_path, store_dir=SCALER_DIR)
mean_ds, stdv_ds = standard_point_maps.get_maps()
_log.info(f"\n Mean data \n {mean_ds} \n Stdv data \n {stdv_ds} \n")


_log.info(f"Maps dict entries: ")
for file in os.listdir(SCALER_DIR):
	if str(file).endswith('_map.nc'):
		filepath = os.path.join(SCALER_DIR, str(file))
		key = str(file).split('.')[0]
		print(f"- {key}: {filepath}\n")
		_log.info(f" - {key}: {filepath}")

		# current_experiment['scalers']['paths'][key] = filepath

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
x_scaler = StandardScaler(mean_ds=mean_ds, stdv_ds=stdv_ds, features=DRIVERS)


# ---- DEFINE TORCH DATASET ------------------------------------------------------------------------

# fire dataset arguments
fire_ds_args = dict(src=new_path, drivers=DRIVERS, targets=TARGETS)

# define pytorch datasets for training and validation
trn_torch_ds = FireDataset(**fire_ds_args, years=trn_years, scalers=[x_scaler, None])
val_torch_ds = FireDataset(**fire_ds_args, years=val_years, scalers=[x_scaler, None])

# ---- DEFINE TRAINER ARGUMENTS --------------------------------------------------------------------

cuda_availability:bool = eval(TORCH_CFG.base.cuda_availability)										# check GPUs availability
device:str = 'cuda' if cuda_availability else 'cpu'													# set device type
if cuda_availability: torch.set_float32_matmul_precision(TORCH_CFG.base.matmul_precision)			# set matricial multiplication precision
_log.info(f" CUDA available: {cuda_availability}\t Device: {device.upper()}")
accelerator:str = 'cuda' if cuda_availability else 'cpu'											# set accelerator
_log.info(f" Accelerator: {accelerator.upper()}")
accumulation_steps = TORCH_CFG.trainer.accumulation_steps											# define trainer accumulation steps
callbacks = [																						# define callbacks
	DiscordBenchmark(webhook_url=DISCORD_CFG.hooks.webhook_gen, benchmark_csv=os.path.join(RUN_DIR, "fabric_benchmark.csv")),
	FabricBenchmark(filename=os.path.join(RUN_DIR, "fabric_benchmark.csv")),
	FabricCheckpoint(dst=CHECKPOINTS_DIR),
	EarlyStopping('val_loss')
]	
_log.info(f"Discord configuration file:")
for key in DISCORD_CFG.keys(): _log.info(f"{key} : {DISCORD_CFG[key]}")								# define discord configuration file
devices = TORCH_CFG.trainer.devices																	# define number of devices (GPUs) that must be used
epochs = TORCH_CFG.trainer.epochs																	# define number of epochs
today = eval(CONFIG.utils.datetime.today)															# define today's date
csv_fname = f'{today}_csv_logs'																		# define csv log name
loggers = CSVLogger(root_dir=LOGS_DIR, name=csv_fname)												# define csv logger
num_nodes = TORCH_CFG.trainer.num_nodes																# define number of nodes used on the cluster
precision = TORCH_CFG.trainer.precision																# define trainer precision
plugins = eval(TORCH_CFG.trainer.plugins)															# define MPI plugin
strategy = eval(TORCH_CFG.model.strategy) if accelerator == 'cuda' else 'auto'						# init distribution strategy
_log.info(f" Strategy: {strategy}")
use_distributed_sampler = eval(TORCH_CFG.trainer.use_distributed_sampler)							# set distributed sampler


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


print("Model args: \n", mdl_args, "\n")
if checked_args:
	mdl_args['base_filter_dim'] = exp_cfg.base_filter_dim # update model base_filter_dim argument with the new value
	actv_str = exp_cfg.activation_cls.split("'")[-2] # get activation function
	activation = eval(actv_str)()
	model = mdl_cls(**mdl_args, activation=activation) # define model

else: model = mdl_cls(**mdl_args) # define model

# define model
# model = mdl_cls(**mdl_args)

# define model loss
loss_str = TORCH_CFG.model.loss
if checked_args:
	loss_str = exp_cfg.loss_cls.split("'")[-2]+'()'
loss = eval(loss_str)

# add loss to model
model.loss = loss

# define model metrics
model.metrics = eval(TORCH_CFG.model.metrics)
print(model)

# load dataloader
batch_size = TORCH_CFG.trainer.batch_size
drop_reminder=TORCH_CFG.trainer.drop_reminder
train_loader = DataLoader(trn_torch_ds,	batch_size=batch_size, shuffle=True, drop_last=drop_reminder)
valid_loader = DataLoader(val_torch_ds, batch_size=batch_size, shuffle=True, drop_last=drop_reminder)


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
trainer.fabric.save(path=last_model, state={'model':trainer.model, 'optimizer':trainer.optimizer, 'scheduler': trainer.scheduler_cfg})

# log
print(f'Program completed')
_log.info(f'Program completed')

# close program
# exit(1)

'''
# current_experiment = dict(
# 	features = dict(),
# 	dataset = dict(),
# 	model = dict(),
# 	trainer = dict(),
# 	scalers = dict(),
# )
# current_experiment['features']['drivers'] = drivers
# current_experiment['features']['targets'] = targets
# current_experiment['dataset']['path_to_zarr'] = new_path
# current_experiment['dataset']['torch'] = dict()
# current_experiment['dataset']['torch']['args'] = dict()
# current_experiment['dataset']['torch']['args'] = fire_ds_args
# current_experiment['dataset']['torch']['cls'] = str(FireDataset)
# current_experiment['dataset']['trn_years'] = trn_years
# current_experiment['dataset']['val_years'] = val_years
# current_experiment['model']['cls'] = chosen_model_cfg.cls
# current_experiment['model']['args'] = dict()
# for key in chosen_model_cfg.args.keys():
# 	current_experiment['model']['args'][key] = chosen_model_cfg.args[key]
# if checked_args:
# 	base_filter_dim = exp_cfg.base_filter_dim
# 	mdl_args['base_filter_dim'] = base_filter_dim # update model base_filter_dim argument with the new value
# 	actv_str = exp_cfg.activation_cls.split("'")[-2] # get activation function
# 	activation = eval(actv_str)()
# 	current_experiment['model']['args']['base_filter_dim'] = base_filter_dim
# 	current_experiment['model']['args']['activation'] = actv_str+'()'
# current_experiment['model']['loss'] = loss_str
# current_experiment['model']['metrics'] = TORCH_CFG.model.metrics
# current_experiment['model']['last_model'] = last_model
# current_experiment['scalers']['paths'] = dict()
# current_experiment['scalers']['cls'] = str(StandardScaler)
# current_experiment['trainer']['args'] = dict()
# current_experiment['trainer']['args']['accelerator'] = accelerator
# current_experiment['trainer']['args']['grad_accum_steps'] = accumulation_steps
# current_experiment['trainer']['args']['devices'] = devices
# current_experiment['trainer']['args']['max_epochs'] = epochs
# current_experiment['trainer']['args']['loggers_cls'] = str(CSVLogger)
# current_experiment['trainer']['args']['loggers_root_dir'] = LOG_DIR
# current_experiment['trainer']['args']['loggers_name'] = csv_fname
# current_experiment['trainer']['args']['num_nodes'] = num_nodes
# current_experiment['trainer']['args']['precision'] = precision
# current_experiment['trainer']['args']['plugins'] = TORCH_CFG.trainer.plugins
# current_experiment['trainer']['batch_size'] = batch_size
# current_experiment['trainer']['checkpoint'] = dict()
# current_experiment['trainer']['checkpoint']['ckpt'] = TORCH_CFG.trainer.checkpoint.ckpt
# current_experiment['trainer']['cls'] = str(FabricTrainer)
# current_experiment['trainer']['cuda_availability'] = cuda_availability
# current_experiment['trainer']['data_loader_cls'] = str(DataLoader)
# current_experiment['trainer']['device'] = device
# current_experiment['trainer']['drop_reminder'] = drop_reminder
# current_experiment['trainer']['matmul_precision'] = TORCH_CFG.base.matmul_precision
# current_experiment['trainer']['optim'] = dict()
# current_experiment['trainer']['optim']['cls'] = TORCH_CFG.trainer.optim.cls
# current_experiment['trainer']['optim']['args'] = eval(TORCH_CFG.trainer.optim.args)
# current_experiment['trainer']['scheduler'] = dict()
# current_experiment['trainer']['scheduler']['cls'] = TORCH_CFG.trainer.scheduler.cls
# current_experiment['trainer']['scheduler']['args'] = eval(TORCH_CFG.trainer.scheduler.args)
# current_experiment['exp_dir'] = RUN_DIR
# filename = exp_name if checked_args else 'experiment'
# current_experiment['exp_name'] = filename
# _log.info(f"Current experiment dictionary: \n {current_experiment}")
# path = os.path.join(RUN_DIR, f'{filename}.toml')
# with open(path , "w") as file:
# 	if type(current_experiment) == dict:
# 		toml.dump(current_experiment, file)
'''

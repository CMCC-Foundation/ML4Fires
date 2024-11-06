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
import sys
from os import path
from datetime import datetime as dt
from Fires._utilities.configuration import load_global_config

def check_if_exists(directory:str):
	if not os.path.exists(directory):
		raise ValueError(f"Path {directory} doesn't exist")

def add_to_syspath(directory:str):
	if directory not in sys.path:
		sys.path.append(directory)

# _path_list = ['.', '..']
# for p in _path_list: sys.path.append(p) if p not in sys.path else print(f"Path: {p} is in sys.path")


# define current working directory
_CWD = os.getcwd()
CURR_DIR = os.path.dirname(_CWD) if _CWD.split('/')[-1] != 'ML4Fires' else _CWD

# FIXME Continue to improve the code
CONFIG_DIR = os.path.join(CURR_DIR, 'config')
# check if the directory exists
check_if_exists(directory=CONFIG_DIR)
# add directory to system path
add_to_syspath(directory=CONFIG_DIR)

# get global configuration file
CONFIG = load_global_config(dir_name=CONFIG_DIR)
# set pytorch configuration file
TORCH_CFG = load_global_config(dir_name=CONFIG_DIR, config_fname=CONFIG.toml.torch_fname)
# set discord configuration file
DISCORD_CFG = load_global_config(dir_name=CONFIG_DIR, config_fname=CONFIG.toml.logs.discord_fname)
# set credentials configuration file
CREDENTIALS_CFG = load_global_config(dir_name=CONFIG_DIR, config_fname=CONFIG.toml.credentials_fname)

# define seed for reproducibility
SEED = CONFIG.utils.seed

# define features
DRIVERS = CONFIG.data.features.drivers
TARGETS = CONFIG.data.features.targets
LS_MASK = CONFIG.data.features.landsea_mask

# define training, validation and testing years
TRN_YEARS = eval(CONFIG.data.features.trn_years_list)
VAL_YEARS = eval(CONFIG.data.features.val_years_list)
TST_YEARS = eval(CONFIG.data.features.tst_years_list)

# max hectares values
MAX_HECTARES_025KM = pow((111/4), 2) * 100
MAX_HECTARES_100KM = pow((111), 2) * 100


# todays date
_today = eval(CONFIG.utils.datetime.today)

# define log dir
LOGS_DIR = eval(CONFIG.dir.LOGS_DIR)
os.makedirs(LOGS_DIR, exist_ok=True)

# define data dir
DATA_DIR = CONFIG.dir.DATA_DIR 
os.makedirs(DATA_DIR, exist_ok=True)

# define scaler dir
SCALER_DIR = eval(CONFIG.dir.SCALER_DIR)
os.makedirs(SCALER_DIR, exist_ok=True)

# define experiments dir
EXPS_DIR = eval(CONFIG.dir.EXPS_DIR)
os.makedirs(EXPS_DIR, exist_ok=True)

# define ITWINAI dir
ITWINAI_DIR = os.path.join(LOGS_DIR, "ITWINAI")
os.makedirs(ITWINAI_DIR, exist_ok=True)

# define Provenance dir
PROVENANCE_DIR = os.path.join(ITWINAI_DIR, "provenance")
os.makedirs(PROVENANCE_DIR, exist_ok=True)

# define lambda function to get experiment dir
EXPS_PTH = lambda dirname : os.path.join(EXPS_DIR, dirname)

#
# Pytorch configurations
#

# define pytorch
RUN_DIR = eval(TORCH_CFG.model.dir.RUN_DIR)
os.makedirs(RUN_DIR, exist_ok=True)

# define checkpoint dir
CHECKPOINTS_DIR = eval(TORCH_CFG.model.dir.CHECKPOINTS_DIR)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)	

#
# Filepaths to data sources
#

DATA_PATH_ORIGINAL = eval(CONFIG.data.files.DATA_PATH_ORIGINAL)
DATA_PATH_025KM = eval(CONFIG.data.files.DATA_PATH_025KM)
DATA_PATH_100KM = eval(CONFIG.data.files.DATA_PATH_100KM)

# DATA_NEW_FPATH = eval(CONFIG.data.files.DATA_NEW_FPATH)
NEW_DS_PATH = eval(CONFIG.data.files.NEW_DS_PATH)

#
# File macros
#

# define filepath for loss metrics history csv file
LOSS_METRICS_HISTORY_CSV = lambda trgt_src: os.path.join(RUN_DIR, _today+'_'+trgt_src+'_loss_metrics_history.csv')

# define filepath for the checkpoint file
CHECKPOINT_FNAME = lambda trgt_src: os.path.join(CHECKPOINTS_DIR, trgt_src+'_model_{epoch:02d}')

# define filepath for the benchmark csv file
BENCHMARK_HISTORY_CSV = os.path.join(RUN_DIR, 'benchmark_history.csv')

# define filepath for train and validation execution time csv file
TRAINVAL_TIME_CSV = os.path.join(RUN_DIR, 'trainval_time.csv')

# define filepath for the last model trained file
LAST_MODEL = os.path.join(RUN_DIR, 'last_model')

# define filepath for the log file
LOG_FILE = os.path.join(RUN_DIR, 'run.log')

# define lambda function to save scaler
SAVE_SCALER_PATH = lambda scaler_fname: os.path.join(SCALER_DIR, scaler_fname)
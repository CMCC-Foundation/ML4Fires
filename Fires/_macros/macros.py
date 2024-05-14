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
from os import path
from datetime import datetime as dt
from Fires._utilities.configuration import load_global_config

# get global configuration file
CONFIG = load_global_config()

# set pytorch configuration file
TORCH_CFG = load_global_config(config_fname=CONFIG.toml.torch_fname)

# set discord configuration file
DISCORD_CFG = load_global_config(config_fname=CONFIG.toml.logs.discord_fname)

_today = eval(CONFIG.utils.datetime.today)

# define current working directory
CURR_DIR = eval(CONFIG.dir.CURR_DIR)
print(CURR_DIR)

# define log dir
LOG_DIR = eval(CONFIG.dir.LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)

# define data dir
DATA_DIR = eval(CONFIG.dir.DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

# define scaler dir
SCALER_DIR = eval(CONFIG.dir.SCALER_DIR)
os.makedirs(SCALER_DIR, exist_ok=True)

# define experiments dir
EXPERIMENTS_DIR = eval(CONFIG.dir.EXPERIMENTS_DIR)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# FILEPATHS TO SEASFIRECUBE DATA
DATA_FPATH = eval(CONFIG.data.files.DATA_FPATH)
DATA_NEW_FPATH = eval(CONFIG.data.files.DATA_NEW_FPATH)
NEW_DS_PATH = eval(CONFIG.data.files.NEW_DS_PATH)

RUN_DIR = eval(TORCH_CFG.model.dir.RUN_DIR)
os.makedirs(RUN_DIR, exist_ok=True)

CHECKPOINTS_DIR = eval(TORCH_CFG.model.dir.CHECKPOINTS_DIR)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# FILE MACROS
LOSS_METRICS_HISTORY_CSV = lambda trgt_src: os.path.join(RUN_DIR, _today+'_'+trgt_src+'_loss_metrics_history.csv')
CHECKPOINT_FNAME = lambda trgt_src: os.path.join(CHECKPOINTS_DIR, trgt_src+'_model_{epoch:02d}')
BENCHMARK_HISTORY_CSV = os.path.join(RUN_DIR, 'benchmark_history.csv')
TRAINVAL_TIME_CSV = os.path.join(RUN_DIR, 'trainval_time.csv')
LAST_MODEL = os.path.join(RUN_DIR, 'last_model')
LOG_FILE = os.path.join(RUN_DIR, 'run.log')

# LAMBDA FUNCTIONS
SAVE_SCALER_PATH = lambda scaler_fname: os.path.join(SCALER_DIR, scaler_fname)
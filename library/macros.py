# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 					Copyright 2023 - CMCC Foundation						#
#																			#
# Site: 			https://www.cmcc.it										#
# CMCC Division:	ASC (Advanced Scientific Computing)						#
# Author:			Emanuele Donno											#
# Email:			emanuele.donno@cmcc.it									#
# 																			#
# Licensed under the Apache License, Version 2.0 (the "License");			#
# you may not use this file except in compliance with the License.			#
# You may obtain a copy of the License at									#
#																			#
#				https://www.apache.org/licenses/LICENSE-2.0					#
#																			#
# Unless required by applicable law or agreed to in writing, software		#
# distributed under the License is distributed on an "AS IS" BASIS,			#
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.	#
# See the License for the specific language governing permissions and		#
# limitations under the License.											#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
from datetime import datetime as dt

MODEL_NAME = "UNET++"

# DIRECTORY MACROS
CURR_DIR = os.getcwd() # path.dirname(path.abspath(os.getcwd()))
print(CURR_DIR)
LOG_DIR = os.path.join(CURR_DIR, 'logs')
DATA_DIR = os.path.join(CURR_DIR, "data")
SCALER_DIR = os.path.join(DATA_DIR, "scaler")		
EXPERIMENTS_DIR = os.path.join(CURR_DIR, "experiments")
RUN_DIR = os.path.join(EXPERIMENTS_DIR, MODEL_NAME)
CHECKPOINTS_DIR = os.path.join(RUN_DIR, 'checkpoints')
TENSORBOARD_DIR = os.path.join(RUN_DIR, 'tensorboard')

# CREATE DIRECTORIES
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# FILE MACROS
_today = str(dt.now()).split('.')[0].replace('-', '').replace(':', '').replace(' ', '_')
LOSS_METRICS_HISTORY_CSV = lambda trgt_src: os.path.join(RUN_DIR, _today+'_'+trgt_src+'_loss_metrics_history.csv')
CHECKPOINT_FNAME = lambda trgt_src: os.path.join(CHECKPOINTS_DIR, trgt_src+'_model_{epoch:02d}')
BENCHMARK_HISTORY_CSV = os.path.join(RUN_DIR, 'benchmark_history.csv')
TRAINVAL_TIME_CSV = os.path.join(RUN_DIR, 'trainval_time.csv')
LAST_MODEL = os.path.join(RUN_DIR, 'last_model')
LOG_FILE = os.path.join(RUN_DIR, 'run.log')

# LAMBDA FUNCTIONS
SAVE_SCALER_PATH = lambda scaler_fname: os.path.join(SCALER_DIR, scaler_fname)

# DATASET CREATION: MERGED DATA INFO
AGGREGATION = "GWIS: \t Spatio-Temporal | sum "
CREATOR_NOTES = ("Masked ocean with lsm variable. Missing years filled with Nan. \n "+
	"GWIS: Dataset created by using the ignition date of final burned areas \n "+
	"FCCI: Masked ocean with lsm variable. Missing years filled with Nan. "+
	"USE ONLY for monthly modeling! FireCCI is a monthly product that could not be fairly distributed to weekly time range. "+
	"Each week of the same month is filled with the same monthly value. To acquire the monthly value take the average of each month")
DESCRIPTION = ("This product was produced by combining ESA FCCI and GWIS data. \n \n "+
	"GWIS: \n Global dataset of individual fire perimeters for 2001-2020. "+
	"The dataset is in ESRI shapefile format and is derived from the MCD64A1 burned area product. "+
	"Each fire shapefile has a unique fire identification code, the initial date, the final date, "+
	"the geometry and a field specifying if it is a daily burned area or a final burned area.\n\n "+
	"FCCI: \n The ESA Fire Disturbance Climate Change Initiative (CCI) project has produced maps "+
	"of global burned area derived from satellite observations. The MODIS Fire_cci v5.1 grid product "+
	"described here contains gridded data on global burned area derived from the MODIS instrument onboard "+
	"the TERRA satellite at 250m resolution for the period 2001 to 2019. This product supercedes the previously "+
	"available MODIS v5.0 product. The v5.1 dataset was initially published for 2001-2017, and has been periodically "+
	"extended to include 2018 to 2020.")
DOWNLOADED_FROM = (" GWIS: \n https://gwis.jrc.ec.europa.eu/apps/country.profile/downloads \n "+
	"ESA FCCI: \n https://catalogue.ceda.ac.uk/uuid/3628cb2fdba443588155e15dee8e5352 ")
LONG_NAME = 'Burned Areas from GWIS and FCCI'
PROVIDER = 'GWIS | ESA CCI'


# import shutil
# shutil.copy(src='macros.py', dst='hyperparameters.txt')
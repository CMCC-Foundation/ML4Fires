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

from Fires._plots.plot_utils import (
	draw_features,
	highlight_ba,
	set_axis,
	draw_tropics_and_equator,
	plot_dataset_map,
)

from Fires._utilities.cli_args_checker import checker

from Fires._utilities.cli_args_parser import CLIParser

from Fires._utilities.logger import Logger as logger

from Fires._utilities.logger_itwinai import (SimpleItwinaiLogger, ItwinaiLightningLogger, ProvenanceLogger)

from Fires._utilities.metrics import (DiceLoss, FocalLoss, TverskyLoss)

from Fires._utilities.callbacks import (
	DiscordBenchmark,
	FabricBenchmark,
	FabricCheckpoint,
)

from Fires._utilities.configuration import (
	load_global_config,
	save_global_config,
	save_exp_config
)

from Fires._utilities.decorators import (
	debug,
	export,
)

from Fires._utilities.swin_model import seed_everything

from Fires._utilities.utils_general import check_backend
from Fires._utilities.utils_inference import load_input_data, create_data_loader, compute_aggregated_data, up_and_lower_bounds, process_and_plot_data
from Fires._utilities.utils_mlflow import setup_mlflow_experiment
from Fires._utilities.utils_trainer import get_trainer_loggers, get_itwinai_loggers, get_callbacks
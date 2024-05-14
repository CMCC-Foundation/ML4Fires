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

from Fires._utilities.cli_args_checker import checker

from Fires._utilities.cli_args_parser import CLIParser

from Fires._utilities.logger import Logger as logger

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
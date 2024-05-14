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

import toml, munch
from typing import Tuple, Union

from Fires._utilities.decorators import export
from Fires._utilities.cli_args_parser import CLIParser

PROGRAM_NAME = r'''
			███╗   ███╗██╗     ██╗  ██╗███████╗██╗██████╗ ███████╗███████╗
			████╗ ████║██║     ██║  ██║██╔════╝██║██╔══██╗██╔════╝██╔════╝
			██╔████╔██║██║     ███████║█████╗  ██║██████╔╝█████╗  ███████╗
			██║╚██╔╝██║██║     ╚════██║██╔══╝  ██║██╔══██╗██╔══╝  ╚════██║
			██║ ╚═╝ ██║███████╗     ██║██║     ██║██║  ██║███████╗███████║
			╚═╝     ╚═╝╚══════╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝

				    ██████████████████████████████████████╗
				    ██ Wildfires Burned Areas Prediction █║
				    ██████████████████████████████████████║
				    ╚═════════════════════════════════════╝
'''
PROGRAM_DESCRIPTION = "The following software is designed to train a ML model that must predict Wildfires Burned Areas on global scale."

# list of base commands
CMD_BASE = [
	[('-c', '--config'), dict(type=str, help='Configuration file for this program')], #, required=True
	[('-nexp', '--experiment_number'), dict(type=int, help='Select batch size (int)')]
]

@export
def checker() -> Tuple[str, dict]|None:
	"""
    Parses command-line arguments and checks for a valid experiment configuration.

    This function uses the `CLIParser` class to parse command-line arguments and validates
	the provided experiment number against the available configurations in a TOML file.

    Returns:
        Tuple[str, dict] | None:
            - If a valid experiment number is provided and found in the configuration file, returns a tuple containing:
                  - The name of the current experiment (e.g., "exp_1")
                  - A dictionary containing the configuration for the current experiment.
            - Otherwise, returns None.
    """
	# create CLI argument parser
	cli_parser = CLIParser(program_name=PROGRAM_NAME, description=PROGRAM_DESCRIPTION)
	# add base options
	cli_parser.add_arguments(parser=None, options=CMD_BASE)
	# parse arguments
	cli_args = cli_parser.parse_args()
	
	if cli_args.config and cli_args.experiment_number:

		# define path to TOML configuration file with all the experiments
		config_path = cli_args.config
		# load from the TOML configuration file the dictionary with all the experiments
		experiments = munch.munchify(toml.load(config_path))
		# list with experiments names from dictionary keys
		exp_keys = experiments.keys()

		# define the number of the current experiment
		n_exp = cli_args.experiment_number
		curr_exp = f'exp_{n_exp}'

		# check the number of experiments in the configuration
		if curr_exp not in exp_keys: 
			print(f" \n No experiment #{n_exp} in the {config_path.split('/')[-1]} file \n ".upper())
			exit(1)
		
		# define experiment dictionary
		exp_config = experiments[curr_exp]

		# print(f" Experiment #{n_exp} \n Path to configuration file: {config_path} \n")
		# for key in exp_config.keys():
		# 		print(f"  - {key}: {exp_config[key]}")
		
		return (curr_exp, exp_config)
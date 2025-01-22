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

import torch
import itertools
from typing import List, Tuple

from Fires._macros.macros import CONFIG_DIR
from Fires._scalers.minmax import MinMaxScaler
from Fires._scalers.standard import StandardScaler
from Fires._utilities.configuration import save_exp_config

BATCH_SIZES = [2, 4, 8]
BASE_FILTER_DIMS = [16, 32, 64]
ACTIVATIONS = [torch.nn.Sigmoid, torch.nn.ReLU]
LOSSES = [torch.nn.L1Loss, torch.nn.BCELoss]
SCALERS = [MinMaxScaler, StandardScaler]


def create_exp_combinations() -> List[dict]:
	"""
	Generates all possible experiment combinations with filtering
	"""
	combinations = list(itertools.product(BASE_FILTER_DIMS, ACTIVATIONS, LOSSES))
	filtered_combinations = []
	for base_filter_dim, activation, loss in combinations:

		if activation != torch.nn.Sigmoid and loss == torch.nn.BCELoss:	continue

		filtered_combinations.append(dict(
			base_filter_dim=base_filter_dim,
			activation_cls=activation,
			loss_cls=loss,
		))
	
	return filtered_combinations

def categorize_exp(experiments:List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
	"""
	Categorizes experiments by base filter dimension.
	"""
	bdim16 = [exp for exp in experiments if exp["base_filter_dim"] == 16]
	bdim32 = [exp for exp in experiments if exp["base_filter_dim"] == 32]
	bdim64 = [exp for exp in experiments if exp["base_filter_dim"] == 64]
	return bdim16, bdim32, bdim64


def upp_experiments() -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
	"""
	Generates and categorizes UPP experiments.
	"""
	all_experiments = create_exp_combinations()
	bdim16, bdim32, bdim64 = categorize_exp(all_experiments)
	experiments = [all_experiments, bdim16, bdim32, bdim64]
	return experiments


if __name__ == '__main__':
	print("Creating experiments for Unet++")
	
	filenames = ['UPP_all.toml', 'UPP_16.toml', 'UPP_32.toml', 'UPP_64.toml']
	experiments = upp_experiments()

	print("Saving experiments oconfigurations...")

	# save experiments in configuration files
	for exp, fname in zip(experiments, filenames):
		try:
			print(f'  - Saving experiment in {fname} configuration file')
			save_exp_config(exp_configuration=exp, config_dir=CONFIG_DIR, filepath=fname)
		except Exception as e:
			print(f"Error saving: {e}")
		
		print("Experiments saved")

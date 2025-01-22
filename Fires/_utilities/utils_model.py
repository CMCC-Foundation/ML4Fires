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

from typing import Any, Iterable
from tqdm import tqdm
import numpy as np
import itertools
import warnings
import math
import time
import os

import torch
from lightning import Fabric
from torch.nn import functional as F

from Fires._utilities.decorators import export


@export
def seed_everything(seed):
	"""
	Seed RNG states of the execution environment.
	
	Args:
		seed (int): The seed value to be used for reproducibility.
	"""
	
	# python seed
	os.environ["PYTHONHASHSEED"] = str(seed)
	# numpy seed
	np.random.seed(seed)
	# torch seed
	torch.manual_seed(seed)
	# pytorch seed
	print(f'Execution seeded successfully with seed {seed}')

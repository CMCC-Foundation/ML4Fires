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

import numpy as np
import torch.nn as nn
import lightning.pytorch as pl

from turtle import forward
from typing import Any, List
from timm.layers import to_2tuple

from Fires._utilities.decorators import export


@export
class BaseLightningModule(pl.LightningModule):
	def __init__(self, *args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)
		self.callback_metrics = {}
	
	def training_step(self, batch, batch_idx):
		# get data from the batch
		x, y = batch
		# forward pass
		y_pred = self(x)
		# compute loss
		loss = self.loss(y_pred, y)
		# define log dictionary
		log_dict = {'train_loss': loss}
		# compute metrics
		for metric in self.metrics:
			log_dict.update({f'train_{metric.name}' : metric(y_pred, y)})
		# log the outputs
		self.callback_metrics = {**self.callback_metrics, **log_dict}
		# return the loss
		return {'loss':loss}

	def validation_step(self, batch, batch_idx):
		# get data from the batch
		x, y = batch
		# forward pass
		y_pred = self(x)
		# compute loss
		loss = self.loss(y_pred, y)
		# define log dictionary
		log_dict = {'val_loss': loss}
		# compute metrics
		for metric in self.metrics:
			log_dict.update({f'val_{metric.name}' : metric(y_pred, y)})
		# log the outputs
		self.callback_metrics = {**self.callback_metrics, **log_dict}
		# return the loss
		return {'loss':loss}

	def on_validation_model_eval(self) -> None:
		self.eval()
	def on_validation_model_train(self) -> None:
		self.train()
	def on_test_model_train(self) -> None:
		self.train()
	def on_test_model_eval(self) -> None:
		self.eval()
	def on_predict_model_eval(self) -> None:
		self.eval()


@export
class BaseUnetPlusPlus(BaseLightningModule):

	def __init__(self,
			input_shape:tuple=(720, 1440, 8),
			num_classes:int=1,
			depth:int=4,
			base_filter_dim:int=32,
			deep_supervision:bool=False,
			*args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)
		
		self.input_shape = input_shape
		self.num_classes = num_classes
		self.depth = depth
		self.base_filter_dim = base_filter_dim
		self.deep_supervision = deep_supervision
		self.model = nn.Identity()
	
	def forward(self, inputs) -> Any:
		return self.model(inputs)


@export
class BaseVGG(BaseLightningModule):
	def __init__(self, 
			channels: List[int], 
			activation: nn.Module = nn.Identity, 
			kernel_size: int = 3, 
			*args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)
		self.channels = channels
		self.activation = activation
		self.kernel_size = kernel_size
		self.model = nn.Identity()

	def forward(self, inputs) -> Any:
		return self.model(inputs)

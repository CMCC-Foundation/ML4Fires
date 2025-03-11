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
# import lightning.pytorch as pl
import lightning as pl

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import F1Score, FBetaScore, MatthewsCorrCoef, Precision, Recall, Accuracy, MeanSquaredError, ConcordanceCorrCoef

from turtle import forward
from typing import Any, Dict, List, Optional
from timm.layers import to_2tuple

from Fires._utilities.decorators import export
from Fires._utilities.general import general_utils

from itwinai.loggers import ConsoleLogger, Prov4MLLogger, MLFlowLogger as IMLFlowLogger, Logger as BaseItwinaiLogger


@export
class BaseLightningModule(pl.LightningModule):
	"""
	A base class for PyTorch Lightning modules, providing essential training, validation, and testing steps.

	Attributes:
		callback_metrics (dict):
			A dictionary to store metrics during training and validation.

	Methods:
		__init__(*args, **kwargs):
			Initializes the base module, setting up callback metrics.
		training_step(batch, batch_idx):
			Performs a training step, calculating loss and logging metrics.
		validation_step(batch, batch_idx):
			Performs a validation step, calculating loss and logging metrics.
		on_validation_model_eval():
			Sets the model to evaluation mode before validation epoch.
		on_validation_model_train():
			Sets the model to training mode after validation epoch.
		on_test_model_train():
			Sets the model to training mode before test epoch.
		on_test_model_eval():
			Sets the model to evaluation mode before test epoch.
		on_predict_model_eval():
			Sets the model to evaluation mode before predict step.
	"""
	def __init__(self, *args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)
		# self.callback_metrics:Dict[str | Any] = {}
		self._trn_loss = {'sum': 0, 'steps': 0}
		self._vld_loss = {'sum': 0, 'steps': 0}
		self._training_metrics = {'steps' : 0, 'metrics' : {}}
		self._validation_metrics = {'steps' : 0, 'metrics' : {}}
	
		self.setup_metrics(kwargs["TORCH_CFG"])
  
	def setup_metrics(self,torch_cfg):
		_metrics = []
        
		for key, value in torch_cfg["metrics"].items():
			matric_func, matric_func_kwargs = general_utils.separate_kwargs(value)
			metric_instance = general_utils.call_instance_of_function(
				**general_utils.process_call_string(input_string=matric_func),
            	**matric_func_kwargs,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
			metric_instance.name = key
			_metrics.append(metric_instance)
            
            
		# # accuracy
		# accuracy = Accuracy(task='binary')
		# accuracy.name = "accuracy"
		# accuracy.to("cuda" if torch.cuda.is_available() else "cpu")
		# _metrics.append(accuracy)

		# # precision
		# precision = Precision(task='binary')
		# precision.name = "precision"
		# precision.to("cuda" if torch.cuda.is_available() else "cpu")
		# _metrics.append(precision)

		# # recall
		# recall = Recall(task='binary')
		# recall.name = "recall"
		# recall.to("cuda" if torch.cuda.is_available() else "cpu")
		# _metrics.append(recall)

		# # f1 score
		# f1_score = F1Score(task='binary')
		# f1_score.name = "f1_score"
		# f1_score.to("cuda" if torch.cuda.is_available() else "cpu")
		# _metrics.append(f1_score)

		# # f2 score
		# f2_score = FBetaScore(task='binary', beta=float(2))
		# f2_score.name = "f2_score"
		# f2_score.to("cuda" if torch.cuda.is_available() else "cpu")
		# _metrics.append(f2_score)

		# # mcc
		# mcc = MatthewsCorrCoef(task='binary')
		# mcc.name = "mcc"
		# _metrics.append(mcc)

		# define model metrics
  
		self.metrics = _metrics
 
	def training_step(self, batch, batch_idx):
		# get data from the batch
		x, y = batch
		# forward pass
		y_pred = self(x)
		# compute loss
		loss = self.loss(y_pred, y)
		# define log dictionary
		# log_dict = {'train_loss': loss}
		self.log("trn_loss", loss, prog_bar=True, on_epoch=True, logger=True)
		
		# binarize real and predicted data
		y_true_bin = (y > 0).int()
		y_pred_bin = (y_pred > 0).int()

		# flatten tensors
		y_true_flat = y_true_bin.view(-1)
		y_pred_flat = y_pred_bin.view(-1)

		self._training_metrics['steps'] += 1

		# compute metrics		
		
		# log the outputs
		# self.callback_metrics = {**self.callback_metrics, **log_dict}
		self.compute_metrics(y_true_flat,y_pred_flat)
        
		# return the loss
		self._training_loss = loss
		self._trn_loss['sum'] += loss
		self._trn_loss['steps'] += 1
		return {'loss':loss}


	def compute_metrics(self,truth,pred):
		for metric in self.metrics:
			metric_name = f'train_{metric.name.lower()}'
			computed_metric = metric(pred, truth)
			self._training_metrics['metrics'].setdefault(metric_name, 0.0)
			self._training_metrics['metrics'][metric_name] += computed_metric
    
    
	def on_train_epoch_end(self):
		if self.loggers[0].experiment is not None:
			self.loggers[0].experiment.log_metrics({"trn_loss": self._trn_loss['sum']/self._trn_loss['steps']}, step=self.current_epoch)
		if self.loggers[-1].experiment is not None:

			context='training'
			self.loggers[-1].experiment.log(item=self.current_epoch, identifier="epoch", kind='metric', step=self.current_epoch, context=context)
			self.loggers[-1].experiment.log(item=self, identifier=f"model_version_{self.current_epoch}", kind='model_version', step=self.current_epoch, context=context)
			self.loggers[-1].experiment.log(item=None, identifier=None, kind='system', step=self.current_epoch, context=context)
			self.loggers[-1].experiment.log(item=None, identifier=None, kind='carbon', step=self.current_epoch, context=context)
			self.loggers[-1].experiment.log(item=None, identifier="train_epoch_time", kind='execution_time', step=self.current_epoch, context=context)
			self.loggers[-1].experiment.log(item=self._trn_loss['sum']/self._trn_loss['steps'], identifier="training_loss", kind='metric', step=self.current_epoch, context=context)

		for metric_name, metric_value in self._training_metrics['metrics'].items():
				self.loggers[-1].experiment.log(item=metric_value/self._training_metrics['steps'], identifier=f"{metric_name}_epoch", kind='metric', step=self.current_epoch, context=context)
			
		self._training_metrics = {'steps' : 0, 'metrics' : {}}
		self._trn_loss = {'sum': 0, 'steps': 0}

		return super().on_train_epoch_end()
	
	def validation_step(self, batch, batch_idx):
		# get data from the batch
		x, y = batch
		# forward pass
		y_pred = self(x)
		# compute loss
		loss = self.loss(y_pred, y)
		# define log dictionary
		# log_dict = {'val_loss': loss}

		self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)

		# binarize real and predicted data
		y_true_bin = (y > 0).int()
		y_pred_bin = (y_pred > 0).int()

		# flatten tensors
		y_true_flat = y_true_bin.view(-1)
		y_pred_flat = y_pred_bin.view(-1)
		self._validation_metrics['steps'] += 1

		# compute metrics
		self.compute_metrics(y_true_flat,y_pred_flat)
		
		# log the outputs
		# self.callback_metrics = {**self.callback_metrics, **log_dict}

		# return the loss
		self._validation_loss = loss
		self._vld_loss['sum'] += loss
		self._vld_loss['steps'] += 1
		return {'loss':loss}

	def configure_optimizers(self):
		optimizer = Adam(self.parameters(), lr=1e-3)
		scheduler = StepLR(optimizer, step_size=1, gamma=0.96)
		return [optimizer], [scheduler]
	
	def on_validation_epoch_end(self):

		if self.loggers[0].experiment is not None:
			self.loggers[0].experiment.log_metrics({"val_loss": self._vld_loss['sum']/self._vld_loss['steps']}, step=self.current_epoch)
		if self.loggers[-1].experiment is not None:
			
			context='validation'
			self.loggers[-1].experiment.log(item=self.current_epoch, identifier="epoch", kind='metric', step=self.current_epoch, context=context)
			self.loggers[-1].experiment.log(item=self, identifier=f"model_version_{self.current_epoch}", kind='model_version', step=self.current_epoch, context=context)
			self.loggers[-1].experiment.log(item=None, identifier=None, kind='system', step=self.current_epoch, context=context)
			self.loggers[-1].experiment.log(item=None, identifier=None, kind='system', step=self.current_epoch, context=context)
			self.loggers[-1].experiment.log(item=None, identifier=None, kind='carbon', step=self.current_epoch, context=context)
			self.loggers[-1].experiment.log(item=None, identifier="validation_epoch_time", kind='execution_time', step=self.current_epoch, context=context)
			self.loggers[-1].experiment.log(item=self._vld_loss['sum']/self._vld_loss['steps'], identifier="validation_loss", kind='metric', step=self.current_epoch, context=context)

		for metric_name, metric_value in self._validation_metrics['metrics'].items():
				self.loggers[-1].experiment.log(item=metric_value/self._validation_metrics['steps'], identifier=f"{metric_name}_epoch", kind='metric', step=self.current_epoch, context=context)

		self._validation_metrics = {'steps' : 0, 'metrics' : {}}
		self._vld_loss = {'sum': 0, 'steps': 0}

		return super().on_validation_epoch_end()
	
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
	"""
	A base class for U-Net++ models, inheriting from BaseLightningModule.

	Attributes:
		input_shape (tuple):
			The shape of input images (height, width, channels).
		num_classes (int):
			The number of output classes for segmentation.
		depth (int):
			The depth of the U-Net++ architecture.
		base_filter_dim (int):
			The number of filters in the first layer.
		deep_supervision (bool):
			Whether to use deep supervision.
		model (nn.Module):
			The underlying U-Net++ model (initialized as nn.Identity).

	Methods:
		__init__(input_shape, num_classes, depth, base_filter_dim, deep_supervision, *args, **kwargs):
			Initializes the base U-Net++ model with specified parameters.
		forward(inputs):
			Performs a forward pass through the U-Net++ model.
	"""
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
	"""
	A base class for VGG-like models, inheriting from BaseLightningModule.

	Attributes:
		channels (List[int]):
			A list of channel numbers for each convolutional block.
		activation (nn.Module):
			The activation function to use after each convolutional block.
		kernel_size (int):
			The size of the convolutional kernels.
		model (nn.Module):
			The underlying VGG-like model (initialized as nn.Identity).

	Methods:
		__init__(channels, activation, kernel_size, *args, **kwargs):
			Initializes the base VGG model with specified parameters.
		forward(inputs):
			Performs a forward pass through the VGG model.
	"""

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

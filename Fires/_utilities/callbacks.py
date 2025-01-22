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

from lightning import Trainer, LightningModule
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import json
import os



class FabricBenchmark:
	"""
	Tracks training and validation metrics during training and saves them
	to a CSV file.

	Args:
		filename (str): The filename to store the CSV data.
	"""
	def __init__(self, filename) -> None:
		self.filename = filename

	def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
		"""
		Called at the end of each validation epoch. This method retrieves
		metrics from the Trainer, updates the internal DataFrame, and saves 
		the DataFrame to the specified CSV file on rank 0.

		Args:
			trainer (Trainer): The PyTorch Lightning Trainer object.
			pl_module (LightningModule): The Lightning Module instance.
		"""

		if trainer.global_rank == 0:
			# get the metrics from the trainer
			metrics = trainer.callback_metrics
			# create csv file if not exists
			if not os.path.exists(self.filename):
				# define csv columns
				columns = [key for key in metrics.keys()]
				# create empty DataFrame
				self.df = pd.DataFrame(columns=columns)
				# save the DataFrame to disk
				self.df.to_csv(self.filename)
			else:
				# get the DataFrame from disk
				self.df = pd.read_csv(self.filename, index_col=0)
				# get the columns
				columns = self.df.columns
			# create the row to be added to DataFrame
			row = [metrics[col].item() for col in columns]
			# add to the DataFrame the row
			self.df.loc[len(self.df.index)] = row
			# store the data to the csv file
			self.df.to_csv(self.filename)
		return


class FabricCheckpoint:
	"""
	Tracks the monitored metric (e.g., validation loss) during training and saves 
	checkpoints to disk when a new minimum is reached.

	Args:
		dst (str): The destination directory to save checkpoints.
		monitor (str, optional): The metric to monitor for improvement. Defaults to 'val_loss'.
		verbose (bool, optional): Whether to print information about saved checkpoints. Defaults to False.
	"""
	
	def __init__(self, dst, monitor: str = 'val_loss', verbose: bool = False) -> None:
		self.dst = dst
		self.monitor = monitor
		self.verbose = verbose
		self.global_min_loss = 9999.9
		self.df = None

	def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
		"""
        Called at the end of each validation epoch. This method collects training and
        validation metrics, checks if a new minimum for the monitored metric is reached,
        and saves a checkpoint if necessary (on rank 0 only).

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer object.
            pl_module (LightningModule): The Lightning Module instance.
        """
		# only process 0
		if trainer.global_rank == 0:
			# collect training and validation metrics
			self.__collect_metrics(trainer.callback_metrics)
			# save checkpoint if necessary
			self.__checkpoint(trainer)

	def __collect_metrics(self, metrics):
		"""
        Creates a DataFrame from the provided metrics dictionary and concatenates it
        with the existing DataFrame (if any).

        Args:
            metrics (dict): A dictionary containing training and validation metrics.
        """
		# define csv columns
		columns = [key for key in metrics.keys()]
		data = {}
		for col in columns: data.update({col: [metrics[col].item()]})
		df = pd.DataFrame(data=data)
		if self.df is None:
			self.df = df
		else:
			self.df = pd.concat([self.df, df]).reset_index(drop=True)

	def __checkpoint(self, trainer):
		"""
        Checks if the monitored metric has improved (reached a new minimum) and 
        saves a checkpoint if it has.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer object.
        """
		# get the loss list that we are monitoring
		losses = self.df[self.monitor].to_numpy()
		# get the current loss
		cur_loss = losses[-1]
		# if we reached a new minimum
		if cur_loss < self.global_min_loss:
			# get the checkpoint output filename
			path = os.path.join(self.dst, f"epoch-{trainer.current_epoch+1:04d}-val_loss-{cur_loss:.2f}.ckpt")
			# update the global minimum with the new one
			self.global_min_loss = cur_loss
			# eventually print the update
			if self.verbose: print(f'Epoch [{trainer.current_epoch+1}/{trainer.max_epochs}]: {self.monitor} improved from {self.global_min_loss} to {cur_loss}, saving checkpoint to {path}')
			# save the model to disk
			trainer.fabric.save(path, {'model':trainer.model, 'optimizer':trainer.optimizer})

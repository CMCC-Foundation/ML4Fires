# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 					Copyright 2023 - CMCC Foundation						
#																			
# Site: 			https://www.cmcc.it										
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
import inspect
import numpy as np
import tensorflow as tf
from .dataset_builder import DatasetBuilder
from .tfr_io import DatasetWriter, TensorCoder
from .decorators import debug, export
from .macros import LOG_DIR
from .logger import Logger as logger
_wfds_builder = logger(log_dir=LOG_DIR).get_logger("WildFiresDatasetBuilder")
_wfds_writer = logger(log_dir=LOG_DIR).get_logger("WildFiresDatasetWriter")

@export
@debug(log=_wfds_builder)
class WildFiresDatasetBuilder(DatasetBuilder):

	def __init__(self, epochs:int, tensor_coder:TensorCoder, filenames:list):
		super().__init__()

		# define logger
		self.logger = _wfds_builder

		# set the number of epochs
		self.epochs = epochs

		# coder for tfrecord decoding
		self.tensor_coder = tensor_coder

		# filenames array
		if not filenames:
			raise ValueError("Filenames list must not be empty")
		self.filenames = filenames # []

		# counter for the number of elements of the dataset
		self.count = self._count_filenames_elems(filenames=filenames)
		
		# datasets array
		self.main_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=self.AUTOTUNE).map(tensor_coder.decoding_fn, num_parallel_calls=self.AUTOTUNE)
		self.datasets_list = [self.main_dataset]


	def source(self, filenames):
		"""
		Adds a new source of filenames for the dataset. The source must contain 
		elements of identical type.

		Parameters
		----------
		filenames : list(str)
		    List of filenames used as source for the dataset

		"""
		# get function name
		fn_name = inspect.currentframe().f_code.co_name
		
		# count the number of elements of the dataset and increment count variable
		self.count += self._count_filenames_elems(filenames=filenames)
		self.logger.info(f"{fn_name} | Number of elements of the dataset: {self.count}")
		
		# save the filenames
		self.filenames += filenames
		self.logger.info(f"{fn_name} | Save filenames: \n {self.filenames}")
		
		# create the dataset
		self.datasets_list += [tf.data.TFRecordDataset(filenames, num_parallel_reads=self.AUTOTUNE).map(self.tensor_coder.decoding_fn, num_parallel_calls=self.AUTOTUNE)]
		
		return self


	def augment(self, aug_fns):
		"""
		Create and adds to the dataset the augmented versions of the data
		
		"""

		# add augmented datasets to the main
		self.datasets_list += [(self.main_dataset.map(lambda x,y: (aug_fn((x,y))), num_parallel_calls=self.AUTOTUNE)) for aug_fn in aug_fns.values()]
		
		return self


	def assemble_dataset(self):
		"""
		Assemble a tf.data.Dataset that has been built from sources and augmentations

		"""

		# get function name
		fn_name = inspect.currentframe().f_code.co_name

		# get the number of repeats of choice dataset
		datasets_len = len(self.datasets_list)
		self.logger.info(f"{fn_name} | Dataset length: {datasets_len}")
		
		# select the order in which the augmented samples must be interleaved
		choice_dataset = tf.data.Dataset.range(datasets_len).repeat(count=self.count)

		# statically interleave elements from all the datasets
		self.dataset = tf.data.Dataset.choose_from_datasets(datasets=self.datasets_list, choice_dataset=choice_dataset)
		
		return self


@export
@debug(log=_wfds_writer)
class WildFiresDatasetWriter(DatasetWriter):

	def __init__(self, drivers_info: list, shard_size: int, dtype=np.float32) -> None:
		super().__init__(drivers_info, shard_size, dtype)

		# define logger
		self.logger = _wfds_writer


	def source(self, **kwargs):
		"""
		Add years and data as sources

		"""

		if 'years' in kwargs.keys():
			self.years = kwargs['years']
		if 'data' in kwargs.keys():
			self.data = kwargs['data']
		return self

	
	def process(self, dst):
		"""
		Process data 
		

		Parameters
		----------
		dst : str
			Path to directory with `.tfrecord` files

		Raises
		------
		Exception
			When years are not added as source
		Exception
			When data is not added as source
		"""

		if not hasattr(self, 'years'):
			raise Exception(f'No data years have been found. Try adding them with source(years=[list, of, years])')
		if not hasattr(self, 'data'):
			raise Exception(f'No dataset have been provided. Try adding it with source(data=xarray.Dataset())')
		loaded_ds = self.data

		# for each year
		for year in self.years:
			existing_files = sorted([os.path.join(dst,f) for f in os.listdir(dst) if f.endswith('.tfrecord') and int(f.split('_')[0]) == year])
			if len(existing_files)!=0:
				self.logger.info(f'Files already existing for year {year}. Skipping')
				continue
			self.logger.info(f'Processing data of year {year}')

			# select data for current year
			ds = loaded_ds.sel(time=str(year)) 

			# convert dataset data to numpy array
			data = self._data_to_numpy(ds)
			self.logger.info(f"Shape: {data[0].shape}")

			# save data to disk
			self.write(dst=dst, data=data, year=year)


	def _data_to_numpy(self, ds):
		"""
		Builds a numpy array from the provided patch dataset and ids to be got. 
		Domain patches are stacked together and then stacked along time.

		"""
		data = []
		for info in self.drivers_info:
			data.append(np.stack([ds[var].load().data for var in info.vars], axis=-1).astype(self.dtype))
		return tuple(data)
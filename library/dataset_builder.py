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

import tensorflow as tf
from .scaling import StandardScaler
from .decorators import export

@export
class DatasetBuilder():
	"""
	Dataset Builder default class. It builds a dataset with the preferred characteristics. It is general purpose, so 
	that it can be used with every desired ML workflow.

	"""
	def __init__(self):
		self.AUTOTUNE = tf.data.AUTOTUNE


	def batch(self, batch_size=None, drop_remainder=False):
		# check if dataset is defined
		self._check_dataset()
		# separate in batches
		bsize = batch_size if batch_size else self.count
		self.dataset = self.dataset.batch(bsize, drop_remainder=drop_remainder, num_parallel_calls=self.AUTOTUNE)
		return self


	def shuffle(self, shuffle_buffer=None):
		# check if dataset is defined
		self._check_dataset()
		# shuffle if necessary
		if shuffle_buffer:
			self.dataset = self.dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
		return self


	def scale(self, scaler:StandardScaler=None):
		# check if dataset is defined
		self._check_dataset()
		def apply_scaling(data):
			X, y = data
			if scaler:
				X = scaler.transform(X, features_type='input')
				y = scaler.transform(y, features_type='output')
			return (X, y)
		# scale the data
		self.dataset = self.dataset.map(lambda X,y: (apply_scaling((X,y))), num_parallel_calls=self.AUTOTUNE)
		return self


	def resize(self, shape:tuple=None):
		# check if dataset is defined
		self._check_dataset()
		def apply_resize(data):
			resized_data = []
			for x in data:
				resized_data.append(tf.image.resize(x, shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR))
			return tuple(resized_data)
		if shape:
			self.dataset = self.dataset.map(lambda X,y: (apply_resize((X,y))), num_parallel_calls=self.AUTOTUNE)
		return self


	def mask(self, mask):
		# check if dataset is defined
		self._check_dataset()
		# apply mask function
		def apply_mask(data):
			X,y = data
			y_masked = tf.where(tf.math.is_nan(y), tf.ones_like(y) * mask, y)
			return (X, y_masked)
		# apply mask on target if label_no_cyclone is provided
		if mask:
			self.dataset = self.dataset.map(lambda X,y: (apply_mask((X,y))), num_parallel_calls=self.AUTOTUNE)
		return self


	def repeat(self):
		# check if dataset is defined
		self._check_dataset()
		# set number of epochs that can be repeated on this dataset
		self.dataset = self.dataset.repeat(count=self.epochs)
		return self


	def optimize(self):
		# check if dataset is defined
		self._check_dataset()
		# add parallelism option
		options = tf.data.Options()
		options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
		options.experimental_threading.max_intra_op_parallelism = 1
		self.dataset = self.dataset.with_options(options)
		# prefetch
		self.dataset = self.dataset.prefetch(buffer_size=self.AUTOTUNE)
		return self


	def _check_dataset(self):
		if not hasattr(self, 'dataset'):
			raise Exception('The dataset variable is not defined. Try calling assemble_dataset() first.')


	def _count_filenames_elems(self, filenames):
		"""
		Counts the number of elements present in the passed dataset files.
		"""
		return sum([int(fname.split('/')[-1].split('.tfrecord')[0].split('_')[-1]) for fname in filenames])
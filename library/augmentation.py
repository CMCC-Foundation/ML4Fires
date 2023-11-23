# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 					Copyright 2023 - CMCC Foundation						#
#																			#
# Site: 			https://www.cmcc.it										#
# CMCC Division:	ASC (Advanced Scientific Computing)						#
# Author:			Emanuele Donno											#
# Email:			emanuele.donno@cmcc.it									#
# 																			#
# Licensed under the Apache License, Version 2.0 (the "License");			#
# you may not use this file except in compliance with the License.			#
# You may obtain a copy of the License at									#
#																			#
#				https://www.apache.org/licenses/LICENSE-2.0					#
#																			#
# Unless required by applicable law or agreed to in writing, software		#
# distributed under the License is distributed on an "AS IS" BASIS,			#
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.	#
# See the License for the specific language governing permissions and		#
# limitations under the License.											#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import tensorflow as tf
from .decorators import debug, export
from .macros import LOG_DIR
from .logger import Logger as logger
_logger = logger(log_dir=LOG_DIR).get_logger("Augmentation")

@export
@debug(log=_logger)
def rot180(data):
	"""
	Performs a rotation of 180 degrees

	Parameters
	----------
	data : 
		Image data that must be rotated

	Returns
	-------
	tuple
		Rotated images for drivers and targets
	"""
	X,Y = data
	X = tf.image.rot90(X, k=2)
	Y = tf.image.rot90(Y, k=2)
	return (X,Y)

@export
@debug(log=_logger)
def left_right(data):
	"""
	Performs a left-right flip

	Parameters
	----------
	data :
		Image data that must be flipped

	Returns
	-------
	tuple
		Flipped images for drivers and targets
	"""
	X,Y = data
	X = tf.image.flip_left_right(X)
	Y = tf.image.flip_left_right(Y)
	return (X,Y)

@export
@debug(log=_logger)
def up_down(data):
	"""
	Performs a up-down flip

	Parameters
	----------
	data :
		Image data that must be flipped

	Returns
	-------
	tuple
		Flipped images for drivers and targets
	"""
	X,Y = data
	X = tf.image.flip_up_down(X)
	Y = tf.image.flip_up_down(Y)
	return (X,Y)
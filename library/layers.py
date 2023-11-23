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
from .decorators import export
from .hyperparams import conv_params
from .configuration import load_global_config
toml_general = load_global_config()
toml_model = eval(toml_general['toml_configuration_files']['toml_model'])

@export
class ConvUnit(tf.keras.layers.Layer):
	"""
	Convolutional Unit used to build UNET++ models.
	It extends Tensorflow Keras Layer class.
	
	"""
	
	def __init__(self, stage, filter, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
		super(ConvUnit, self).__init__(trainable, name, dtype, dynamic, **kwargs)

		self.stage = stage
		self.filter = filter
		self.conv_1 = tf.keras.layers.Conv2D(**conv_params(filter=filter, name=f"conv_{stage}_1"))
		self.conv_2 = tf.keras.layers.Conv2D(**conv_params(filter=filter, name=f"conv_{stage}_2"))
		self.dropout = tf.keras.layers.Dropout(rate=toml_model['layers']['drop_rate'])
	
	def call(self, inputs, training=False):
		x = self.conv_1(inputs)
		x = self.dropout(x, training)
		x = self.conv_2(x)
		x = self.dropout(x, training)
		return x
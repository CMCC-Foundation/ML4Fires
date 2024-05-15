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

from torchvision.transforms.functional import rotate, vflip, hflip

from Fires._macros.macros import LOG_DIR
from Fires._utilities.decorators import debug, export
from Fires._utilities.logger import Logger as logger
_logger = logger(log_dir=LOG_DIR).get_logger("Augmentation")

@export
@debug(log=_logger)
def rot180(data):
	"""Rotates an image (assumed to be the first element in the data tuple) by 180 degrees.

	Args:
		data: A tuple containing the image and potentially other data. The first
			 element is assumed to be the image to be rotated.

	Returns:
		A tuple containing the rotated image (at index 0) and the other elements
		from the input data.
	"""
	X, Y = data
	X = rotate(img=X, angle=90.0)
	Y = rotate(img=Y, angle=90.0)
	return (X, Y)

@export
@debug(log=_logger)
def left_right(data):
	"""Flips an image (assumed to be the first element in the data tuple) horizontally.

	Args:
		data: A tuple containing the image and potentially other data. The first
			 element is assumed to be the image to be flipped.

	Returns:
		A tuple containing the horizontally flipped image (at index 0) and the
		other elements from the input data.
	"""
	X, Y = data
	X = hflip(img=X)
	Y = hflip(img=Y)
	return (X, Y)

@export
@debug(log=_logger)
def up_down(data):
	"""Flips an image (assumed to be the first element in the data tuple) vertically.

	Args:
		data: A tuple containing the image and potentially other data. The first
			 element is assumed to be the image to be flipped.

	Returns:
		A tuple containing the vertically flipped image (at index 0) and the
		other elements from the input data.
	"""
	X, Y = data
	X = vflip(img=X)
	Y = vflip(img=Y)
	return (X, Y)
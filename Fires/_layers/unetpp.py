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

import torch.nn as nn

class VGGBlock(nn.Module):
	"""
	Implements a fundamental building block of the VGG network architecture.

	This block consists of:

		1. Two convolutional layers with 3x3 kernels and padding=1 to preserve spatial dimensions.
		2. Batch Normalization after each convolutional layer to improve training stability and speed.
		3. ReLU activation after each Batch Normalization to introduce non-linearity.
		4. (Optional) Dropout layers after ReLU (currently commented out) to prevent overfitting during training.

	Attributes:
		conv1 (nn.Conv2d): The first convolutional layer.
		bn1 (nn.BatchNorm2d): Batch Normalization for the first convolutional layer.
		conv2 (nn.Conv2d): The second convolutional layer.
		bn2 (nn.BatchNorm2d): Batch Normalization for the second convolutional layer.
		relu (nn.ReLU): ReLU activation function.
		drop (nn.Dropout2d): Dropout layer (commented out).

	Args:
		in_channels (int): Number of input channels.
		middle_channels (int): Number of channels in the intermediate feature maps.
		out_channels (int): Number of output channels.
	"""
	def __init__(self, in_channels, middle_channels, out_channels):
		super().__init__()
		# self.drop = nn.Dropout2d(p=0.5)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(middle_channels)
		self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		"""
		Defines the forward pass of the VGGBlock.

		Args:
			x (torch.Tensor): Input tensor.

		Returns:
			torch.Tensor: Output tensor after passing through the block's layers.
		"""
		out = self.conv1(x)
		out = self.bn1(out)
		# out = self.drop(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		# out = self.drop(out)
		out = self.relu(out)

		return out

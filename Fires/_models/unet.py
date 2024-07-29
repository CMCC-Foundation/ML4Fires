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

from turtle import forward
import lightning.pytorch as pl
from typing import Any, List
import torch.nn as nn
import torch

from Fires._models.base import BaseLightningModule
from Fires._layers.unetpp import VGGBlock
from Fires._utilities.decorators import export

@export
class Unet(BaseLightningModule):
	"""
	U-Net model for image segmentation, based on the PyTorch Lightning framework.

	Attributes:
		input_shape (tuple):
			Shape of the input image (height, width, channels). Default: (720, 1440, 7).
		num_classes (int):
			Number of segmentation classes (including background). Default: 1.
		depth (int):
			Depth of the U-Net++ architecture (number of downsampling levels). Default: 5.
		base_filter_dim (int):
			Number of filters in the first convolutional layer. Default: 32.
		activation (torch.nn.Module):
			Activation function to use after convolutional layers. Default: ReLU.
		pool* (nn.MaxPool2d):
			Max-pooling layer for downsampling.
		up* (nn.Upsample):
			Upsampling layer (bilinear with aligned corners).
		conv*_* (VGGBlock):
			Convolutional blocks in the U-Net architecture.
		final (nn.Conv2d):
			Final convolutional layer for output.

	Methods:
		__init__(*args, **kwargs):
			Initializes the U-Net++ model.
		forward(input):
			Performs a forward pass through the model, returning either a single output 
			tensor or a list of output tensors if deep supervision is enabled. 
	"""
	def __init__(self,
		input_shape: tuple = (180, 360, 7),
		num_classes: int = 1,
		depth: int = 5,
		base_filter_dim: int = 32,
		activation = torch.nn.modules.activation.ReLU(),
		*args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		# Define number of filers for each level
		nb_filter = [base_filter_dim*pow(2, n) for n in range(depth+1)] # [32, 64, 128, 256, 512]

		# Define input channels
		input_channels = input_shape[-1]

		# Define activation function
		self.activation = activation

		# Pooling layers
		self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
		self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
		self.pool3 = nn.MaxPool2d(kernel_size=(3, 3))
		self.pool4 = nn.MaxPool2d(kernel_size=(3, 3))
		self.pool5 = nn.MaxPool2d(kernel_size=(5, 5))

		# Upsampling layer
		self.up1 = nn.Upsample(scale_factor=(5, 5), mode='bilinear', align_corners=True)
		self.up2 = nn.Upsample(scale_factor=(3, 3), mode='bilinear', align_corners=True)
		self.up3 = nn.Upsample(scale_factor=(3, 3), mode='bilinear', align_corners=True)
		self.up4 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
		self.up5 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
		
		# Encoder layers
		self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
		self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
		self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
		self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
		self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
		self.conv5_0 = VGGBlock(nb_filter[4], nb_filter[5], nb_filter[5])

		# Decoder layers
		self.conv4_1 = VGGBlock(nb_filter[5] + nb_filter[4], nb_filter[4], nb_filter[4])
		self.conv3_2 = VGGBlock(nb_filter[4] + nb_filter[3], nb_filter[3], nb_filter[3])
		self.conv2_3 = VGGBlock(nb_filter[3] + nb_filter[2], nb_filter[2], nb_filter[2])
		self.conv1_4 = VGGBlock(nb_filter[2] + nb_filter[1], nb_filter[1], nb_filter[1])
		self.conv0_5 = VGGBlock(nb_filter[1] + nb_filter[0], nb_filter[0], nb_filter[0])

		# Final layer
		self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


	def forward(self, input):

		# Encoder
		x0_0 = self.conv0_0(input)
		x1_0 = self.conv1_0(self.pool1(x0_0))
		x2_0 = self.conv2_0(self.pool2(x1_0))
		x3_0 = self.conv3_0(self.pool3(x2_0))
		x4_0 = self.conv4_0(self.pool4(x3_0))
		x5_0 = self.conv5_0(self.pool5(x4_0))

		# Decoder
		x4_1 = self.conv4_1(torch.cat([self.up1(x5_0), x4_0], 1))
		x3_2 = self.conv3_2(torch.cat([self.up2(x4_1), x3_0], 1))
		x2_3 = self.conv2_3(torch.cat([self.up3(x3_2), x2_0], 1))
		x1_4 = self.conv1_4(torch.cat([self.up4(x2_3), x1_0], 1))
		x0_5 = self.conv0_5(torch.cat([self.up5(x1_4), x0_0], 1))

		output = self.activation(self.final(x0_5))
		return output
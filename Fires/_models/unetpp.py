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
class UnetPlusPlus(BaseLightningModule):
	def __init__(self,
		input_shape: tuple = (720, 1440, 8),
		num_classes: int = 1,
		depth: int = 4,
		base_filter_dim: int = 32,
		deep_supervision: bool = False,
		activation = torch.nn.modules.activation.ReLU(),
		*args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		nb_filter = [base_filter_dim*pow(2, n) for n in range(depth+1)] # [32, 64, 128, 256, 512]
		input_channels = input_shape[-1]
		self.deep_supervision = deep_supervision
		self.activation = activation

		self.pool = nn.MaxPool2d(2, 2)
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

		self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
		self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
		self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
		self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
		self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

		self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
		self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
		self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

		self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
		self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

		self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

		self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

		if self.deep_supervision:
			self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
			self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
			self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
			self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
		else:
			self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


	def forward(self, input):
		x0_0 = self.conv0_0(input)
		x1_0 = self.conv1_0(self.pool(x0_0))
		x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

		x2_0 = self.conv2_0(self.pool(x1_0))
		x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
		x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

		x3_0 = self.conv3_0(self.pool(x2_0))
		x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
		x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
		x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

		x4_0 = self.conv4_0(self.pool(x3_0))
		x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
		x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
		x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
		x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

		if self.deep_supervision:
			output1 = self.activation(self.final1(x0_1))
			output2 = self.activation(self.final2(x0_2))
			output3 = self.activation(self.final3(x0_3))
			output4 = self.activation(self.final4(x0_4))
			return [output1, output2, output3, output4]

		else:
			output = self.activation(self.final(x0_4))
			return output



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
	def __init__(self, in_channels, middle_channels, out_channels):
		super().__init__()
		# self.drop = nn.Dropout2d(p=0.5)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(middle_channels)
		self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		# out = self.drop(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		# out = self.drop(out)
		out = self.relu(out)

		return out

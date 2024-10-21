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

import torch
import torch.nn as nn
import numpy as np
from Fires._utilities.decorators import export

@export
class DiceLoss(nn.Module):
	def __init__(self, smooth=1.0):
		super(DiceLoss, self).__init__()
		self.smooth = smooth
		
	def forward(self, inputs, targets):
		# Applica sigmoid alle predizioni se sono logits
		inputs = torch.sigmoid(inputs)
		
		# Appiattisci i tensori
		inputs = inputs.view(-1)
		targets = targets.view(-1)
		
		# Calcola l'intersezione
		intersection = (inputs * targets).sum()
		
		# Calcola il coefficiente di Dice
		dice_coeff = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
		
		# Calcola la Dice Loss
		loss = 1 - dice_coeff
		
		return loss


@export
class TverskyLoss(nn.Module):
	def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
		super(TverskyLoss, self).__init__()
		self.alpha = alpha
		self.beta = beta
		self.smooth = smooth
		
	def forward(self, inputs, targets):
		# Applica sigmoid alle predizioni se sono logits
		# inputs = torch.sigmoid(inputs)
		
		# Appiattisci i tensori
		inputs = inputs.view(-1)
		targets = targets.view(-1)
		
		# Calcola TP, FP e FN
		TP = (inputs * targets).sum()
		FP = ((1 - targets) * inputs).sum()
		FN = (targets * (1 - inputs)).sum()
		
		# Calcola l'indice di Tversky
		tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
		
		# Calcola la Tversky Loss
		loss = 1 - tversky_index
		
		return loss


@export
class FocalLoss(nn.Module):
	def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction
		
	def forward(self, inputs, targets):
		# Calcola la Binary Cross Entropy Loss con logits
		BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
		
		# Calcola p_t
		pt = torch.exp(-BCE_loss)
		
		# Calcola la Focal Loss
		F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
		
		if self.reduction == 'mean':
			return F_loss.mean()
		elif self.reduction == 'sum':
			return F_loss.sum()
		else:
			return F_loss

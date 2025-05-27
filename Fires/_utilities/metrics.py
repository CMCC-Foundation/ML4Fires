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

        
        
	"""
	Combines Binary Cross Entropy (BCE) and L1 loss functions with configurable weights.

	This custom loss function enables a weighted combination of BCE and L1 loss, 
	providing a flexible approach for tasks that require both classification 
	accuracy and regression precision. It is particularly useful in scenarios such 
	as wildfire prediction, where balancing binary classification (e.g., burned vs. 
	unburned) and pixel-wise regression is crucial.

	Parameters
	----------
	weight_bce : float, optional
		Weight assigned to the BCE loss component. Defaults to 0.7.
	weight_l1 : float, optional
		Weight assigned to the L1 loss component. Defaults to 0.3.

	Notes
	-----
	- The sum of `weight_bce` and `weight_l1` does not need to be 1, but 
	it is recommended to ensure a balanced contribution.
	- The flexibility of this loss function allows tuning based on task-specific 
	needs and data characteristics.

	"""

@export
class WeightedBCE_L1Loss(nn.Module):
	def __init__(self, weight_bce=0.8, weight_l1=0.2):
		super().__init__()
		self.bce = nn.BCELoss()
		self.l1 = nn.L1Loss()
		self.weight_bce = weight_bce
		self.weight_l1 = weight_l1

	def forward(self, preds, targets):
		loss_bce = self.bce(preds, targets)
		loss_l1 = self.l1(preds, targets)
		return self.weight_bce * loss_bce + self.weight_l1 * loss_l1
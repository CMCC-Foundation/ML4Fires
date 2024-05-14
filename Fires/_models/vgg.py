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

import lightning.pytorch as pl
from typing import Any, List
import torch.nn as nn

from Fires._models.base import BaseVGG


class VGG_V1(BaseVGG):
    def __init__(self, 
            channels: List[int], 
            activation: nn.Module = nn.Identity, 
            kernel_size: int = 3, 
            *args: Any, **kwargs: Any) -> None:
        super().__init__(channels, activation, kernel_size, *args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=64, kernel_size=kernel_size, padding='same'), activation(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding="same"), activation(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding="same"), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"), activation(), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding="same"), activation(), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, padding="same"), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, padding="valid"), activation(), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding="valid"), activation(), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding="valid"), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Flatten(), 

            nn.Linear(in_features=512, out_features=512), activation(), 
            nn.Linear(in_features=512, out_features=256), activation(), 
            nn.Linear(in_features=256, out_features=128), activation(), 
            nn.Linear(in_features=128, out_features=64), activation(), 
            nn.Linear(in_features=64, out_features=channels[1]), 
        )



class VGG_V2(BaseVGG):
    def __init__(self, 
            channels: List[int], 
            activation: nn.Module = nn.Identity, 
            kernel_size: int = 3, 
            *args: Any, **kwargs: Any) -> None:
        super().__init__(channels, activation, kernel_size, *args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=32, kernel_size=kernel_size, padding='same'), activation(), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"), activation(), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"), activation(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"), activation(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"), activation(), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"), activation(), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding="same"), activation(), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, padding="same"), activation(), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, padding="same"), activation(), 

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, padding="valid"), activation(), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding="valid"), activation(), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding="valid"), activation(), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding="valid"), activation(), 

            nn.Flatten(), 

            nn.Linear(in_features=512, out_features=1024), activation(), 
            nn.Linear(in_features=1024, out_features=512), activation(), 
            nn.Linear(in_features=512, out_features=256), activation(), 
            nn.Linear(in_features=256, out_features=128), activation(), 

            nn.Linear(in_features=128, out_features=channels[1]), 
        )



class VGG_V3(BaseVGG):
    def __init__(self, 
            channels: List[int], 
            activation: nn.Module = nn.Identity, 
            kernel_size: int = 3, 
            *args: Any, **kwargs: Any) -> None:
        super().__init__(channels, activation, kernel_size, *args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=32, kernel_size=kernel_size, padding='same'), activation(), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"), activation(), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding="same"), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"), activation(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"), activation(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"), activation(), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"), activation(), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"), activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding="same"), activation(), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, padding="same"), activation(), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, padding="same"), activation(), 

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, padding="valid"), activation(), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding="valid"), activation(), 

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2, padding="valid"), activation(), 
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, padding="valid"), activation(), 

            nn.Flatten(), 

            nn.Linear(in_features=1024, out_features=1024), activation(), 
            nn.Linear(in_features=1024, out_features=512), activation(), 
            nn.Linear(in_features=512, out_features=512), activation(), 
            nn.Linear(in_features=512, out_features=256), activation(), 

            nn.Linear(in_features=256, out_features=channels[1]),             
        )

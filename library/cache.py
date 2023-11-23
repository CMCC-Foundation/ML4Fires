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

import os
import pickle
from .decorators import debug, export
from .macros import LOG_DIR
from .logger import Logger as logger
_cache_log = logger(log_dir=LOG_DIR).get_logger(log_name="cache")

@export
@debug(log=_cache_log)
def read_cache(cache_path: str):
    cached = None
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
    return cached

@export
@debug(log=_cache_log)
def write_cache(ds, cache_path: str):
    with open(cache_path, "wb") as f:
        pickle.dump(ds, f)
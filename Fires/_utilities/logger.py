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

import os
import json
import logging
import requests
from datetime import datetime as dt

class Logger():
	"""
	Class used to define multiple loggers in order to write
	logs in different log files.
	"""
	def __init__(self, log_dir:str, filename:str="logfile.log") -> None:
		"""
		Creates and initialize an instance of Logger() class.
		
		Parameters
		----------
		log_dir : str
			path to log folder, it's recommended to use `os.path.join(os.getcwd(), 'logs')`
		filename : str
			name of the log file, it must end with `.log` or `.txt` extension

		Raises
		------
		NameError
			if the filename doesn't end with `.log` or `.txt`, it raises a `NameError()` exception
		"""
		
		ext = filename.split('.')[1]
		if ext not in ['txt', 'log']:
			raise NameError(f"Received {ext} Expected TXT or LOG extension")
		
		today = dt.today().strftime(format="%Y%m%d")
		self.filename = f"{today}_{filename}"
		self.log_dir = log_dir
		self.log_format = '%(asctime)s|%(levelname)s|%(name)s|\t %(message)s'
		self.date_format = '%Y-%m-%d %I:%M:%S %p'
		self.log_level = logging.DEBUG
		self.formatter = logging.Formatter(fmt=self.log_format, datefmt=self.date_format)

		logging.basicConfig(
			level=self.log_level,
			filename=os.path.join(self.log_dir, self.filename),
			format=self.log_format,
			datefmt=self.date_format
		)

	def get_logger(self, log_name, level=logging.INFO):
		"""
		Creates the logger.

		Parameters
		----------
		log_name : str
			defines the name of the logger and it's used to create the specific logger folder
			where to store the log file
		level : logging level, optional
			defines the base logging level, by default logging.INFO

		Returns
		-------
		logger
			Return a logger with the specified name.
		"""
		spec_log_dir = os.path.join(self.log_dir, log_name)
		os.makedirs(spec_log_dir, exist_ok=True)
		fname = os.path.join(spec_log_dir, self.filename)
		
		handler = logging.FileHandler(fname)
		handler.setFormatter(self.formatter) 
		self.logger = logging.getLogger(log_name)
		self.logger.propagate = False
		self.logger.setLevel(level)

		if self.logger.hasHandlers():
				self.logger.handlers.clear()
			
		self.logger.addHandler(handler)
		return self.logger
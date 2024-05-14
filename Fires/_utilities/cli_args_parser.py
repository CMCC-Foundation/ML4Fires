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

import argparse
from typing import List, Tuple

from Fires._utilities.decorators import export

@export
class CLIParser():
	"""
	A versatile class for creating robust command-line interfaces with built-in error handling and input validation.

	Provides methods to:
	- Initialize the parser with program name and description.
	- Add arguments and default values.
	- Create mutually exclusive argument groups.
	- Parse arguments, handle errors, and return parsed values.
	"""

	def __init__(self, program_name:str, description:str) -> None:
		"""
		Initializes the CLI parser with the provided program name and description.

		Args:
			program_name (str):
				The name of the program displayed in usage messages.
			description (str):
				A brief description of the program's purpose.
		"""
		self.parser = argparse.ArgumentParser(prog=program_name, description=description)

	def add_arguments(self, parser:argparse.ArgumentParser, options:List[Tuple[str, str, dict]]) -> None:
		"""
		Adds a List of Tuple arguments with specified name, type checking, and validation.

		Args:
			parser (argparse.ArgumentParser):
				The parser that must be updated with new commands.

			options (List[Tuple[str, str, dict]]):
				A list of tuples defining the arguments. Each tuple should contain:
					- (str): Short name (e.g., "-v").
					- (str): Long name (e.g., "--verbose").
					- (dict): Keyword arguments for `argparse.add_argument()` (optional).
		"""
		if parser is None:
			parser = self.parser

		for opt in options:
			names, kwargs = opt
			s_name, l_name = names
			parser.add_argument(s_name, l_name, **kwargs)
		
	def add_mutually_exclusive_group(self, options:List[Tuple[str, str, dict]]) -> None:
		"""
		Creates a mutually exclusive group of arguments, where only one can be provided at a time.

		Args:
			options (List[Tuple[str, str, dict]]): A list of tuples defining the arguments.
				Each tuple should contain:
					- (str): Short name (e.g., "-v").
					- (str): Long name (e.g., "--verbose").
					- (dict): Keyword arguments for `argparse.add_argument()` (optional).
		"""
		mutually_exclusive_group = self.parser.add_mutually_exclusive_group()
		self.add_arguments(parser=mutually_exclusive_group, options=options)
	
	def parse_args(self) -> argparse.Namespace:
		"""
		Parses command-line arguments, handles potential errors, and returns the parsed arguments.

		Returns:
			argparse.Namespace: An object containing parsed arguments as attributes.

		Raises:
			argparse.ArgumentError: If an invalid argument is provided or validation fails.
		"""
		self.parser.print_help()
		try:
			args = self.parser.parse_args()
			return args
		except argparse.ArgumentError as e:
			print(f"Error: {e}")
			exit(1)
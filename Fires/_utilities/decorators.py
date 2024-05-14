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

import functools
import sys
__all__ = ['debug', 'export']


def export(fn):
	"""
	Registers a function for export by potentially adding its name to the
	`__all__` attribute of its module.

	Args:
		fn (callable): The function to be exported.

	Returns:
		callable: The decorated function.
	"""

	mod = sys.modules[fn.__module__]
	if hasattr(mod, '__all__'):
		if fn.__name__ not in mod.__all__:
			mod.__all__.append(fn.__name__)
	else:
		mod.__all__ = [fn.__name__]
	return fn


def debug(log=None):
	"""
	Decorator that logs the function signature, return value, and separates
	calls with a separator line.

	Args:
		log (object, optional):
			An object (typically a logger) that has an `info` method for logging messages.
			Defaults to None, in which case messages are printed to the standard output.

	Returns:
		decorator: A decorator function that can be used to wrap other functions.
	"""
	@functools.wraps(log)
	def decorator(func):
		"""
		Inner wrapper function that logs the function call and return value.
		"""
		@functools.wraps(func)
		def wrapper_debug(*args, **kwargs):
			args_repr = [repr(a) for a in args]
			kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
			signature = ", ".join(args_repr + kwargs_repr)
			
			if log is not None:
				log.info(f"Calling {func.__name__}({signature})")
			else:
				print(f"Calling {func.__name__}({signature})")

			value = func(*args, **kwargs)
			
			if log is not None:
				log.info(f"{func.__name__!r} returned {value!r}")
				log.info("\n\n ---------------------------------------------- \n\n")
			else: 
				print(f"{func.__name__!r} returned {value!r}")
				print("\n\n ---------------------------------------------- \n\n")
		
			return value
		return wrapper_debug
	return decorator

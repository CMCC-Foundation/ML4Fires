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

import time
from .decorators import export

@export
class Timer():
	def __init__(self, timers=['tot_exec_elapsed_time', 'io_elapsed_time', 'training_elapsed_time']):
		# initialize execution times data structure
		self.exec_times = {}
		self.partials = {}
		for t in timers:
			self.exec_times.update({t:0})
			self.partials.update({t:0})

	def start(self, timer):
		# update partial timers to start counting
		self.partials.update({timer:-time.time()})

	def stop(self, timer):
		# add ( stop - start ) time to global execution time
		self.exec_times[timer] += self.partials[timer] + time.time()
		# reset partial
		self.partials[timer] = 0


@export
def init_timer(runtype):
	"""
	Initializes the timer for ML workflows. Time is divided into:
	1. total execution time
	2. io execution time
	3. training/test execution time

	"""
	# define timer names
	tot_exec_timer = 'tot_exec_elapsed_time'
	io_timer = 'io_elapsed_time'
	if runtype == 'training':
		exec_timer = 'training_elapsed_time'
	elif runtype == 'test':
		exec_timer = 'inference_elapsed_time'
	# running time setup
	timer = Timer(timers=[tot_exec_timer, io_timer, exec_timer])
	return timer, tot_exec_timer, io_timer, exec_timer
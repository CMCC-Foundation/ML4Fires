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
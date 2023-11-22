import os
import numpy as np
import tensorflow as tf
from .cache import read_cache
from .decorators import debug, export
from .macros import LOG_DIR, SCALER_DIR
from .logger import Logger as logger
_standard_log = logger(log_dir=LOG_DIR).get_logger(log_name="Standard Scaler")
_minmax_log = logger(log_dir=LOG_DIR).get_logger(log_name="MinMax Scaler")

@export
@debug(log=_standard_log)
class StandardScaler():
	"""
	Scaler class for Tensorflow datasets. It takes in input a 
	tf.data.Dataset and returns an object that can be used for scaling 

	"""
	
	@debug(log=_standard_log)
	def __init__(self, ndims=2, dtype=tf.float32, drivers_info:list=[]) -> None:
		"""
		Parmeters
		---------
		ndims : int | default : 2
			Number of dimensions of the input data, excluding the batch (usually the first dim).
		dtype : tf.dtype | default : float32
			datatype of the computed scaler and output data.

		"""
		self.logger = _standard_log
		self.ndims = ndims
		self.dtype = dtype
		self.drivers = drivers_info[0].vars
		self.targets = drivers_info[1].vars
		self.len_drivers = len(self.drivers)
		self.len_targets = len(self.targets)

		

	@debug(log=_standard_log)
	def fit(self):

		# load maps
		self.mean_drv, self.stdv_drv, self.mean_trg, self.stdv_trg = self.__load_maps(op1='mean', op2='stdv')
	

	@debug(log=_standard_log)
	def __check_nans(self, map_type, op1_np, op2_np):
		op1_np_isnan = np.isnan(op1_np).all()
		op2_np_isnan = np.isnan(op2_np).all()
		if op1_np_isnan and op2_np_isnan:
			raise ValueError(f"Loaded maps are NaN \n {map_type.upper()} OP1: {op1_np_isnan} \n {map_type.upper()} OP2: {op2_np_isnan}")
		else:
			self.logger.info(f"{map_type.upper()} maps: \n {op1_np} \n {op2_np}")
	

	@debug(log=_standard_log)
	def __load_maps(self, op1, op2):
		
		# get map filepaths
		map_op1 = os.path.join(SCALER_DIR, f"map_trn_{op1}.npy")
		map_op2 = os.path.join(SCALER_DIR, f"map_trn_{op2}.npy")

		# read maps from cache files
		cache_op1 = np.load(map_op1)
		cache_op2 = np.load(map_op2)
		
		# select driver features from maps
		op1_drv = cache_op1[:, :, :self.len_drivers]
		op2_drv = cache_op2[:, :, :self.len_drivers]
		
		# check if numpy maps are completely filled with NaN values
		self.__check_nans(map_type='drv', op1_np=op1_drv, op2_np=op2_drv)

		target_ba = ['fcci_ba', 'gwis_ba', 'merge_ba']
		target_ba_vm = ['fcci_ba_valid_mask', 'gwis_ba_valid_mask', 'merge_ba_valid_mask']
		list_of_targets = target_ba + target_ba_vm
		
		if self.len_targets == 1 and self.targets[0] in list_of_targets:
			pos_item = list_of_targets.index(self.targets[0]) + self.len_drivers
			self.logger.info(f"Target position: {pos_item}")
			op1_trg = cache_op1[:, :, pos_item]
			op2_trg = cache_op2[:, :, pos_item]
			op1_trg = op1_trg.reshape(720, 1440, 1)
			op2_trg = op2_trg.reshape(720, 1440, 1)
		
		# check if numpy maps are completely filled with NaN values
		self.__check_nans(map_type='trg', op1_np=op1_trg, op2_np=op2_trg)

		return op1_drv, op2_drv, op1_trg, op2_trg
	
			
	@debug(log=_standard_log)
	def transform(self, input_tensor, features_type:str='input'):

		if features_type.lower() not in ['input', 'output']:
			raise ValueError(f"Supported only 'input' and 'output' (received: '{features_type.lower()}')")
		self.features_type = features_type.lower()
		
		# compute scaling factor and term to scale data
		self.__standard_transform()
		
		# compute division and get scaled output tensor
		out_tensor = tf.math.multiply_no_nan(input_tensor, self.scale_) + self.min_	
		self.logger.info(f"Scaled \n {out_tensor}")

		# fill output tensor NaN values with 0
		fillnan = tf.zeros_like(out_tensor)
		out_tensor = tf.where(tf.math.is_nan(out_tensor), fillnan, out_tensor)
		self.logger.info(f"Scaled (no NaN) \n {out_tensor}")

		# return float32 output tensor
		return tf.cast(out_tensor, dtype=self.dtype)
	

	@debug(log=_standard_log)
	def __standard_transform(self):
		data_stdv = self.stdv_drv if self.features_type == 'input' else self.stdv_trg
		data_mean = self.mean_drv if self.features_type == 'input' else self.mean_trg
		
		# compute the scaling factor
		self.scale_ = tf.math.divide_no_nan(tf.cast(1, dtype=tf.float32), data_stdv)
		
		# compute the minimum term (division between mean and dtandard deviation values)		
		self.min_ = - tf.math.multiply_no_nan(data_mean, self.scale_)


@export
@debug(log=_minmax_log)
class MinMaxScaler():
	"""
	Scaler class for Tensorflow datasets. It takes in input a 
	tf.data.Dataset and returns an object that can be used for scaling 

	"""
	
	@debug(log=_minmax_log)
	def __init__(self, ndims=2, feature_range=(0,1), dtype=tf.float32, drivers_info:list=[]) -> None:
		"""
		Parmeters
		---------
		ndims : int | default : 2
			Number of dimensions of the input data, excluding the batch (usually the first dim).
		feature_range : tuple | default : (0,1)
			output range of input features
		dtype : tf.dtype | default : float32
			datatype of the computed scaler and output data.

		"""
		self.logger = _minmax_log
		self.ndims = ndims
		self.feature_range = feature_range
		self.dtype = dtype
		self.drivers = drivers_info[0].vars
		self.targets = drivers_info[1].vars
		self.len_drivers = len(self.drivers)
		self.len_targets = len(self.targets)
				

	@debug(log=_minmax_log)
	def fit(self):

		# load numpy maps
		self.min_drv, self.max_drv, self.min_trg, self.max_trg = self.__load_maps(op1='min', op2='max')
		
		# compute denominator as difference between max and min maps
		self.den_drv = np.subtract(self.max_drv, self.min_drv)
		self.den_trg = np.subtract(self.max_trg, self.min_trg)
	

	@debug(log=_minmax_log)
	def __check_nans(self, map_type, op1_np, op2_np):
		op1_np_isnan = np.isnan(op1_np).all()
		op2_np_isnan = np.isnan(op2_np).all()
		if not op1_np_isnan and not op2_np_isnan:
			raise ValueError(f"Loaded maps are NaN \n {map_type.upper()} OP1: {op1_np_isnan} \n {map_type.upper()} OP2: {op2_np_isnan}")
		else:
			self.logger.info(f"{map_type.upper()} maps: \n {op1_np} \n {op2_np}")
	

	@debug(log=_minmax_log)
	def __load_maps(self, op1, op2):
		
		# get map filepaths
		map_op1 = os.path.join(SCALER_DIR, f"map_trn_{op1}.npy")
		map_op2 = os.path.join(SCALER_DIR, f"map_trn_{op2}.npy")

		# read maps from cache files
		cache_op1 = np.load(map_op1)
		cache_op2 = np.load(map_op2)
		
		# select features from maps
		op1_drv = cache_op1[:, :, :8]
		op2_drv = cache_op2[:, :, :8]
		
		# check if numpy maps are completely filled with NaN values
		self.__check_nans(map_type='drv', op1_np=op1_drv, op2_np=op2_drv)
		
		target_ba = ['fcci_ba', 'gwis_ba', 'merge_ba']
		target_ba_vm = ['fcci_ba_valid_mask', 'gwis_ba_valid_mask', 'merge_ba_valid_mask']
		list_of_targets = target_ba + target_ba_vm

		if self.len_targets == 1 and self.targets[0] in list_of_targets:
			pos_item = list_of_targets.index(self.targets[0]) + 8
			self.logger.info(f"Target position: {pos_item}")
			op1_trg = cache_op1[:, :, pos_item]
			op2_trg = cache_op2[:, :, pos_item]
			op1_trg = op1_trg.reshape(720, 1440, 1)
			op2_trg = op2_trg.reshape(720, 1440, 1)
		
		# check if numpy maps are completely filled with NaN values
		self.__check_nans(map_type='trg', op1_np=op1_trg, op2_np=op2_trg)
		
		# fill NaN values with zeros
		op1_trg = np.nan_to_num(op1_trg)
		op2_trg = np.nan_to_num(op2_trg)

		return op1_drv, op2_drv, op1_trg, op2_trg
	
			
	@debug(log=_minmax_log)
	def transform(self, input_tensor, features_type:str='input'):

		if features_type.lower() not in ['input', 'output']:
			raise ValueError(f"Supported only 'input' and 'output' (received: '{features_type.lower()}')")
		self.features_type = features_type.lower()

		# fill NaN values in input tensor with 0
		# input_tensor = tf.where(tf.math.is_nan(input_tensor), tf.zeros_like(input_tensor), input_tensor)

		# compute scaling factor and minimum term to scale data
		self.__minmax_transform()
		out_tensor = tf.math.multiply_no_nan(input_tensor, self.scale_) + self.min_

		# fill NaN values with -1
		fillnan = -tf.ones_like(out_tensor)
		out_tensor = tf.where(tf.math.is_nan(out_tensor), fillnan, out_tensor)
		
		# return float32 output tensor
		return tf.cast(out_tensor, dtype=self.dtype)
	
		
	@debug(log=_minmax_log)
	def __minmax_transform(self):
		data_den = self.den_drv if self.features_type == 'input' else self.den_trg
		data_min = self.min_drv if self.features_type == 'input' else self.min_trg

		# compute the scaling factor
		self.scale_ = tf.math.divide_no_nan(tf.cast((self.feature_range[1] - self.feature_range[0]), dtype=tf.float32), data_den)		
		
		# compute the minimum term (division between min and the difference between max and min values)		
		self.min_ = self.feature_range[0] - tf.math.multiply_no_nan(data_min, self.scale_)
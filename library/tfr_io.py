import os
import numpy as np
import tensorflow as tf
from .decorators import debug, export
from .macros import LOG_DIR
from .logger import Logger as logger
log = logger(log_dir=LOG_DIR)

@export
class DriverInfo():
	"""
	Class storing information about driver variables, shape and datatype.
	"""
	def __init__(self, vars, shape, dtype=tf.float32):
		self.vars = vars
		self.shape = shape
		self.dtype = dtype


@export
class TensorCoder():
	"""
	Tensor Encoder and Decoder class. It provides two functionalities:
	1) encodes variables tensors into a serialized version to be stored as TFRecord
	2) decodes TFRecords serialized data into tensors usable in ML pipelines

	"""
	def __init__(self, drivers_info=[]):
		"""
		Parameters
		----------
		drivers_info : list(DriverInfo)
			A list of driver information for the decoder to be read from file.
		"""
		self.drivers_info = drivers_info
		self.logger = log.get_logger("Tensor Coder")


	def encoding_fn(self, **kwargs):
		"""
		Builds a serialized version of the dataset. kwargs must be tensors.
		"""
		# feature dictionary
		feature = {}
		# for each keyword argument
		for key, value in kwargs.items():
			# add the serialized variable to feature dictionary
			feature.update({key:self.tensor_feature(value)})
		# define features using the feature dictionary
		features = tf.train.Features(feature=feature)
		# serialize data examples
		return tf.train.Example(features=features).SerializeToString()


	def decoding_fn(self, serialized_data):
		"""
		Decoding function for a dataset written to disk as tensor_encoding_fn()
		"""
		# define features dictionary
		features = {}
		# cycle with respect to driver info list
		for info in self.drivers_info:
			for var in info.vars:
				# add features for each variable
				features.update({var : tf.io.FixedLenFeature([], tf.string)})
		# parse the serialized data so we get a dict with our data.
		parsed_data = tf.io.parse_single_example(serialized_data, features=features)
		# accumulator for data elements
		data = []
		# for each output tensor driver
		for info in self.drivers_info:
			self.logger.info(f"Shape: \t {info.shape} \n Variables: \n {info.vars}")
			# parsed single examples to tensors and stack them
			if len(info.vars) == 1 and len(info.shape) == 1:
				data_tensor = tf.ensure_shape(tf.io.parse_tensor(serialized=parsed_data[info.vars[0]], out_type=info.dtype), shape=info.shape)
			else:
				data_tensor = tf.stack([tf.ensure_shape(tf.io.parse_tensor(serialized=parsed_data[var], out_type=info.dtype), shape=info.shape) for var in info.vars], axis=-1)
			data.append(data_tensor)
		if len(data)==1:
			return data[0]
		return tuple(data)
	
	def tensor_feature(self, value):
		"""
		Returns a bytes_list from a tensor.
		"""
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()]) )



@export
class TensorWriter():
	"""
	Record Writer for Tensorflow TFRecords dataset

	"""
	def __init__(self, drivers_info=[]):
		"""
		Parameters
		----------
		drivers_info : list(DriverInfo)
			List of driver information about the data to be stored.
			e.g.
				drv_info = DriverInfo(vars=['t2m', 'r', ...], shape=(256, 512))
				tar_info = DriverInfo(vars=['latlon', 'coo', ...], shape=(2,))
				drivers_info = [drv_info, tar_info]
				writer = TensorWriter(drivers_info)
		"""
		self.reader = TensorCoder(drivers_info=drivers_info)
		self.drivers_info = drivers_info
	
	def write(self, drivers:tuple, outfile:str, n:int):
		"""
		Writes to disk the data drivers to be stored.

		Parameters
		----------
		drivers : tuple
			It is a tuple of data elements, each of which has shape N x ... x C
		"""
		# skip if file already exists
		if os.path.exists(outfile):
			return
		# open the tf record writer
		with tf.io.TFRecordWriter(outfile) as writer:
			for i in range(n):
				# create the structure to contain driver data
				data = dict()
				# iterate over drivers and their info
				for driver, info in zip(drivers, self.drivers_info):
					# for each variable add the driver element to the dictionary
					for v, var in enumerate(info.vars):
						if len(info.shape)==2:
							data.update({var:driver[i,:,:,v]})
						elif len(info.shape)==1:
							data.update({var:driver[i,:,v]})
				# create a binary record of the shard
				record = self.reader.encoding_fn(**data)
				# write the record on disk
				writer.write(record)
		return


@export
class DatasetWriter():
	"""
	Dataset Writer is a class that supports easy TFRecord data writing to disk. It automatically shards the dataset into shard sizes elements.

	"""

	def __init__(self, drivers_info:list, shard_size:int, dtype=np.float32) -> None:
		self.dtype = dtype
		self.shard_size = shard_size
		self.drivers_info = drivers_info
		self.writer = TensorWriter(drivers_info=drivers_info)


	def write(self, dst:str, data:tuple, **kwargs):
		"""
		Shards the data and store it to disk on the destination folder.

		Parameters
		----------
		dst: str
			Pathlike destination folder to store the data.
		data: tuple
			Data to be stored as numpy arrays. Data should look like this:
			# X.shape == N x ... x C # channel last notation
			# y.shape == N x ... x C # channel last notation
			X, y = ... # build data as you need
			data = [X, y]
		shard_size: int
			Size of each shard file.
		kwargs
			Additional arguments to build the filename of the shard.
			e.g.
			kwargs = {patch_type:'nearest', year:1980}
			# dst_file will be like this:
			dst_file = f'{patch_type}_{year}_shard_{i+1}_data_{shard_n}.tfrecord'
		"""
		# get total number of patches
		n = data[0].shape[0]

		# compute the number of shards 
		n_shards = self.get_num_shards(n=n, shard_size=self.shard_size)

		# for each shard
		for i in range(n_shards):
			shard_data = []

			# get the shard from each input map
			for dd in data:
				shard_data.append(dd[(i * self.shard_size):((i+1) * self.shard_size),])
			
			# get the number of elements into the shard
			shard_n = shard_data[0].shape[0]

			# define the output filename
			dst_file = ''
			for _,value in kwargs.items():
				dst_file += str(value)+'_'
			
			shard_idx = '0'+str(i+1) if i+1 in list(range(10)) else str(i+1)
			dst_file += f'shard_{shard_idx}_data_{shard_n}.tfrecord'
			dst_fpath = os.path.join(dst, dst_file)

			# convert the shard data to a tuple
			shard_data = tuple(shard_data)

			# write record to disk
			self.writer.write(drivers=shard_data, outfile=dst_fpath, n=shard_n)

	
	def get_num_shards(self, n, shard_size=64):
		"""Get the number of shards of {shard_size} items to be saved into storage."""
		n_shards = n // shard_size
		if n % shard_size:
			n_shards += 1 # add one shard if there are any remaining samples
		return n_shards

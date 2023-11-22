import inspect
import tensorflow as tf
from .layers import ConvUnit
from .decorators import export, debug
from .hyperparams import max_pool_params, max_pool_params_05_10, max_pool_params_03_03
from .macros import LOG_DIR

from .logger import Logger as logger
_old_unet_log = logger(log_dir=LOG_DIR).get_logger(log_name="Old UNET++")
_new_unet_log = logger(log_dir=LOG_DIR).get_logger(log_name="New UNET++")

from .configuration import load_global_config
toml_general = load_global_config()
toml_model = eval(toml_general['toml_configuration_files']['toml_model'])


@export
@debug(log=_old_unet_log)
class UNETPlusPlus():
	"""
	Class to define UNET++ Tensorflow model.
	"""

	def __init__(self, input_shape:tuple, num_classes:int=1, depth:int=4, base_filter_dim:int=32, deep_supervision:bool=False) -> None:
		
		if input_shape is None:
			raise ValueError(f"Input shape must be a tuple and not {input_shape}")
		
		self.logger = _old_unet_log
		self.input_shape=input_shape
		self.num_classes=num_classes
		self.depth=depth
		self.base_filter_dim=base_filter_dim
		self.deep_supervision=deep_supervision
		self.filters=[base_filter_dim*pow(2, i) for i in range(depth+1)]
		self.input_layer = tf.keras.layers.Input(shape=input_shape, name='main_input')
		self.initializer = toml_model['layers']['initializer']
		self.padding = toml_model['layers']['padding']
		self.activation = toml_model['layers']['activation']
		self.regularizer = eval(toml_model['layers']['regularizer'])


	@debug(log=_old_unet_log)
	def build(self):
		fn_name = inspect.currentframe().f_code.co_name

		# compute encoder layers
		self.__EncoderBranch(input=self.input_layer, filters=self.filters)

		# compute middle layers
		self.mid_layers_structure = [[layer] for _, layer in enumerate(self.outputs)]
		self.__MiddleLayers()

		# compute deep supervision layers
		self.__DeepSupervisionLayers()

		# create UNET++ model
		self.model = tf.keras.models.Model(inputs=self.input_layer, outputs=self.outs)

		# create model summary
		self.model.summary(print_fn=lambda x: self.logger.info(f'{x}'))

		self.logger.info(f'{fn_name} | Returning UNET++ Model')
		return self.model


	@debug(log=_old_unet_log)
	def __EncoderBranch(self, input, filters):
		fn_name = inspect.currentframe().f_code.co_name
		self.logger.info(f'{fn_name} | Encoder branch')

		skips, self.outputs = [], []
		x = input
		for i, filter in enumerate(filters):
			if i < len(filters)-1:
				skip, x = self.__EncodeUnit(x, stage=f'{i}0', filter=filter)
				skips.append(skip)
			else:
				x = ConvUnit(stage=f'{i}0', filter=filter, name=f"ENC_{i}0_Conv_Unit")(x)

		self.outputs = skips + [x]
	

	@debug(log=_old_unet_log)
	def __MiddleLayers(self):
		fn_name = inspect.currentframe().f_code.co_name
		self.logger.info(f'{fn_name} | Middle layers and Decoder branch computation')

		# list of input layers for each middle step
		self.main_inputs = self.outputs.copy()

		# iterate over steps
		for col in range(len(self.outputs)-1):

			# list of middle layers
			temp_list = []

			# iterate between 0 to input_layers list length - 1 in order to create middle layers
			for row in range(len(self.main_inputs)-1):

				# take a couple of input layers and reverse the order
				layers = self.main_inputs[row:row+2][::-1]

				x = self.__CreationUnit(inputs=layers, row=row, col=col, filter=self.filters[row], mid_layers=self.mid_layers_structure[row])
				# add the computed Decode Unit layer to the middle layers structure
				self.mid_layers_structure[row].append(x)
				# add the computed Decode Unit layer to the middle layers list
				temp_list.append(x)
			# set input layers list as middle layers list
			self.main_inputs = temp_list
		

	@debug(log=_old_unet_log)
	def __EncodeUnit(self, input, stage, filter):
		"""
		Encode Unit
		-----------
		The `Encode Unit` is used to build the UNET++ Encoder branch

		### Args:
		- `input` (keras Layer):\\
			input layer used to build encoder branch
		- `stage` (str):\\
			string that describes in which stage is the creation of UNET++ encoder branch
		- `filter` (int):\\
			filter used in the current stage

		### Returns:
		- `conv_unit`, `max_pool` (tuple):\\
			the first one is the skip connection used to build middle layers during UNET++ creation\\
			the second one is used as input to the next Encode Unit in order to build Encoder branch
		"""
		conv_unit = ConvUnit(filter=filter, stage=stage, name=f"ENC_{stage}_Conv_Unit")(input)
		max_pool = tf.keras.layers.MaxPooling2D(**max_pool_params(stage=stage))(conv_unit)
		return conv_unit, max_pool
	

	@debug(log=_old_unet_log)
	def __DecodeUnit(self, input1, input2, filter, stage):
		"""
		Decode Unit
		-----------
		The `Decode Unit` is used to build UNET++ middle layers and Decoder branch.

		### Args:
		- `input1` (keras Layer):\\
			this is used as input layer in the upsampling stage
		- `input2` (list, keras Layer):\\
			this is a list of layers and are used as input in the concatenation stage
		- `filter` (int):\\
			filter used during the current stage
		- `stage` (str):\\
			string that describes in which stage is the creation of UNET++ decoder branch

		### Returns:
		- `x` (Conv Unit Layer):\\
			Conv Unit layer that is at the end of Decode Unit block
		"""
		x = tf.keras.layers.Conv2DTranspose(filter, (2, 2), strides=(2, 2), name=f'DEC_{stage}_Upsampling', padding='same')(input1)
		x = tf.keras.layers.Concatenate(name=f'DEC_{stage}_Merge')([x]+input2)
		x = ConvUnit(filter=filter, stage=stage, name=f"DEC_{stage}_Conv_Unit")(x)
		return x


	@debug(log=_old_unet_log)
	def __CreationUnit(self, inputs, row, col, filter, mid_layers):
		"""
		CREATION UNIT
		-------------
		The `CreationUnit` is used to compute the input layers for the `DecodeUnit`.\\	
		The `in_1` variable is the reversed input layer and is the first input for Decode Unit.\\	
		The `in_2` variable is the list of layers to concatenate in the `DecodeUnit`.\\
		In the beginning, the `in_2` must contain only the second reversed input layer;\\
		later, it must contain the second reversed input layer + list of previous middle layers.\\
		These layers must be concatenated to the upsampled layer in the `DecodeUnit`.

		### Returns:
		- Decode Unit
		"""
		in_1 = inputs[0]
		in_2 = [inputs[1]] if col==0 else ([mid_layers] if len(mid_layers) == 1 else mid_layers)
		x = self.__DecodeUnit(input1=in_1, input2=in_2, stage=f'{row}{col+1}', filter=filter)
		return x


	@debug(log=_old_unet_log)
	def __DeepSupervisionLayers(self):
		fn_name = inspect.currentframe().f_code.co_name
		self.logger.info(f"{fn_name} | Creating Deep Supervision Layers")
		
		deep_layers = self.mid_layers_structure[0][1:]
		conv_dict = dict(
			filters=self.num_classes,
			kernel_size=(1, 1),
			activation='linear', #'sigmoid' tf.keras.layers.PReLU(),
			kernel_initializer=self.initializer,
			padding=self.padding,
			kernel_regularizer=self.regularizer)
		
		nested_outs=[tf.keras.layers.Conv2D(**conv_dict, name=f'output_{i+1}',)(x) for i, x in enumerate(deep_layers)]
		self.outs = [nested_outs[-1]] if not self.deep_supervision else nested_outs


@export
@debug(log=_new_unet_log)
class UNETPlusPlusNew():
	"""
	Class to define UNET++ Tensorflow model.
	"""
	
	def __init__(self, input_shape:tuple, num_classes:int=1, depth:int=7, base_filter_dim:int=16, deep_supervision:bool=False) -> None:
		
		if input_shape is None:
			raise ValueError(f"Input shape must be a tuple and not {input_shape}")
		
		self.logger = _new_unet_log
		self.input_shape=input_shape
		self.num_classes=num_classes
		self.depth=depth
		self.base_filter_dim=base_filter_dim
		self.deep_supervision=deep_supervision
		self.filters=[base_filter_dim*pow(2, i) for i in range(depth+1)]
		self.input_layer = tf.keras.layers.Input(shape=input_shape, name='main_input')
		self.initializer = toml_model['layers']['initializer']
		self.padding = toml_model['layers']['padding']
		self.regularizer = eval(toml_model['layers']['regularizer'])


	@debug(log=_new_unet_log)
	def build(self):
		fn_name = inspect.currentframe().f_code.co_name

		# compute encoder layers
		self.__EncoderBranch()

		# compute middle layers
		self.mid_layers_structure = [[layer] for _, layer in enumerate(self.outputs)]
		self.__MiddleLayers()

		# compute deep supervision layers
		self.__DeepSupervisionLayers()

		# create UNET++ model
		self.model = tf.keras.models.Model(inputs=self.input_layer, outputs=self.outs)

		# create model summary
		self.model.summary(print_fn=lambda x: self.logger.info(f'{x}'))
		
		self.logger.info(f'{fn_name} | Returning UNET++ Model')
		return self.model


	@debug(log=_new_unet_log)
	def __EncoderBranch(self):
		fn_name = inspect.currentframe().f_code.co_name
		self.logger.info(f'{fn_name} | Encoder branch')

		skips, self.outputs = [], []
		x = self.input_layer
		for i, filter in enumerate(self.filters):

			if i < len(self.filters)-1:
				encunit_args = dict(input=x, stage=f'{i}0', filter=filter)
				skip, x = self.__EncodeUnit(**encunit_args)
				skips.append(skip)

			else:
				x = ConvUnit(stage=f'{i}0', filter=filter, name=f"ENC_{i}0_Conv_Unit")(x)

		self.outputs = skips + [x]


	@debug(log=_new_unet_log)
	def __EncodeUnit(self, input, stage, filter):
		"""
		Encode Unit
		-----------
		The `Encode Unit` is used to build the UNET++ Encoder branch

		### Args:
		- `input` (keras Layer):\\
			input layer used to build encoder branch
		- `stage` (str):\\
			string that describes in which stage is the creation of UNET++ encoder branch
		- `filter` (int):\\
			filter used in the current stage

		### Returns:
		- `conv_unit`, `max_pool` (tuple):\\
			the first one is the skip connection used to build middle layers during UNET++ creation\\
			the second one is used as input to the next Encode Unit in order to build Encoder branch
		"""
		level = int(stage[0])
		self.logger.info(f"Level (Stage): {level}")
		if level == 0:
			mpool_params = max_pool_params_05_10(stage=stage)
		elif level > 0 and level < 3:
			mpool_params = max_pool_params_03_03(stage=stage)
		else:
			mpool_params = max_pool_params(stage=stage)

		conv_unit = ConvUnit(filter=filter, stage=stage, name=f"ENC_{stage}_Conv_Unit")(input)
		max_pool = tf.keras.layers.MaxPooling2D(**mpool_params)(conv_unit)
		return conv_unit, max_pool
	

	@debug(log=_new_unet_log)
	def __MiddleLayers(self):
		fn_name = inspect.currentframe().f_code.co_name
		self.logger.info(f'{fn_name} | Middle layers and Decoder branch computation')

		# list of input layers for each middle step
		self.main_inputs = self.outputs.copy()

		# iterate over steps
		for col in range(len(self.outputs)-1):

			# list of middle layers
			temp_list = []

			# iterate between 0 to input_layers list length - 1 in order to create middle layers
			for row in range(len(self.main_inputs)-1):

				# take a couple of input layers and reverse the order
				layers = self.main_inputs[row:row+2][::-1]

				x = self.__CreationUnit(inputs=layers, row=row, col=col, filter=self.filters[row], mid_layers=self.mid_layers_structure[row])
				
				# add the computed Decode Unit layer to the middle layers structure
				self.mid_layers_structure[row].append(x)
				
				# add the computed Decode Unit layer to the middle layers list
				temp_list.append(x)
			
			# set input layers list as middle layers list
			self.main_inputs = temp_list
		

	@debug(log=_new_unet_log)
	def __CreationUnit(self, inputs, row, col, filter, mid_layers):
		"""
		CREATION UNIT
		-------------
		The `CreationUnit` is used to compute the input layers for the `DecodeUnit`.\\	
		The `in_1` variable is the reversed input layer and is the first input for Decode Unit.\\	
		The `in_2` variable is the list of layers to concatenate in the `DecodeUnit`.\\
		In the beginning, the `in_2` must contain only the second reversed input layer;\\
		later, it must contain the second reversed input layer + list of previous middle layers.\\
		These layers must be concatenated to the upsampled layer in the `DecodeUnit`.

		### Returns:
		- Decode Unit
		"""
		in_1 = inputs[0]
		in_2 = [inputs[1]] if col==0 else ([mid_layers] if len(mid_layers) == 1 else mid_layers)
		x = self.__DecodeUnit(input1=in_1, input2=in_2, stage=f'{row}{col+1}', filter=filter)
		return x


	@debug(log=_new_unet_log)
	def __DecodeUnit(self, input1, input2, filter, stage):
		"""
		Decode Unit
		-----------
		The `Decode Unit` is used to build UNET++ middle layers and Decoder branch.

		### Args:
		- `input1` (keras Layer):\\
			this is used as input layer in the upsampling stage
		- `input2` (list, keras Layer):\\
			this is a list of layers and are used as input in the concatenation stage
		- `filter` (int):\\
			filter used during the current stage
		- `stage` (str):\\
			string that describes in which stage is the creation of UNET++ decoder branch

		### Returns:
		- `x` (Conv Unit Layer):\\
			Conv Unit layer that is at the end of Decode Unit block
		"""
		i = int(stage[0])
		self.logger.info(f"Level (Stage): {i}")
		if i == 0:
			upsample_args = dict(filters=filter, kernel_size=(5, 10), strides=(5, 10), name=f'DEC_{stage}_Upsampling_5X10', padding='same')
		elif i > 0 and i < 3:
			upsample_args = dict(filters=filter, kernel_size=(3, 3), strides=(3, 3), name=f'DEC_{stage}_Upsampling_3X3', padding='same')
		else:
			upsample_args = dict(filters=filter, kernel_size=(2, 2), strides=(2, 2), name=f'DEC_{stage}_Upsampling', padding='same')

		x = tf.keras.layers.Conv2DTranspose(**upsample_args)(input1)
		x = tf.keras.layers.Concatenate(name=f'DEC_{stage}_Merge')([x]+input2)
		x = ConvUnit(filter=filter, stage=stage, name=f"DEC_{stage}_Conv_Unit")(x)
		return x


	@debug(log=_new_unet_log)
	def __DeepSupervisionLayers(self):
		fn_name = inspect.currentframe().f_code.co_name
		self.logger.info(f"{fn_name} | Creating Deep Supervision Layers")
		 
		deep_layers = self.mid_layers_structure[0][1:]
		conv_dict = dict(
			filters=self.num_classes,
			kernel_size=(1, 1),
			activation='linear', # 'sigmoid',
			kernel_initializer=self.initializer,
			padding=self.padding,
			kernel_regularizer=self.regularizer)
		
		nested_outs=[tf.keras.layers.Conv2D(**conv_dict, name=f'output_{i+1}',)(x) for i, x in enumerate(deep_layers)]
		self.outs = [nested_outs[-1]] if not self.deep_supervision else nested_outs




import tensorflow as tf
from .decorators import export
from .hyperparams import conv_params
from .configuration import load_global_config
toml_general = load_global_config()
toml_model = eval(toml_general['toml_configuration_files']['toml_model'])

@export
class ConvUnit(tf.keras.layers.Layer):
	
	def __init__(self, stage, filter, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
		super(ConvUnit, self).__init__(trainable, name, dtype, dynamic, **kwargs)

		self.stage = stage
		self.filter = filter
		self.conv_1 = tf.keras.layers.Conv2D(**conv_params(filter=filter, name=f"conv_{stage}_1"))
		self.conv_2 = tf.keras.layers.Conv2D(**conv_params(filter=filter, name=f"conv_{stage}_2"))
		self.dropout = tf.keras.layers.Dropout(rate=toml_model['layers']['drop_rate'])
	
	def call(self, inputs, training=False):
		x = self.conv_1(inputs)
		x = self.dropout(x, training)
		x = self.conv_2(x)
		x = self.dropout(x, training)
		return x
import tensorflow as tf
from .configuration import load_global_config
toml_general = load_global_config()
toml_model = eval(toml_general['toml_configuration_files']['toml_model'])

_layers_dict = toml_model['layers']

# Layers params
conv_params = lambda filter, name : dict(
    filters=filter, 
	kernel_size=eval(_layers_dict['kernel']), 
	activation=_layers_dict['activation'],
	name=name, 
	kernel_initializer=_layers_dict['initializer'], 
	padding=_layers_dict['padding'], 
	kernel_regularizer=eval(_layers_dict['regularizer']))

max_pool_params = lambda stage: dict(
    pool_size=eval(_layers_dict['maxpool_size']), 
    strides=eval(_layers_dict['maxpool_strides']),
	data_format=_layers_dict['maxpool_data_format'],
    name=f'{stage}_MaxPooling'
)

max_pool_params_05_10 = lambda stage: dict(
    pool_size = (5, 10),
	strides = (5, 10),
	data_format="channels_last",
    name=f'{stage}_MaxPooling_5X10'
)

max_pool_params_03_03 = lambda stage: dict(
    pool_size = (3, 3),
	strides = (3, 3),
	data_format="channels_last",
    name=f'{stage}_MaxPooling_3X3'
)
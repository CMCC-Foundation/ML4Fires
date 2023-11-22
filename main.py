# from workflow import Workflow
# from macros_itwn import SHIFT_LIST

# if __name__ == "__main__":
# 	wflow_args = dict(shift_list=SHIFT_LIST, day_delay=0, scaler_type='standard', study_case='AREAS')
# 	list_of_sources = ['FCCI', 'GWIS', 'MERGE']
# 	for target_source in [list_of_sources[0]]:
# 		workflow = Workflow(target_source=target_source, **wflow_args)
# 		workflow.train_model()


import os
import joblib
import pickle
import random
import tensorflow as tf
from glob import glob

import sys
_lib_dir = os.path.join(os.getcwd(), 'library')
if _lib_dir not in sys.path:
	sys.path.append(_lib_dir)

from library.decorators import debug, export
from library.macros import (LOG_DIR, DATA_DIR, SCALER_DIR, LOSS_METRICS_HISTORY_CSV, CHECKPOINT_FNAME, RUN_DIR)
from library.logger import Logger as logger
main_log = logger(log_dir=LOG_DIR).get_logger('Main')

from library.configuration import load_global_config
toml_general = load_global_config()
toml_model = eval(toml_general['toml_configuration_files']['toml_model'])

from library.dataset_builder_wf import WildFiresDatasetBuilder
from library.scaling import StandardScaler
from library.dataset_creator import DatasetCreator
from library.tfr_io import TensorCoder, DriverInfo
from library.models import UNETPlusPlusNew as UPPN
from library.augmentation import rot180, left_right, up_down



# define model configuration
data_dict = toml_general['data']
model_config = data_dict['selected_configuration']
main_log.info(f"Defining parameters: \n {toml_model[model_config]}")

# define target source (FCCI, GWIS or MERGE) and training  model parameters
target_source = toml_model[model_config]['target_source']				# target		: "FCCI"
batch_size = toml_model[model_config]['bsize']							# batch size	: 4
base_shape = eval(toml_model[model_config]['base_shape'])				# base shape	: (720, 1440)
in_shape = (*base_shape, 8)												# input shape	: (720, 1440, 8)
shard_size = toml_model[model_config]['shard_size']						# shard size	: 2
epochs = toml_model[model_config]['epochs']								# epochs		: 10
shift_list = toml_model[model_config]['shift_list']						# shift list	: [0] days
standard_scaler_type = toml_model[model_config]['scaler_type_zscore']	# scaler type	: 'standard' scaler
minmax_scaler_type = toml_model[model_config]['scaler_type_minmax']		# scaler type	: 'minmax' scaler
base_scaler_name = toml_model[model_config]['scaler_name']				# scaler name	: 'scaler.dump'
shuffle = eval(toml_model[model_config]['shuffle'])						# shuffle data	: True

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#	FILENAMES: DIVIDE TFRECORD FILES IN TRAINING AND VALIDATION, SHUFFLE FILENAMES
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# create path to tfrecord files
tfrecords_dir = data_dict['tfrecords_dir']
folder = os.path.join(DATA_DIR, target_source, tfrecords_dir)
main_log.info(f"Folder: {folder}")

# get tfrecord files and split in training and validation files
trn_filenames = sum(sorted(glob(f'{folder}/{year}*.tfrecord') for year in range(2001,2017)), [])
val_filenames = sum(sorted(glob(f'{folder}/{year}*.tfrecord') for year in range(2017,2019)), [])
main_log.info(f"\nTraining files \n {trn_filenames} \nValidation files \n{val_filenames}")

# set seed for random shuffling
if shuffle:
	seed = data_dict['seed']
	main_log.info(f"Set random seed: {seed}")

	# shuffle files
	random.seed(seed)
	random.shuffle(trn_filenames)
	random.shuffle(val_filenames)

main_log.info("Filenames have been shuffled")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#	TRAINING - DATASET CREATION: SELECT TARGET SOURCE, GET DRIVERS INFO, CREATE TENSOR CODER, CREATE DATASET BUILDER
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# create dictionary with FwiDatasetBuilder args
dataset_creator = DatasetCreator(target_source=target_source, shift_list=shift_list)
_, drivers_info = dataset_creator.build()

# pop target variable depending on study case
study_case = eval(toml_model['model']['ba_hectares'])
pop_item = 1 if study_case else 0
drivers_info[1].vars.pop(pop_item)

main_log.info(f"Driver info: \n vars \t {drivers_info[0].vars}\n shape: \t {drivers_info[0].shape}")
main_log.info(f"Target info: \n vars \t {drivers_info[1].vars}\n shape: \t {drivers_info[1].shape}")

# create tensor coder
tensor_coder = TensorCoder(drivers_info=drivers_info)
main_log.info("Creating Tensor Coder")

# create augmentation dictionary
aug_dict = dict(rot180=rot180, left_right=left_right, up_down=up_down)

def get_WF_DS_Builder(filenames:list):
	return (WildFiresDatasetBuilder(epochs=epochs, tensor_coder=tensor_coder, filenames=filenames)
		.augment(aug_fns = aug_dict)
		.assemble_dataset()
		.batch(batch_size=batch_size))

# Training Dataset: create a WildFiresDatasetBuilder instance for training files
trn_ds_builder = get_WF_DS_Builder(filenames=trn_filenames)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#	TRAINING - DATASET CREATION: CREATE SCALER, SCALE DATASET, OPTIMIZE DATASET, ADD REPEATS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Training Dataset: create, fit and save scaler
main_log.info(f"Creating and fitting {standard_scaler_type.upper()} scaler")

scaler = StandardScaler(drivers_info=drivers_info)
scaler.fit()

try:
	scaler_fname = f"{target_source}_00_{standard_scaler_type}_{base_scaler_name}"
	scaler_path = os.path.join(SCALER_DIR, scaler_fname)
	joblib.dump(scaler, scaler_path)
	main_log.info(f"Scaler saved in {scaler_path}")
except:
	main_log.error(f"Error saving scalers in {scaler_path}")

# Training Dataset: scale the dataset
trn_ds_builder = trn_ds_builder.scale(scaler=scaler)

# Training Dataset: optimize the dataset
trn_ds_builder = trn_ds_builder.optimize()

# Training Dataset: get steps per epoch
trn_steps_per_epoch = round(trn_ds_builder.count/batch_size)

# Training Dataset: get the dataset
trn_dataset = trn_ds_builder.dataset.repeat(count=trn_steps_per_epoch*epochs)

main_log.info(f"Training dataset has been scaled. Training steps per epoch: {trn_steps_per_epoch}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#	VALIDATION - DATASET CREATION: GET DATASET BUILDER, SCALE DATA, OPTIMIZE DATASET, ADD REPETITIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Validation Dataset: create a WildFiresDatasetBuilder instance for validation files
val_ds_builder = get_WF_DS_Builder(filenames=val_filenames)

# Validation Dataset: scale the dataset
val_ds_builder = val_ds_builder.scale(scaler=scaler)

# Validation Dataset: optimize the dataset
val_ds_builder = val_ds_builder.optimize()

# Validation Dataset: get steps per epoch
val_steps_per_epoch = round(val_ds_builder.count/batch_size)

# Validation Dataset: get the dataset
val_dataset = val_ds_builder.dataset.repeat(count=val_steps_per_epoch*epochs)

main_log.info(f"Validation dataset has been scaled. Validation steps per epoch: {val_steps_per_epoch}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#	DATASET DISTRIBUTION: DEFINE LOSS, DEFINE MIRRORED STRATEGY, DISTRIBUTE DATASET, DEFINE MODEL, METRICS AND OPTIMIZER IN MIRRORED STRATEGY SCOPE
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# define losses, metrics and callbacks
loss = eval(toml_model['model']['loss'])
strategy = eval(toml_model['model']['strategy'])
main_log.info(f"Defined loss ({loss.name}) and mirrored strategy")

# Distribute Datasets: distribute training and validation datasets
distr_trn_dataset = strategy.experimental_distribute_dataset(trn_dataset)
distr_val_dataset = strategy.experimental_distribute_dataset(val_dataset)
main_log.info("Datasets have been distributed")

# define metrics and optimizer in MirroredStrategy scope
with strategy.scope():
	metrics = eval(toml_model['model']['metrics'])
	lr = toml_model['model']['learning_rate']
	optimizer = eval(toml_model['model']['optimizer'])

	# define and compile the model
	model = UPPN(input_shape=in_shape, num_classes=1).build()
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#	DEFINE CALLBACKS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# define dict with callbacks args
cllbk_dict = dict(monitor='val_loss', verbose=1, mode='min')
# define callbacks
csvlogger_fname = LOSS_METRICS_HISTORY_CSV(trgt_src=target_source)
csvlogger_cllbk = tf.keras.callbacks.CSVLogger(csvlogger_fname)
earlystop_cllbk = tf.keras.callbacks.EarlyStopping(**cllbk_dict, patience=100, min_delta=1e-4, restore_best_weights=True)
mdlchckpt_fname = CHECKPOINT_FNAME(trgt_src=target_source)
mdlchckpt_cllbk = tf.keras.callbacks.ModelCheckpoint(**cllbk_dict, filepath=mdlchckpt_fname, save_best_only=True, save_weights_only=False)
callbacks = [csvlogger_cllbk, earlystop_cllbk, mdlchckpt_cllbk]
main_log.info(f"Callbacks: \n {callbacks}")



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#	MODEL TRAINING: TRAIN MODEL AND SAVE HISTORY IN A FILE
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# define filenames where history and best model must be saved
name = f"{target_source}_00"
HIST_PATH = os.path.join(RUN_DIR, f'{name}_training_history')
FILE_MODEL = os.path.join(RUN_DIR, f'{name}_last_model')

# fit the model on train and valid data
history = model.fit(
	distr_trn_dataset, 
	validation_data=distr_val_dataset, 
	steps_per_epoch=trn_steps_per_epoch, 
	validation_steps=val_steps_per_epoch, 
	epochs=epochs, 
	callbacks=callbacks)

# save history in order to plot losses
with open(HIST_PATH, 'wb') as file_pi:
	pickle.dump(history.history, file_pi)
main_log.info(f"Saved training history in {HIST_PATH} file")

# save the best model
model.save(filepath=FILE_MODEL, save_format='tf', include_optimizer=True)
main_log.info(f"Saved model in {FILE_MODEL}")

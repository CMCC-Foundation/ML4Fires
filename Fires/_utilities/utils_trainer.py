

import os
from typing import List
from datetime import datetime as dt

# Lightning imports
import lightning.pytorch.loggers as lp_logs
import lightning.pytorch.callbacks as lp_cllbks

# Pytorch Lightning module imports
import pytorch_lightning.loggers as pl_log

# Itwinai imports
from itwinai.loggers import MLFlowLogger as Itwinai_MLFLogger, LoggersCollection, Prov4MLLogger

# ML4Fires imports
from Fires._macros.macros import CHECKPOINTS_DIR, CONFIG, DISCORD_CFG, LOGS_DIR, RUN_DIR, PROVENANCE_DIR
from Fires._utilities.callbacks import DiscordBenchmark, FabricBenchmark, FabricCheckpoint
from Fires._utilities.logger import Logger as logger
from Fires._utilities.decorators import debug, export
from Fires._utilities.logger_itwinai import ItwinaiLightningLogger, ItwinaiMLFlowLogger, ProvenanceLogger


# define logger
_log = logger(log_dir=LOGS_DIR).get_logger("Trainer Utilities")


@export
@debug(log=_log)
def get_trainer_loggers() -> List:
	"""
	Returns a list of logger instances for use with the PyTorch Lightning Trainer.

	This function initializes and returns a list of logger objects to track 
	the training process. These loggers record relevant metrics and logs, 
	facilitating monitoring and debugging of the model's performance.

	Currently, the function includes a `CSVLogger`.
	The code for `MLFlowLogger` is provided but commented out.

	**Notes**
		To enable the `MLFlowLogger`, uncomment the relevant lines in the function and ensure 
		that the necessary environment variables (`MLFLOW_EXPERIMENT_NAME`, `MLFLOW_TRACKING_URI`) are set.

	**Returns**
		`_loggers` (List[Logger]):
			A list of logger instances to be used for monitoring the training process.

	**Example**
		```
		from Fires.utils import get_trainer_loggers
		from lightning.pytorch import Trainer

		# define loggers for Fabric trainer
		loggers = get_trainer_loggers()

		# define Pytroch Lightning Trainer and set loggers argument
		trainer = Trainer(loggers = loggers)
		```
	"""

	# define loggers for Fabric trainer
	_loggers = []
		
	# define pytorch_lightning.loggers.MLFlowLogger
	# _mlflow_logger = pl_log.MLFlowLogger(experiment_name=os.getenv('MLFLOW_EXPERIMENT_NAME'), run_name=os.getenv('MLFLOW_RUN_NAME'), tracking_uri=os.getenv('MLFLOW_TRACKING_URI'), log_model=True)
	# _loggers.append(_mlflow_logger)

	# define CSV logger
	_csv_logger = pl_log.CSVLogger(save_dir=RUN_DIR, name='csv_logs')
	_loggers.append(_csv_logger)	

	return _loggers


@export
@debug(log=_log)
def get_itwinai_loggers() -> LoggersCollection:
	"""
	Initializes and returns a `LoggersCollection` instance containing iTwinAI loggers.

	This function sets up a list of loggers specifically configured for iTwinAI, allowing for tracking
	of machine learning experiments and model provenance. The loggers currently included are:
	
	- `Itwinai_MLFLogger` for experiment tracking with MLFlow.
	- `Prov4MLLogger` for tracking the provenance of the experiments, ensuring reproducibility and traceability.

	**Returns**
		`_logger_collection` (LoggersCollection):
			A collection of configured loggers for use in tracking iTwinAI experiments.

	**Environment Variables**
	- `MLFLOW_EXPERIMENT_NAME` (str): The name of the MLFlow experiment.
	- `MLFLOW_TRACKING_URI` (str): The URI for the MLFlow tracking server.

	**Example**
		```
		from Fires.utils import get_itwinai_loggers
		from lightning.pytorch import Trainer

		# define itwinai loggers
		loggers = get_itwinai_loggers()

		# define Pytroch Lightning Trainer
		trainer = Trainer()

		# set itwinai_logger as new property of the trainer
		trainer.itwinai_logger = loggers

		```

	**Notes**
		- Ensure that `MLFLOW_EXPERIMENT_NAME` and `MLFLOW_TRACKING_URI` are defined in the environment 
		  to use the MLFlow logger effectively.
		- The provenance logger saves data every `n` logs, as defined by the `save_after_n_logs` parameter.

	"""
	# define empty list of loggers
	_loggers = []

	# define Itwinai MLFlow logger
	_itwinai_mlflow_logger = Itwinai_MLFLogger(experiment_name=os.getenv('MLFLOW_EXPERIMENT_NAME'), run_name=os.getenv('MLFLOW_RUN_NAME'), tracking_uri=os.getenv('MLFLOW_TRACKING_URI'), log_freq='epoch')
	_loggers.append(_itwinai_mlflow_logger)

	# define Itwinai Provenance logger
	_itwinai_provenance_logger = Prov4MLLogger(experiment_name=os.getenv('MLFLOW_EXPERIMENT_NAME'), provenance_save_dir=PROVENANCE_DIR, save_after_n_logs=1)
	_loggers.append(_itwinai_provenance_logger)

	# define loggers collection
	_logger_collection = LoggersCollection(_loggers)

	return _logger_collection


@export
@debug(log=_log)
def get_callbacks() -> List:
	"""
	Initializes and returns a list of callback instances for the Fabric trainer.

	This function sets up a collection of callbacks to manage different aspects of the training process,
	including early stopping, checkpointing, and benchmarking. Some callbacks, such as the `DiscordBenchmark`
	and `FabricBenchmark`, are provided but currently commented out.

	**Available Callbacks**
	- `EarlyStopping`: Stops training when validation loss stops improving.
	- `ModelCheckpoint`: Saves the model checkpoint with the best validation loss.
	
	**Optional Callbacks**
	- `DiscordBenchmark`: Sends benchmark data to a Discord webhook.
	- `FabricBenchmark`: Logs benchmark data to a specified CSV file.
	- `FabricCheckpoint`: Saves checkpoints to a specified directory.

	**Notes**
		To enable additional callbacks, uncomment their initialization lines.
		Make sure to configure environment variables or file paths as required.

	**Returns**
		`_callbacks` List[Callback]:
			A list of callback instances configured for use with the Fabric trainer.

	**Example**
		```
		from Fires.utils import get_callbacks
		from lightning.pytorch import Trainer

		# define callbacks for the trainer
		callbacks = get_callbacks()

		# define Pytroch Lightning Trainer and set callbacks argument
		trainer = Trainer(callbacks = callbacks)
		```
	"""

	# define empty list of callbacks for the trainer
	_callbacks = []

	# define Discord benchmark callback
	_discord_bench_cllbk = DiscordBenchmark(webhook_url=DISCORD_CFG.hooks.webhook_gen, benchmark_csv=os.path.join(RUN_DIR, "fabric_benchmark.csv"))
	# _callbacks.append(_discord_bench_cllbk)

	# define Fabric benchmark callback
	_fabric_bench_cllbk = FabricBenchmark(filename=os.path.join(RUN_DIR, "fabric_benchmark.csv"))
	# _callbacks.append(_fabric_bench_cllbk)

	# define Fabric checkpoint callback
	_fabric_check_cllbk = FabricCheckpoint(dst=CHECKPOINTS_DIR)
	# _callbacks.append(_fabric_check_cllbk)

	# define Early Stopping callback
	_earlystop_cllbk = lp_cllbks.EarlyStopping('val_loss', patience=10)
	_callbacks.append(_earlystop_cllbk)

	# define ModelCheckpoint callback, monitoring 'val_loss'
	_model_checkpoint_callback = lp_cllbks.ModelCheckpoint(dirpath=RUN_DIR, monitor="val_loss", save_top_k=1)
	_callbacks.append(_model_checkpoint_callback)

	return _callbacks

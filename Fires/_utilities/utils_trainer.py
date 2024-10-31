

import os
from typing import List

# Lightning imports
import lightning.pytorch.loggers as lp_logs
import lightning.pytorch.callbacks as lp_cllbks

# Pytorch Lightning module imports
import pytorch_lightning.loggers as pl_log

# Itwinai imports
from itwinai.loggers import MLFlowLogger as Itwinai_MLFLogger

# ML4Fires imports
from Fires._macros.macros import CHECKPOINTS_DIR, CONFIG, DISCORD_CFG, LOGS_DIR, RUN_DIR
from Fires._utilities.callbacks import DiscordBenchmark, FabricBenchmark, FabricCheckpoint
from Fires._utilities.logger import Logger as logger
from Fires._utilities.decorators import debug, export
from Fires._utilities.logger_itwinai import ItwinaiLightningLogger, ProvenanceLogger


# define logger
_log = logger(log_dir=LOGS_DIR).get_logger("Trainer Utilities")


@export
@debug(log=_log)
def get_loggers(run_name:str) -> List:

	# # get MLFlow logger
	# mlf_logger = MLFlowLogger(
	# 	experiment_name="ML4Fires_Juno",
	# 	tracking_uri=TRACKING_URI,
	# 	log_model=True,
	# )

	# define today's date
	today = eval(CONFIG.utils.datetime.today)
	_log.info(f" | Today: {today}")

	# define csv log name
	csv_fname = f'{today}_csv_logs'
	_log.info(f" | CSV Filename: {csv_fname}")

	# define loggers for Fabric trainer
	_loggers = []
	
	# define Itwinai Logger 
	_itwinai_logger = ItwinaiLightningLogger(savedir=os.path.join(LOGS_DIR, "ITWINAI"))
	_loggers.append(_itwinai_logger)

	# define Itwinai MLFlow logger
	_itwinai_mlflow_logger = Itwinai_MLFLogger(experiment_name=run_name, tracking_uri=os.getenv('MLFLOW_TRACKING_URI'), log_freq=10)
	# _loggers.append(_itwinai_mlflow_logger)
	
	# define pytorch_lightning.loggers.MLFlowLogger
	_mlflow_logger = pl_log.MLFlowLogger(experiment_name="ML4Fires_LOCAL", run_name=run_name, tracking_uri=os.getenv('MLFLOW_TRACKING_URI'), log_model=True)
	_loggers.append(_mlflow_logger)

	# define CSV logger
	_csv_logger = pl_log.CSVLogger(save_dir=RUN_DIR, name='csv_logs')
	_loggers.append(_csv_logger)

	# define Provenance logger
	# _provenance_logger = Prov4MLLogger(experiment_name=run_name, provenance_save_dir=os.path.join(LOGS_DIR, 'prov_logs'), save_after_n_logs=1)
	_provenance_logger = ProvenanceLogger(savedir=os.path.join(LOGS_DIR, "ITWINAI", "provenance"), experiment_name=run_name, save_after_n_logs=1)
	_loggers.append(_provenance_logger)

	return _loggers


@export
@debug(log=_log)
def get_callbacks() -> List:

	# define callbacks for Fabric trainer
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
	_earlystop_cllbk = lp_cllbks.EarlyStopping('val_loss')
	_callbacks.append(_earlystop_cllbk)

	# init ModelCheckpoint callback, monitoring 'val_loss'
	_model_checkpoint_callback = lp_cllbks.ModelCheckpoint(dirpath=RUN_DIR, monitor="val_loss", save_top_k=1)
	_callbacks.append(_model_checkpoint_callback)

	return _callbacks

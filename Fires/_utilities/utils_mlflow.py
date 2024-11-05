
import os

import mlflow

# ML4Fires imports
from Fires._macros.macros import CREDENTIALS_CFG, LOGS_DIR, CONFIG
from Fires._utilities.logger import Logger as logger
from Fires._utilities.decorators import debug, export

# define environment variables
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
os.environ['MLFLOW_TRACKING_USERNAME'] = CREDENTIALS_CFG.credentials.username
os.environ['MLFLOW_TRACKING_PASSWORD'] = CREDENTIALS_CFG.credentials.password
os.environ['MLFLOW_TRACKING_URI'] = CONFIG.mlflow.TRACKING_URI
os.environ['MLFLOW_EXPERIMENT_NAME'] = CONFIG.mlflow.EXPERIMENT_NAME

# define logger
_log = logger(log_dir=LOGS_DIR).get_logger("MLFLow Utilities")

@export
@debug(log=_log)
def setup_mlflow_experiment():
	"""
	Configures MLflow tracking URI and sets the experiment name.
	This function should be called once at the beginning of the script.
	"""
	mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
	mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME'))
	_log.info(f"MLflow Experiment set to '{os.getenv('MLFLOW_EXPERIMENT_NAME')}' with tracking URI '{os.getenv('MLFLOW_TRACKING_URI')}'")


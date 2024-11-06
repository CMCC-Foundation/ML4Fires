
import os

import mlflow
import torch

# ML4Fires imports
from Fires._macros.macros import CREDENTIALS_CFG, LOGS_DIR, CONFIG
from Fires._utilities.logger import Logger as logger
from Fires._utilities.decorators import debug, export
from Fires._utilities.utils_general import check_backend

# define environment variables
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = CONFIG.mlflow.TRACKING_INSECURE_TLS
os.environ['MLFLOW_TRACKING_USERNAME'] = CREDENTIALS_CFG.credentials.username
os.environ['MLFLOW_TRACKING_PASSWORD'] = CREDENTIALS_CFG.credentials.password
os.environ['MLFLOW_TRACKING_URI'] = CONFIG.mlflow.TRACKING_URI
os.environ['MLFLOW_EXPERIMENT_NAME'] = CONFIG.mlflow.EXPERIMENT_NAME

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

@export
@debug(log=_log)
def load_model_from_mlflow_registry(model_name, version=1, tag=None):
    # set tracking uri
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

    if version:
        # Load by specific version
        model_uri = f"models:/{model_name}/{version}"
        local_path = os.path.join(os.getcwd(), 'MLFLOW', f"{model_name}/{version}")
    elif tag:
        # Load by tag (if the tag is set in the UI)
        model_uri = f"models:/{model_name}/{tag}"
        local_path = os.path.join(os.getcwd(), 'MLFLOW', f"{model_name}/{tag}")
    else:
        raise ValueError("Either version or tag must be specified for model loading.")
        
    os.makedirs(local_path, exist_ok=True)    
    model = mlflow.pytorch.load_model(model_uri, map_location=torch.device(check_backend()), dst_path=local_path)
    return model

@export
@debug(log=_log)
def load_model_from_mlflow(run_id, scaler=True, provenance=False):
    # set tracking uri
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

    local_path = os.path.join(os.getcwd(), 'MLFLOW', f"{run_id}")
    os.makedirs(local_path, exist_ok=True)   

    client = mlflow.MlflowClient()
    if scaler:
        artifact_path = client.download_artifacts(run_id=run_id, path="scaler", dst_path=local_path)
    if provenance:
        artifact_path = client.download_artifacts(run_id=run_id, path=f"provgraph_{CONFIG.mlflow.EXPERIMENT_NAME}.dot", dst_path=local_path)
        artifact_path = client.download_artifacts(run_id=run_id, path=f"provgraph_{CONFIG.mlflow.EXPERIMENT_NAME}.json", dst_path=local_path)

    model_uri = f'runs:/{run_id}/last_model'
    model = mlflow.pytorch.load_model(model_uri, map_location=torch.device(check_backend()), dst_path=local_path)

    return model
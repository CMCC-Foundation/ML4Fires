import xarray as xr
import torch
import warnings
warnings.filterwarnings("ignore")

from Fires._utilities.utils_inference import do_inference_from_ds
from Fires._utilities.utils_mlflow import load_model_from_local_path
from Fires._utilities.utils_general import check_backend

def get_prediction_for_data(dataset_path: str,
                            model_path,
                           verbose=False):
    model = load_model_from_local_path(path=model_path).to(check_backend())
    if verbose:
        print(model)
    return do_inference_from_ds(dataset=xr.open_dataset(dataset_path),
                        model=model)
import xarray as xr
import torch
import warnings
warnings.filterwarnings("ignore")

from Fires._utilities.utils_inference import do_inference
from Fires._utilities.utils_mlflow import load_model_from_local_path


def get_prediction_for_data(dataset: xr.Dataset,
                            model_path,
                           verbose=False):
    model = load_model_from_local_path(path=model_path)
    if verbose:
        print(model)
    return do_inference(dataset=dataset,
                        model=model)
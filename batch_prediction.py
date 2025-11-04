import xarray as xr
import torch
import warnings
import os
warnings.filterwarnings("ignore")

from Fires._utilities.utils_inference import do_inference_from_ds, get_scaler_from_path
from Fires._utilities.utils_mlflow import load_model_from_local_path
from Fires._utilities.utils_general import check_backend

def get_prediction_for_data(dataset_path: str,
                            model_path,
                            verbose=False,
                            output_name="global_burned_areas"):
    
    model = load_model_from_local_path(path=model_path).to(check_backend())
    if verbose:
        print(model)
    scaler_path = os.path.join(model_path.split("last_model")[0],"scaler/scaler.dump")
    return do_inference_from_ds(dataset=xr.open_dataset(dataset_path),
                                model=model,
                                scaler=get_scaler_from_path(scaler_path),
                                var_name=output_name)


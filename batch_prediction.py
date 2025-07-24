import xarray as xr
import torch
import warnings
warnings.filterwarnings("ignore")

from Fires._utilities.utils_inference import do_inference

def get_prediction_for_data(dataset: xr.Dataset,
                            model_path):
    model = torch.load(model_path)
    return do_inference(dataset,
                        model)
import xarray as xr

import warnings
warnings.filterwarnings("ignore")

from Fires._utilities.utils_inference import do_inference

def get_prediction_for_data(dataset: xr.Dataset,
                            model):
    return do_inference(dataset,
                        model)
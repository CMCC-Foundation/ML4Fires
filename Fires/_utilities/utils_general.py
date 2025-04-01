
import torch
import importlib as implib

from Fires._macros.macros import LOGS_DIR, TORCH_CFG
from Fires._utilities.logger import Logger as logger
from Fires._utilities.decorators import debug, export

# define logger
_log = logger(log_dir=LOGS_DIR).get_logger("General Utilities")

@export
@debug(log=_log)
def check_backend() -> str:
	"""
	Determines the available backend engine for PyTorch computations.

	This function checks if the MPS (Metal Performance Shaders) or CUDA backends
	are available and sets the appropriate backend accordingly. If neither MPS 
	nor CUDA is available, it defaults to the CPU backend.

	Returns
	-------
	str
		The name of the backend to use for PyTorch computations ('mps', 'cuda', or 'cpu').
	"""

	if torch.backends.mps.is_available():
		backend:str = 'mps'
	elif torch.cuda.is_available():
		backend:str = 'cuda'
	else:
		backend:str = 'cpu'
	
	if backend in ['mps', 'cuda']:
		matmul_precision = TORCH_CFG.base.matmul_precision
		torch.set_float32_matmul_precision(matmul_precision)

	_log.info(f" | {backend.upper()} available")
	return backend

def process_call_string(input_string: str):
    split_string = input_string.split("::")
    return {
        "library_to_load": split_string[0].replace("/", "."),
        "function_to_call": split_string[1],
    }

def call_instance_of_function(library_to_load, function_to_call, **kwargs):

    library_instance = import_module_from_path(library_to_load)
    function_call = getattr(library_instance, function_to_call)
    kwargs = _process_kwargs(kwargs)
    return function_call(**kwargs)


def import_module_from_path(library_to_path):
    return implib.import_module(library_to_path.replace("/", "."))


def separate_kwargs(input: dict):
    kwargs = {}
    for key in input.keys():
        if key != "target":
            kwargs[key] = input[key]
    return input.target, kwargs


def _process_kwargs(kwargs):
    processed_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            if str(value).find("eval") == 0:
                processed_kwargs[key] = eval(value)
            else:
                processed_kwargs[key] = value
        else:
            processed_kwargs[key] = value

    return processed_kwargs


import torch
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


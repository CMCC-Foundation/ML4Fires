import os
import pickle
from .decorators import debug, export
from .macros import LOG_DIR
from .logger import Logger as logger
_cache_log = logger(log_dir=LOG_DIR).get_logger(log_name="cache")

@export
@debug(log=_cache_log)
def read_cache(cache_path: str):
    cached = None
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
    return cached

@export
@debug(log=_cache_log)
def write_cache(ds, cache_path: str):
    with open(cache_path, "wb") as f:
        pickle.dump(ds, f)
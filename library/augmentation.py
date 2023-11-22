import tensorflow as tf
from .decorators import debug, export
from .macros import LOG_DIR
from .logger import Logger as logger
_logger = logger(log_dir=LOG_DIR).get_logger("Augmentation")

@export
@debug(log=_logger)
def rot180(data):
    X,Y = data
    X = tf.image.rot90(X, k=2)
    Y = tf.image.rot90(Y, k=2)
    return (X,Y)

@export
@debug(log=_logger)
def left_right(data):
    X,Y = data
    X = tf.image.flip_left_right(X)
    Y = tf.image.flip_left_right(Y)
    return (X,Y)

@export
@debug(log=_logger)
def up_down(data):
    X,Y = data
    X = tf.image.flip_up_down(X)
    Y = tf.image.flip_up_down(Y)
    return (X,Y)
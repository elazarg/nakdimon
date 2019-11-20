# https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-554290095

import tensorflow as tf
import subprocess
import sys


def get_gpus_memory():
    """Get the max gpu memory.

    Returns
    -------
    usage: list
        Returns a list of total memory for each gpus.
    """
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.total",
        "--format=csv,nounits,noheader"
    ]).decode("utf-8")

    gpus_memory = [int(x) for x in result.strip().split("\n")]
    return gpus_memory


def setup_gpus(allow_growth=True, memory_fraction=.75):
    """Setup GPUs.

    Parameters:
    allow_growth (Boolean)
    memory_fraction (Float): Set maximum memory usage, with 1 using
        maximum memory
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for i, gpu in enumerate(gpus):
                memory = get_gpus_memory()[i]

                # tf.config.experimental.set_memory_growth(gpu, allow_growth)

                # Setting memory limit to max*fraction
                tf.config.experimental.set_virtual_device_configuration(gpu,
                            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory * memory_fraction)])

                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs", file=sys.stderr)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

from importlib.resources import files

MODELS_DIR = 'models'
MAIN_MODEL = files('nakdimon').joinpath('Nakdimon.h5')

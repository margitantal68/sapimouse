from enum import Enum

class NormalizationDirection( Enum ):
    ROWS = "rows"
    COLUMNS = "columns"
    ALL = "rows&columns together"
    USERS = "users-columns"
    NONE = "none"

class NormalizationType( Enum ):
    MINMAX ="minmax"
    ZSCORE ="zscore"
    NONE ="none"

class ModelType( Enum ):
    FCN = "fcn"
 
class DataType( Enum ):
    MOUSE = "mouse"
 

#  which representation learning is used
class RepresentationType(Enum):
    RAW ="rawdata"
    AE = "autoencoder"
    EE = "endtoend"


MODEL_TYPE = ModelType.FCN

OUTPUT_FIGURES = "output_png"
TRAINED_MODELS_PATH = "TRAINED_MODELS"
TRAINING_CURVES_PATH ="TRAINING_CURVES"

# Init random generator
RANDOM_STATE = 11235

# Model parameters
BATCH_SIZE = 16
EPOCHS = 100

# Temporary filename - used to save ROC curve data
TEMP_NAME = "scores.csv"

# CNN model Input shape MOUSE - Actions
# FEATURES = 47
# DIMENSIONS = 1

# CNN model Input shape MOUSE - SapiMouse
FEATURES = 128
DIMENSIONS = 2

# Scores & ROC settings
# Create score distribution plots for evaluations
SCORES = True

# Apply score normalization
SCORE_NORMALIZATION = True

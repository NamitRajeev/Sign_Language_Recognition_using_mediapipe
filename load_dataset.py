import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Define path to dataset
DATASET_PATH = "ISL_custom_data"

# Parameters
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

# Load the dataset (80% training, 20% validation)
train_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Normalize pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

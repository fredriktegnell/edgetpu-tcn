import tensorflow as tf
from tensorflow.keras.layers import Activation, SpatialDropout2D, Layer, Input, Dense, Flatten, Conv2D
from tensorflow.keras.models import Model
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

# Setting the float type for TensorFlow backend
tf.keras.backend.set_floatx('float32')

# Constants for the model
NUM_TIME_STEPS = 118  # Chosen based on the average length of reviews
NUM_FEATURES = 100  # Vector dimension from the Word2Vec model
BATCH_SIZE = 1  # Adjust based on GPU memory availability
EPOCHS = 2  # Number of complete passes through the training dataset during model training

# Model hyperparameters
NUM_FILTERS = 8
KERNEL_SIZE = 3
NUM_TCN_LAYERS = 3
DROPOUT_RATE = 0.1
DILATION_RATES = [2**i for i in range(NUM_TCN_LAYERS)]

# Specify the input and output file paths
W2V_MODEL = Word2Vec.load("word2vec/model/word2vec_model.bin")
TRAINING_PATH = "dataset/preprocessed_data/train_preprocessed.csv"
TRAINING_LABELS_PATH = "dataset/labels/train_labels.csv"
VALIDATION_PATH = "dataset/preprocessed_data/validation_preprocessed.csv"
VALIDATION_LABELS_PATH = "dataset/labels/validation_labels.csv"
OUTPUT_PATH = "quantized_model.tflite"

class TCN(Layer):
    """
    Temporal Convolutional Network (TCN) Layer.
    
    This layer applies a series of dilated convolutions to process sequences.
    """
    def __init__(self, num_filters, kernel_size, dilation_rate, dropout_rate):
        """
        Initialize the TCN layer.
        
        Parameters:
        - num_filters (int): Number of convolutional filters.
        - kernel_size (int): Size of the convolutional kernel.
        - dilation_rate (int): Dilation rate for the convolution.
        - dropout_rate (float): Dropout rate for spatial dropout.
        """
        super(TCN, self).__init__()
        self.conv2d = Conv2D(filters=num_filters, kernel_size=(kernel_size, 1), 
                             padding='same', dilation_rate=(dilation_rate, 1))
        self.activation = Activation('relu')
        self.dropout = SpatialDropout2D(rate=dropout_rate)

    def call(self, input):
        """
        Forward pass for the TCN layer.
        
        Parameters:
        - input (Tensor): Input tensor for the layer.
        
        Returns:
        - Tensor: Output tensor after applying convolutions and activations.
        """
        x = self.conv2d(input)
        x = self.activation(x)
        x = self.dropout(x)
        return x

def create_tcn_model(input_shape, num_filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, dilation_rates=DILATION_RATES):
    """
    Create the TCN model architecture.
    
    Parameters:
    - input_shape (tuple): Shape of the input tensor.
    
    Returns:
    - Model: A compiled Keras model with the TCN architecture.
    """

    input_layer = Input(batch_shape=(BATCH_SIZE,) + input_shape)
    x = input_layer

    # Add TCN layers
    for dilation_rate in dilation_rates:
        tcn_layer = TCN(num_filters, kernel_size, dilation_rate, dropout_rate)
        x = tcn_layer(x)

    # Flatten the output for the dense layer
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    return Model(inputs=input_layer, outputs=x)

def load_data(file_path, labels_path):
    """
    Load data and convert text reviews to numerical form.
    
    Parameters:
    - file_path (str): Path to the preprocessed data CSV file.
    - labels_path (str): Path to the labels CSV file.
    
    Returns:
    - tuple: A tuple containing the data and labels as numpy arrays.
    """
    reviews_df = pd.read_csv(file_path)
    labels_df = pd.read_csv(labels_path)
    reviews = [review.split() for review in reviews_df["review"]]

    x_data = np.zeros((len(reviews), NUM_TIME_STEPS, NUM_FEATURES, 1))
    for i, review in enumerate(reviews):
        for j, word in enumerate(review[:NUM_TIME_STEPS]):
            if word in W2V_MODEL.wv:
                x_data[i, j, :, 0] = W2V_MODEL.wv[word]

    y_data = labels_df["sentiment"].apply(lambda x: 1 if x == "positive" else 0).values

    return x_data, y_data

def representative_data_gen():
    """
    Generator function to provide representative data for quantization.
    
    Yields:
    - array: A batch of input data.
    """
    for input_value in x_train[:100]:
        yield [np.expand_dims(input_value, axis=0)]

# Create, compile, and train the TCN model
input_shape = (NUM_TIME_STEPS, NUM_FEATURES, 1)
tcn_model = create_tcn_model(input_shape)
tcn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
x_train, y_train = load_data(TRAINING_PATH, TRAINING_LABELS_PATH)
x_val, y_val = load_data(VALIDATION_PATH, VALIDATION_LABELS_PATH)
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_val = x_val.astype('float32')
y_val = y_val.astype('float32')
tcn_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))

# Quantize the model for Edge TPU compatibility
converter = tf.lite.TFLiteConverter.from_keras_model(tcn_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

# Save the quantized model
with open(OUTPUT_PATH, 'wb') as f:
    f.write(tflite_quant_model)

import os
import pathlib
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from pycoral.utils import edgetpu

# Define paths for the TensorFlow model, CSV file, and labels
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
MODEL_FILE = os.path.join(SCRIPT_DIR, 'quantized_model_edgetpu.tflite')
CSV_FILE = os.path.join(SCRIPT_DIR, 'dataset/preprocessed_data/test_preprocessed.csv')
LABEL_FILE = os.path.join(SCRIPT_DIR, 'dataset/labels/test_labels.csv')

# Load the Word2Vec model for converting text reviews to numerical representations
w2v_model = Word2Vec.load("word2vec/model/word2vec_model.bin")

# Initialize the TensorFlow Lite interpreter for Edge TPU
interpreter = edgetpu.make_interpreter(MODEL_FILE)
interpreter.allocate_tensors()

# Load the testing data and actual labels
data = pd.read_csv(CSV_FILE)
reviews = data['review'].values
actual_labels = pd.read_csv(LABEL_FILE)['sentiment'].values

# Convert a review string to its numerical representation using the Word2Vec model
def review_to_numerical(review, w2v_model, max_length=118, num_features=100):
    """
    Convert a text review to its numerical representation using Word2Vec.
    
    Args:
    - review (str): The text review.
    - w2v_model (Word2Vec): The trained Word2Vec model.
    - max_length (int): The maximum length of the review.
    - num_features (int): The number of features for each word.
    
    Returns:
    - numpy.array: The numerical representation of the review.
    """
    numerical_review = np.zeros((max_length, num_features))
    words = review.split()
    for i, word in enumerate(words[:max_length]):
        if word in w2v_model.wv:
            numerical_review[i] = w2v_model.wv[word]
    return numerical_review

# Initialize counters for accuracy calculation
correct_predictions = 0
total_predictions = 0

# Run inference on each review and compare with actual labels
for review, actual_label in zip(reviews, actual_labels):
    review_input = review_to_numerical(review, w2v_model).reshape(118, 100, 1)
    
    # Quantize the input data for Edge TPU
    input_details = interpreter.get_input_details()
    scale, zero_point = input_details[0]['quantization']
    review_input = np.round(review_input / scale + zero_point).astype(np.int8)
    review_input = review_input[np.newaxis, ...]
    
    # Set the input tensor, invoke the interpreter, and get the output tensor
    interpreter.set_tensor(input_details[0]['index'], review_input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Determine the predicted sentiment based on the output
    predicted_sentiment = "positive" if output[0] > 0.5 else "negative"
    
    # Update counters based on prediction accuracy
    if predicted_sentiment == actual_label:
        correct_predictions += 1
    total_predictions += 1

# Calculate and print the accuracy
accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy: {accuracy}%")

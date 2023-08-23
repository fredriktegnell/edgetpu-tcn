import pandas as pd
from gensim.models import Word2Vec

# Configuration for Word2Vec training

# Dimensionality of the word vectors.
# A larger dimension captures more information but requires more memory and resources.
# Common values range from 100 to 300.
VECTOR_DIM = 100 

# Maximum distance between the current and predicted word within a sentence.
# A larger window captures more context but may also introduce noise.
# Common values range from 5 to 10.
MAX_DIST = 5 

# Minimum frequency of a word to be included in the vocabulary.
# Words that appear less frequently than this threshold are ignored.
# Common values range from 5 to 10.
MIN_FREQ = 5 

# Number of threads used for training.
# This should be set based on the available CPU cores for parallel processing.
# In this case, the machine has 8 cores.
THREAD_COUNT = 8 

# Load the pre-processed training data
# The data is expected to be in a CSV format with a 'review' column containing the preprocessed text.
data = pd.read_csv('dataset/preprocessed_data/train_preprocessed.csv')

# Convert the 'review' column into a list of reviews
reviews = data['review'].tolist()

# Tokenize each review into a list of words to prepare data for Word2Vec training
sentences = [review.split() for review in reviews]

# Train the Word2Vec model
# The model learns to generate word embeddings based on the provided sentences.
model = Word2Vec(sentences, vector_size=VECTOR_DIM, window=MAX_DIST, min_count=MIN_FREQ, workers=THREAD_COUNT)

# Save the trained model for future use
# The model is saved in a binary format and can be loaded later for generating word embeddings or further training.
model.save('word2vec/model/word2vec_model.bin')

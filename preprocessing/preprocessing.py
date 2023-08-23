import pandas as pd
import string
import re
import nltk
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# Specify the input and output file paths for training, validation, and testing datasets
DATASETS = {
    "train": ("dataset/train.csv", "dataset/preprocessed_data/train_preprocessed.csv"),
    "validation": ("dataset/validation.csv", "dataset/preprocessed_data/validation_preprocessed.csv"),
    "test": ("dataset/test.csv", "dataset/preprocessed_data/test_preprocessed.csv")
}

# Download the stopwords dataset from nltk
nltk.download('stopwords')

# Initialize the spaCy English model
# 'parser' and 'ner' are disabled as they are not needed for lemmatization
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def review_to_words(review):
    """
    Convert a raw review to a cleaned string of words.
    
    Parameters:
    - review (str): the original review string.
    
    Returns:
    - str: a preprocessed string.
    """
    
    # 1. Remove HTML tags using BeautifulSoup
    review_text = BeautifulSoup(review, features="html.parser").get_text()
    
    # 2. Remove non-letters using regex
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    
    # 3. Convert to lowercase and split into individual words
    words = review_text.lower().split()
    
    # 4. Remove stopwords
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    
    # 5. Lemmatization using spaCy
    doc = nlp(" ".join(words))
    words = [token.lemma_ for token in doc]
    
    # 6. Join the words back into a single string and return
    return " ".join(words)

def preprocess_dataset(input_file, output_file):
    """
    Load a dataset, preprocess its reviews, and save the preprocessed data to a new CSV file.
    
    Parameters:
    - input_file (str): Path to the original dataset.
    - output_file (str): Path to save the preprocessed dataset.
    """
    # Load the dataset
    data = pd.read_csv(input_file)

    # Apply the preprocessing function to each review in the dataset
    data['review'] = data['review'].apply(review_to_words)

    # Save the preprocessed data to a new CSV file
    data.to_csv(output_file, index=False)

# Process each dataset
for dataset_type, (input_path, output_path) in DATASETS.items():
    preprocess_dataset(input_path, output_path)

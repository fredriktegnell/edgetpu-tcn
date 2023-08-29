# TCN compatible with Edge TPU

## Description
This project implements a Temporal Convolutional Network (TCN) compatible with the Edge TPU USB Accelerator. The primary motivation behind this project is to leverage the power of Edge TPU for real-time sentiment analysis tasks. The project utilizes TensorFlow for model training and quantization, and Gensim's Word2Vec for word embeddings.

## System Capabilities
- Data Preprocessing: The system can preprocess raw movie reviews by removing unnecessary characters, converting text to lowercase, and performing lemmatization.
- Word Embeddings: Using the Word2Vec model, the system can convert words in the reviews into numerical vectors, capturing the semantic meaning of each word.
- Model Training: The system can train a TCN model on the preprocessed reviews, learning to predict the sentiment of each review.
- Model Quantization: After training, the system quantizes the TCN model, making it compatible with the Edge TPU.
- Sentiment Analysis: With the trained and quantized model, the system can predict the sentiment of new movie reviews, classifying them as either positive or negative.

## Performance Results
Below are some benchmark results obtained on the following system:

**CPU**: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz

| Metric                          | Edge TPU | CPU |
|---------------------------------|----------|------|
| Average Inference Time (single review)  | 35.40ms  | 0.46ms|
| Accuracy (quantized_model) | 80.3% | 80.3% |

(Note: The above numbers are based on benchmark tests conducted in my environment and may vary based on different setups.)

### Reducing Inference Time on Edge TPU
To achieve faster inference times on the Edge TPU, consider the following optimizations:

1. **USB Connection**: The benchmark results were obtained using a USB 2.1 connection. Upgrading to a USB 3.0 connection, if available, can significantly reduce data transfer latency between the host device and the Edge TPU, thereby improving overall inference speed.

2. **Model Complexity**: Simplifying the model by reducing the number of layers or parameters can lead to faster inference times. However, it's essential to balance this with potential impacts on model accuracy.

3. **Quantization-Aware Training**: While attempts to use quantization-aware training were made, it was not successfully implemented for this project. However, for future endeavors, it's worth noting that quantization-aware training can potentially produce models that are optimized for the Edge TPU, leading to faster and more accurate inferences compared to models quantized post-training.

By considering these optimizations and potential improvements, there's an opportunity to further enhance the performance of models on the Edge TPU.

## Table of Contents
- [TCN compatible with Edge TPU](#tcn-compatible-with-edge-tpu)
  - [Description](#description)
  - [System Capabilities](#system-capabilities)
  - [Performance Results](#performance-results)
    - [Reducing Inference Time on Edge TPU](#reducing-inference-time-on-edge-tpu)
  - [Table of Contents](#table-of-contents)
  - [Installation \& Setup](#installation--setup)
    - [Prerequisites](#prerequisites)
      - [Libraries and Frameworks](#libraries-and-frameworks)
      - [Hardware](#hardware)
    - [Setup](#setup)
  - [Usage](#usage)
    - [Dataset Split](#dataset-split)
    - [Data Preprocessing](#data-preprocessing)
    - [Word2Vec Training](#word2vec-training)
    - [Word2Vec Model Evaluation](#word2vec-model-evaluation)
    - [TCN Model Training \& Quantization](#tcn-model-training--quantization)
    - [Sentiment Analysis](#sentiment-analysis)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Quantization and Edge TPU](#quantization-and-edge-tpu)

## Installation & Setup
### Prerequisites

#### Libraries and Frameworks
| Library/Framework  | Version      |
|--------------------|--------------|
| TensorFlow        | 2.13.0       |
| CUDA              | 11.8.0       |
| cuDNN             | 8.6.0.163    |
| Gensim            | 4.3.0        |
| Pandas            | 1.5.2        |
| NumPy             | 1.25.2       |
| PyCoral           | 2.0.0        |
| Edge TPU Runtime  | 16.0         |
| Edge TPU Compiler | 16.0         |
| Scikit-learn      | 1.3.0        |
| NLTK              | 3.8.1        |
| Spacy             | 3.5.3        |
| BeautifulSoup     | 4.12.2       |

#### Hardware
- Edge TPU USB Accelerator

| .py File                   | Libraries & Hardware Used                                                                                     |
|----------------------------|----------------------------------------------------------------------------------------------------|
| `dataset_split.py`         | Pandas, Scikit-learn                                                                               |
| `preprocessing.py`         | Pandas, NLTK, Spacy, BeautifulSoup                                                                  |
| `word2vec_training.py`     | Pandas, Gensim                                                                                     |
| `word2vec_model_evaluator.py` | Gensim                                                                                       |
| `tcn.py`                   | TensorFlow, CUDA, cuDNN, Gensim, Pandas, NumPy
| `sentiment_analysis.py`    | Tensorflow, Pandas, NumPy, Gensim, PyCoral, Edge TPU Runtime, Edge TPU Compiler, Hardware: Edge TPU USB Accelerator

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/fredriktegnell/edgetpu-tcn
   cd edgetpu-tcn
   ```

## Usage
All of the following commands are intended to be run from the root directory.
### Dataset Split
Splits a dataset into training, validation and testing parts which then are saved to new CSV files:
  ```bash
  python3 preprocessing/dataset_split.py
  ```
Expected Output: Three new CSV files — train.csv, validation.csv, and test.csv — will be generated, each containing a subset of the original dataset.
### Data Preprocessing
Processes raw review data by performing several preprocessing steps such as removing HTML tags, non-letter characters, converting to lowercase, removing stopwords, and lemmatization. The preprocessed reviews are then saved to new CSV files for training, validation, and testing:
  ```bash
  python3 preprocessing/preprocessing.py
  ```
Expected Output: Preprocessed reviews saved to new CSV files (train_preprocessed.csv, validation_preprocessed.csv, and test_preprocessed.csv). Each file will contain reviews that have been cleaned, tokenized, and lemmatized.
### Word2Vec Training
Trains a Word2Vec model using preprocessed reviews from the training dataset. The trained model generates word embeddings and is saved for future use:
  ```bash
  python3 word2vec/word2vec_training.py
  ```
Expected Output: A trained Word2Vec model saved as word2vec_model.bin in the word2vec/model/ directory.
### Word2Vec Model Evaluation
Loads the trained Word2Vec model and evaluates its performance by:
- Calculating the semantic similarity between pairs of words.
- Performing word analogy tasks.
- Conducting word vector arithmetic.
- Checking for words in the model's vocabulary and identifying out-of-vocabulary words.
The evaluation results are saved to an output text file:
  ```bash
  python3 word2vec/word2vec_model_evaluator.py
  ```
Expected Output: An output text file named output.txt in the word2vec/ directory containing evaluation results such as word similarities, analogy results, vector arithmetic, and vocabulary checks.
### TCN Model Training & Quantization
Trains a Temporal Convolutional Network (TCN) model on preprocessed reviews from the training dataset. The model is then quantized for compatibility with Edge TPU and saved for future use:
  ```bash
  python3 tcn.py
  ```
This script performs the following tasks:
- Sets up the TCN model architecture with specified hyperparameters.
- Loads the preprocessed training and validation datasets.
- Converts text reviews to numerical form using a pre-trained Word2Vec model.
- Trains the TCN model on the training dataset.
- Quantizes the trained model for compatibility with Edge TPU.
- Saves the quantized model in TensorFlow Lite format.

Expected Output:
- A trained TCN model will be saved in the TensorFlow format.
- A quantized version of the TCN model will be saved as quantized_model.tflite for compatibility with Edge TPU.
### Sentiment Analysis

Evaluates the sentiment of reviews using a quantized TCN model optimized for either the Edge TPU or the CPU. The device for inference can be specified using the `device` argument:

  ```bash
  python3 sentiment_analysis.py [device]
  ```
Replace [device] with either edgetpu or cpu depending on your preference.

This script performs the following tasks:
- Parses the device argument to determine the inference device.
- Loads the appropriate quantized TCN model based on the chosen device.
- Initializes the TensorFlow Lite interpreter for the chosen device.
- Loads the testing dataset and its corresponding labels.
- Converts each review in the testing dataset to its numerical representation using the Word2Vec model.
- Runs inference on each review using the quantized TCN model.
- Calculates and prints the average inference time for the reviews.
- Compares the predicted sentiment with the actual labels to calculate the accuracy of the model.
- Prints the overall accuracy of the sentiment analysis.

Expected Output:
- The script will process each review in the test dataset using the quantized TCN model.
- The average inference time for processing the reviews will be printed.
- At the end, the script will print the overall accuracy of the sentiment analysis, indicating the percentage of reviews correctly classified.

## Dataset

The data used in this project comes from the [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) provided by Stanford University. This dataset is designed for binary sentiment classification and contains 50,000 highly polar movie reviews for training, validation and testing.

**Citation**: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

## Model Architecture
The TCN model is designed to work efficiently with the Edge TPU USB Accelerator. It leverages temporal convolutions to analyze sequences of data, making it ideal for sentiment analysis tasks.

## Quantization and Edge TPU
The model is quantized to ensure compatibility with the Edge TPU. Quantization reduces the model size and speeds up inference times, making real-time sentiment analysis feasible.
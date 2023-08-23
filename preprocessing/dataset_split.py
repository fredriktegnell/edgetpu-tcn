import pandas as pd
from sklearn.model_selection import train_test_split

# Specify the input and output file paths
DATASET_PATH = "dataset/IMDB Dataset.csv"
TRAINING_PATH = "dataset/train.csv"
TRAINING_LABELS_PATH = "dataset/labels/train_labels.csv"
VALIDATION_PATH = "dataset/validation.csv"
VALIDATION_LABELS_PATH = "dataset/labels/validation_labels.csv"
TESTING_PATH = "dataset/test.csv"
TESTING_LABELS_PATH = "dataset/labels/test_labels.csv"

# Load the dataset from a CSV file
dataset = pd.read_csv(DATASET_PATH)

# Splitting the dataset into features (x) and labels (y)
x = dataset["review"]
y = dataset["sentiment"]

# Splitting the dataset into training, validation, and test sets with split ratio 80, 10, 10
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Making sure the sizes of the resulting sets are correct
print("Training set size:", len(x_train))
print("Validation set size:", len(x_val))
print("Test set size:", len(x_test))

# Saving the divided datasets to separate CSV files
x_train.to_csv(TRAINING_PATH, index=False)
y_train.to_csv(TRAINING_LABELS_PATH, index=False)
x_val.to_csv(VALIDATION_PATH, index=False)
y_val.to_csv(VALIDATION_LABELS_PATH, index=False)
x_test.to_csv(TESTING_PATH, index=False)
y_test.to_csv(TESTING_LABELS_PATH, index=False)
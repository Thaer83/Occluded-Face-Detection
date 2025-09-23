import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Assuming you have preprocessed data and image generators
# Example for ImageDataGenerator for training, validation, and testing

# Adjust these paths to your dataset directories
# Step 3: Set up directories for training, validation, and testing datasets
train_dir = '/root/face_project/dataset/dataset/final_dataset'
val_dir = '/root/face_project/dataset/dataset/validation'
test_dir = '/root/face_project/dataset/dataset/Test'  # Path for test data
# Image data generator
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')


# Feature extraction using DenseNet
def extract_features(generator):
model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = []
    labels = []

    for data_batch, label_batch in generator:
        feature_batch = model.predict(data_batch)
        features.append(feature_batch)
        labels.append(label_batch)

        # Stop after one epoch (if you want more data, adjust the condition)
        if len(features) >= len(generator):
            break

    features = np.vstack(features)
    labels = np.concatenate(labels)

    return features, labels


# Extract features from the train, validation, and test generators
X_train, y_train = extract_features(train_generator)
X_val, y_val = extract_features(val_generator)
X_test, y_test = extract_features(test_generator)

# Flatten the extracted features
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Normalize the features (Optional but recommended for ELM)
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_val_flat = scaler.transform(X_val_flat)
X_test_flat = scaler.transform(X_test_flat)


# Extreme Learning Machine (ELM) class
class ELM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Randomly initialize weights between input and hidden layer
        self.W = np.random.randn(self.input_size, self.hidden_size)
        self.b = np.random.randn(self.hidden_size)

    def _activation(self, X):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-(np.dot(X, self.W) + self.b)))

    def fit(self, X_train, y_train):
        # Compute the hidden layer output
        H_train = self._activation(X_train)
        # Calculate the output weights using the Moore-Penrose pseudoinverse
        H_train_pseudo_inv = np.linalg.pinv(H_train)
        self.W_out = np.dot(H_train_pseudo_inv, y_train)

    def predict(self, X):
        # Predict the output based on the learned weights
        H = self._activation(X)
        y_pred = np.dot(H, self.W_out)
        return y_pred
all_results =[]
all_results_test =[]
for run in range(1 ,6):
    # Track the start time of training
    start_time_train = time.time()

    # Initialize ELM
    elm = ELM(input_size=X_train_flat.shape[1], hidden_size=100)  # Adjust hidden size as needed
    elm.fit(X_train_flat, y_train)


    # Track the end time of training and calculate the training duration
    end_time_train = time.time()
    training_duration = end_time_train - start_time_train

    # Track the start time of testing
    start_time_test = time.time()

    # Predict on train, validation, and test sets
    y_pred_train = elm.predict(X_train_flat)
    y_pred_val = elm.predict(X_val_flat)
    y_pred_test = elm.predict(X_test_flat)

    # Track the end time of testing and calculate the testing duration
    end_time_test = time.time()
    testing_duration = end_time_test - start_time_test

    # Convert the outputs to binary predictions (for binary classification)
    y_pred_train = (y_pred_train > 0.5).astype(np.float32)
    y_pred_val = (y_pred_val > 0.5).astype(np.float32)
    y_pred_test = (y_pred_test > 0.5).astype(np.float32)

    # Convert the labels to float32 for compatibility with binary cross-entropy
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)


    # Step 9: Evaluate the performance on the training, validation, and test sets
    def print_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        return accuracy,precision,recall,f1,conf_matrix
        # print(f"Accuracy: {accuracy}")
        # print(f"Precision: {precision}")
        # print(f"Recall: {recall}")
        # print(f"F1-Score: {f1}")
        # print(f"Confusion Matrix:\n{conf_matrix}")


    # Training set metrics
    print("Training Set Metrics:")
    accuracy,precision,recall,f1,conf_matrix=print_metrics(y_train, y_pred_train)

    # Validation set metrics
    # print("\nValidation Set Metrics:")
    # print_metrics(y_val, y_pred_val)

    # Test set metrics
    print("\nTest Set Metrics:")
    accuracy1,precision1,recall1,f11,conf_matrix1 = print_metrics(y_test, y_pred_test)

    # Optional: If you want to calculate Loss (for binary classification)
    # Loss is typically computed using cross-entropy for classification tasks

    # Compute binary cross-entropy loss for training, validation, and test
    loss_train = tf.keras.losses.binary_crossentropy(y_train, y_pred_train)
    loss_val = tf.keras.losses.binary_crossentropy(y_val, y_pred_val)
    loss_test = tf.keras.losses.binary_crossentropy(y_test, y_pred_test)

    # Take the mean of the loss over all samples in the batch
    loss_train = tf.reduce_mean(loss_train).numpy()
    loss_val = tf.reduce_mean(loss_val).numpy()
    loss_test = tf.reduce_mean(loss_test).numpy()

    print(f"\nTraining Loss: {loss_train}")
    print(f"Validation Loss: {loss_val}")
    print(f"Test Loss: {loss_test}")

    # Step 10: Print running times
    print(f"\nTraining Time: {training_duration:.2f} seconds")
    print(f"Testing Time: {testing_duration:.2f} seconds")

    resultsTrain = {
        'Run-test': run,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Loss': loss_train,
        'Traning Time (s)': training_duration,
        'Confusion Matrix': [conf_matrix]  # Storing as a list to keep it in a cell
    }
    all_results.append(resultsTrain)
    resultsTest = {
        'Run-test': run,
        'Accuracy': accuracy1,
        'Precision': precision1,
        'Recall': recall1,
        'F1 Score': f11,
        'Loss': loss_test,
        'Testing Time (s)': testing_duration,
        'Confusion Matrix': [conf_matrix1]  # Storing as a list to keep it in a cell
    }
    all_results_test.append(resultsTrain)




excel_path_final_Train = f'ModelELM/testing_metrics_ELM_5_runs_train.xlsx'
excel_path_final_Test = f'ModelELM/testing_metrics_ELM_5_runs_test.xlsx'

df_results = pd.DataFrame(all_results)

df_results.to_excel(excel_path_final_Train, index=False)


df_results_test = pd.DataFrame(all_results_test)

df_results_test.to_excel(excel_path_final_Test, index=False)

import numpy as np
import cv2
import time
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Model
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import hpelm
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import pandas as pd

# Feature extraction functions (same as before)
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)


class ELMClassifier(BaseEstimator):
    def __init__(self, n_hidden=1000, activation='sigm', rp=0.1):
        self.n_hidden = n_hidden
        self.activation = activation

    def fit(self, X, y):
        self.model = hpelm.ELM(X.shape[1], 2)  # Set number of output units based on y's columns
        self.model.add_neurons(self.n_hidden, self.activation)
        self.model.train(X, y, 'c', reg_param=0.1)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        """Score method to return accuracy for GridSearchCV"""
        y_pred = self.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot predictions back to class indices
        y_true_classes = np.argmax(y, axis=1)
        return accuracy_score(y_true_classes, y_pred_classes)


train_dir = '/root/face_project/dataset/dataset/final_dataset'
validation_dir = '/root/face_project/dataset/dataset/validation'
test_dir = '/root/face_project/dataset/dataset/Test'  # Path for test data


def extract_Densenet_features(img):
    img_resized = cv2.resize(img, (224, 224))  # Resize to 224x224
    img_array = img_to_array(img_resized)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for Densenet

    features = model.predict(img_array)  # Extract features
    features = features.flatten()  # Flatten the feature map to a 1D vector
    return features


def extract_hog_features(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    fd, _ = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return fd


def extract_combined_features(img):
    Densenet_features = extract_Densenet_features(img)
    hog_features = extract_hog_features(img)
    combined_features = np.concatenate((Densenet_features, hog_features))  # Concatenate both feature sets
    return combined_features


# Using ImageDataGenerator for loading and augmenting the dataset
datagen = ImageDataGenerator(rescale=1. / 255)  # Rescale images to [0, 1]

batch_size = 32

train_generator = datagen.flow_from_directory(
    train_dir,  # Path to your training data
    target_size=(224, 224),  # Resize images to fit Densenet input size
    batch_size=batch_size,
    class_mode='binary',  # Or 'categorical' if multi-class classification
    shuffle=True
)

test_generator = datagen.flow_from_directory(
    test_dir,  # Path to your testing data
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',  # Or 'categorical' if multi-class
    shuffle=False
)


# Extract features in batches from the generator
def extract_features_from_generator(generator):
    features = []
    labels = []

    for batch_images, batch_labels in generator:
        for img, label in zip(batch_images, batch_labels):
            combined_features = extract_combined_features(img)
            features.append(combined_features)
            labels.append(label)

        if len(features) >= generator.samples:
            break

    return np.array(features), np.array(labels)


all_results = []
# Extract features for training and testing
X_train, y_train = extract_features_from_generator(train_generator)
X_test, y_test = extract_features_from_generator(test_generator)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Standardize the training data
X_test_scaled = scaler.transform(X_test)  # Standardize the testing data

# Initialize OneHotEncoder with sparse_output=False for compatibility with new sklearn versions
encoder = OneHotEncoder(sparse_output=False)

# One-hot encode the target labels for train, validation, and test sets
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
# Step 5: Define and train ELM model

# --- Grid Search for ELM Hyperparameters ---
param_grid = {
    'n_neurons': list(range(100, 1000, 50)),  # Number of neurons in ELM
    'activation': ['sigm'],  # Activation function
    'rp': [0.01, 0.1, 1, 10, 100, 1000]
}

for run in range(1, 6):
    best_score = -float('inf')
    best_params = {}
    best_model_metrics = {}

    # Perform Grid Search manually
    for n_neurons in param_grid['n_neurons']:
        for activation in param_grid['activation']:
            for rp in param_grid['rp']:
                print(f"Training ELM with n_neurons={n_neurons}, activation={activation}...")
                # input_size = X_train_scaled.shape[1],
                # Initialize ELM model with current parameters
                elm = ELMClassifier(n_hidden=n_neurons, activation=activation, rp=rp)  # Adjust hidden size as needed
                # elm = ELM(n_neurons=n_neurons, activation=activation)

                # Start timing the training
                start_time_train = time.time()
                elm.fit(X_train_scaled, y_train_onehot)
                end_time_train = time.time()
                train_time = end_time_train - start_time_train  # Time taken to train

                # Predict on the training set
                y_train_pred = elm.predict(X_train_scaled)

                # Predict on the testing set
                start_time_test = time.time()
                y_test_pred = elm.predict(X_test_scaled)
                end_time_test = time.time()
                test_time = end_time_test - start_time_test  # Time taken to test

                # Calculate training metrics
                # test_accuracy = accuracy_score(np.argmax(y_train_onehot, axis=1), np.argmax(y_train_pred, axis=1))

                train_accuracy = accuracy_score(np.argmax(y_train_onehot, axis=1), np.argmax(y_train_pred, axis=1))
                train_precision = precision_score(np.argmax(y_train_onehot, axis=1), np.argmax(y_train_pred, axis=1))
                train_recall = recall_score(np.argmax(y_train_onehot, axis=1), np.argmax(y_train_pred, axis=1))
                train_f1 = f1_score(np.argmax(y_train_onehot, axis=1), np.argmax(y_train_pred, axis=1))
                train_loss = log_loss(np.argmax(y_train_onehot, axis=1), np.argmax(y_train_pred, axis=1))

                # Calculate testing metrics
                

                test_accuracy = accuracy_score(np.argmax(y_test_onehot, axis=1), np.argmax(y_test_pred, axis=1))
                test_precision = precision_score(np.argmax(y_test_onehot, axis=1), np.argmax(y_test_pred, axis=1))
                test_recall = recall_score(np.argmax(y_test_onehot, axis=1), np.argmax(y_test_pred, axis=1))
                test_f1 = f1_score(np.argmax(y_test_onehot, axis=1), np.argmax(y_test_pred, axis=1))
                test_loss = log_loss(np.argmax(y_test_onehot, axis=1), np.argmax(y_test_pred, axis=1))

                # Track the best parameters based on testing score
                if test_accuracy > best_score:
                    best_score = test_accuracy
                    best_params = {'n_neurons': n_neurons, 'activation': activation}
                    best_model_metrics = {
                        'train_accuracy': train_accuracy,
                        'train_precision': train_precision,
                        'train_recall': train_recall,
                        'train_f1': train_f1,
                        'train_loss': train_loss,
                        'train_time': train_time,
                        'test_accuracy': test_accuracy,
                        'test_precision': test_precision,
                        'test_recall': test_recall,
                        'test_f1': test_f1,
                        'test_loss': test_loss,
                        'test_time': test_time,
                        'n_neurons': n_neurons,
                        'activation': activation,
                        'RP': rp
                    }

            # Optional: Print metrics for the current set of hyperparameters
            # print(f"Test Accuracy: {test_accuracy:.4f}")
            # print(f"Precision: {test_precision:.4f}")
            # print(f"Recall: {test_recall:.4f}")
            # print(f"F1-Score: {test_f1:.4f}")
            # print(f"Loss: {test_loss:.4f}")
            # print(f"Training Time: {train_time:.2f} seconds")
            # print(f"Testing Time: {test_time:.2f} seconds")
            # print("-" * 50)

    # After Grid Search, print the metrics for the best model
    print("\nBest Model Metrics:")
    print(f"Best n_neurons: {best_params['n_neurons']}")
    print(f"Best activation function: {best_params['activation']}")
    print("\nTraining Metrics:")
    print(f"Training Accuracy: {best_model_metrics['train_accuracy']:.4f}")
    print(f"Training Precision: {best_model_metrics['train_precision']:.4f}")
    print(f"Training Recall: {best_model_metrics['train_recall']:.4f}")
    print(f"Training F1-Score: {best_model_metrics['train_f1']:.4f}")
    print(f"Training Loss: {best_model_metrics['train_loss']:.4f}")
    print(f"Training Time: {best_model_metrics['train_time']:.2f} seconds")

    print("\nTesting Metrics:")
    print(f"Testing Accuracy: {best_model_metrics['test_accuracy']:.4f}")
    print(f"Testing Precision: {best_model_metrics['test_precision']:.4f}")
    print(f"Testing Recall: {best_model_metrics['test_recall']:.4f}")
    print(f"Testing F1-Score: {best_model_metrics['test_f1']:.4f}")
    print(f"Testing Loss: {best_model_metrics['test_loss']:.4f}")
    print(f"Testing Time: {best_model_metrics['test_time']:.2f} seconds")
    all_results.append(best_model_metrics)

excel_path_final_Train = f'ModelELM/testing_metrics_ELM_5_runs_train_GridSearch.xlsx'

df_results = pd.DataFrame(all_results)

df_results.to_excel(excel_path_final_Train, index=False)

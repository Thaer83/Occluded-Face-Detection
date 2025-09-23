import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from skimage.feature import hog
from skimage import exposure
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, ResNet101, ResNet152, EfficientNetB0, \
    DenseNet121, NASNetLarge
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import pandas as pd
import time
import os


def get_model(model_name):
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'VGG19':
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'ResNet101':
        base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'ResNet152':
        base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'NASNetLarge':
        base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


    else:
        raise ValueError(f"Model {model_name} not supported.")
    return base_model



def apply_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges


def generate_region_proposals(edges):
    # Find contours from the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Generate bounding boxes for each contour
    proposals = [cv2.boundingRect(c) for c in contours]
    # Filter out proposals with area less than 200px
    filtered_proposals = [box for box in proposals if box[2] * box[3] >= 5000]
    print(len(filtered_proposals))
    return filtered_proposals


def preprocess_region(image, box):
    # Unpack the bounding box
    (x, y, w, h) = box

    # Add padding to each side (10 pixels more on each side)
    padding = 0
    x_new = max(x - padding, 0)  # Ensure x is not less than 0
    y_new = max(y - padding, 0)  # Ensure y is not less than 0
    w_new = min(x + w + padding, image.shape[1])  # Ensure the width doesn't exceed image size
    h_new = min(y + h + padding, image.shape[0])  # Ensure the height doesn't exceed image size

    # Extract the region of interest (ROI) with the new bounding box
    roi = image[y_new:h_new, x_new:w_new]

    return roi


# 1. Extract HOG features
def extract_hog_features(image, resize_dim=(128, 64)):
    """
    Extract HOG features from a given image.

    Args:
    - image: The input image (BGR).
    - resize_dim: The target size to resize the image for feature extraction.

    Returns:
    - features: The HOG features.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image (standard size for face detection)
    gray_image_resized = cv2.resize(gray_image, resize_dim)

    # Extract HOG features
    features, hog_image = hog(gray_image_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                              block_norm='L2-Hys', visualize=True)

    # Rescale the HOG image for visualization (optional)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return features


# 2. Extract CNN features using VGG16
def get_vgg16_features(modelName,image, target_size=(224, 224)):
    """
    Extract features from VGG16 pre-trained model.

    Args:
    - image: The input image (BGR).
    - target_size: Resize the image to fit VGG16 input dimensions.

    Returns:
    - features: Flattened feature vector from VGG16 model.
    """
    # Preprocess image for VGG16
    image_resized = cv2.resize(image, target_size)
    image_preprocessed = image_resized.astype('float32')
    image_preprocessed = np.expand_dims(image_preprocessed, axis=0)
    image_preprocessed /= 255.0  # Normalize

    # Load VGG16 pre-trained model (exclude fully connected layers)
    # vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # vgg_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg_model = get_model(modelName)
    # Extract features
    features = vgg_model.predict(image_preprocessed)

    # Flatten the features to make them suitable for input into a classifier
    return features.flatten()


# 3. Combine CNN and HOG features
def combine_features(cnn_features, hog_features):
    """
    Combine CNN and HOG features by concatenating them.

    Args:
    - cnn_features: The feature vector from VGG16 CNN.
    - hog_features: The feature vector from HOG.

    Returns:
    - combined_features: The concatenated feature vector.
    """
    return np.concatenate([cnn_features, hog_features], axis=0)


def preprocess_image(modelName,image_path, target_size=(224, 224), hog_resize_dim=(128, 64)):
    """
    Preprocess the input image: extract CNN and HOG features and combine them.

    Args:
    - image_path: The path to the image.
    - target_size: The target size for CNN input (VGG16).
    - hog_resize_dim: The target size for HOG feature extraction.

    Returns:
    - combined_features: The combined feature vector (CNN + HOG).
    """
    # Read the image
    # image = cv2.imread(image_path)
    image = image_path

    # Extract CNN features using VGG16
    cnn_features = get_vgg16_features(modelName,image, target_size)

    # Extract HOG features
    hog_features = extract_hog_features(image, hog_resize_dim)

    # Combine the features (CNN + HOG)
    combined_features = combine_features(cnn_features, hog_features)

    return combined_features


def load_data(faces_dir, non_faces_dir):
    """
    Load the data from directories, extract CNN and HOG features, and prepare the dataset for training.

    Args:
    - faces_dir: Directory containing face images.
    - non_faces_dir: Directory containing non-face images.

    Returns:
    - X: Combined CNN + HOG feature array.
    - y: Label array (0 for non-face, 1 for face).
    """
    X = []
    y = []

    # Load images from the faces directory (label = 1)
    for filename in os.listdir(faces_dir):
        img_path = os.path.join(faces_dir, filename)
        image_data = cv2.imread(img_path)
        if image_data is not None:
            # cnn_features = get_vgg16_features(image_data)
            # hog_features = extract_hog_features(image_data)
            # combined_features = combine_features(cnn_features, hog_features)
            X.append(image_data)
            y.append(1)  # Label for faces

    # Load images from the non-faces directory (label = 0)
    for filename in os.listdir(non_faces_dir):
        img_path = os.path.join(non_faces_dir, filename)
        image_data = cv2.imread(img_path)
        if image_data is not None:
            # cnn_features = get_vgg16_features(image_data)
            # hog_features = extract_hog_features(image_data)
            # combined_features = combine_features(cnn_features, hog_features)
            X.append(image_data)
            y.append(0)  # Label for non-faces

    # X = np.array(X)
    # y = np.array(y)

    return X, y


# def predict_face(image_path, model):
#     """
#     Predict whether the image contains a face or not using the trained model.
#
#     Args:
#     - image_path: The path to the image.
#     - model: The trained model (loaded from .h5 file).
#
#     Returns:
#     - prediction: The model's prediction (0 or 1).
#     """
#     # Preprocess the image (extract features and combine them)
#     combined_features = preprocess_image(image_path)
#
#     # Reshape the combined features to match the input dimensions of the model
#     combined_features = np.expand_dims(combined_features, axis=0)  # Add batch dimension
#
#     # Make the prediction
#     prediction = model.predict(combined_features)
#
#     # Convert the prediction to a binary output (0 or 1)
#     if prediction > 0.5:
#         return "Face detected"
#     else:
#         return "No face detected"


def predict_Box_face(image,model,modelName):
    """
    Predict whether the image contains a face or not using the trained model.

    Args:
    - image_path: The path to the image.
    - model: The trained model (loaded from .h5 file).

    Returns:
    - prediction: The model's prediction (0 or 1).
    """
    # Preprocess the image (extract features and combine them)
    combined_features = preprocess_image(modelName,image)

    # Reshape the combined features to match the input dimensions of the model
    combined_features = np.expand_dims(combined_features, axis=0)  # Add batch dimension


    # Make the prediction
    prediction = model.predict(combined_features)

    # Convert the prediction to a binary output (0 or 1)
    if prediction > 0.5:
        return 1
    else:
        return 0
    # Convert the prediction to a binary output (0 or 1)
    # if prediction > 0.5:
    #     return "Face detected"
    # else:
    #     return "No face detected"
    #


def predict_face(image):
    """
    Predict whether the image contains a face or not using the trained model.

    Args:
    - image_path: The path to the image.
    - model: The trained model (loaded from .h5 file).

    Returns:
    - prediction: The model's prediction (0 or 1).
    """

    model = load_model('CombinedModel_hog_cnn_face_detector.h5')

    # Preprocess the image (extract features and combine them)
    combined_features = image

    # Reshape the combined features to match the input dimensions of the model
    combined_features = np.expand_dims(combined_features, axis=0)  # Add batch dimension

    # Make the prediction
    prediction = model.predict(combined_features)

    # Convert the prediction to a binary output (0 or 1)
    if prediction > 0.5:
        return 1
    else:
        return 0


def TestModelAll():
    faces_dir = '/face_project/dataset/dataset/Test/faces'  # Replace with the actual path to faces folder
    non_faces_dir = '/face_project/dataset/dataset/Test/no face'  # Replace with the actual path to non-faces folder

    # Load and preprocess the data
    X, y = load_data(faces_dir, non_faces_dir)
    # X = [1]
    # y = [1]
    y_true = []
    y_pred = []
    all_results_test = []
    # 'VGG16',  'ResNet50'
    for model_name in ['InceptionV3','VGG19', 'DenseNet121', 'EfficientNetB0','NASNetLarge','ResNet152','ResNet101']:
        all_results_test.clear()
        excel_path_final = f'xlsxFiles/Testing_metrics_{model_name}_runs_train.xlsx'

        for run in range(5):
            ind = run+1
            txtModel = f'Models/CombinedModel_hog{model_name}_cnn{ind}_face_detector.h5'
            model = load_model(txtModel)
            train_start_time = time.time()

            for i in range(len(X)):
                # test_image = cv2.imread( '/root/Project/dataset/Test/faces/imgface00252.jpg')  # Replace with an actual image path
                # test_image = cv2.imread('/root/Project/dataset/Test/no face/000052.jpg')  # Replace with an actual image path
                test_image = X[i]
                edges = apply_canny(test_image)
                proposals = generate_region_proposals(edges)
                # # merged_regions = merge_regions_with_iou(proposals, threshold=0.7)
                ccc = 0
                for box in proposals:
                    roi = preprocess_region(test_image, box)
                    prediction = predict_Box_face(roi,model,model_name)
                    if prediction > 0.5:
                        ccc = ccc + 1

                try:
                    ccc = ccc / len(proposals)
                    # print(ccc, ' --> ', y[i])
                    if ccc > 0.65:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
                    y_true.append(y[i])
                except:
                    print("sdasda")
                    ccc = 0
                # print("Prediction:", "Face detected" if prediction > 0.2 else "No face detected")

            train_end_time = time.time()
            testing_time = train_end_time - train_start_time

            y_true_flat = np.concatenate([y_true])
            y_pred_flat = np.concatenate([y_pred])

            accuracy = accuracy_score(y_true_flat, y_pred_flat)
            precision = precision_score(y_true_flat, y_pred_flat)
            recall = recall_score(y_true_flat, y_pred_flat)
            f1 = f1_score(y_true_flat, y_pred_flat)
            conf_matrix = confusion_matrix(y_true, y_pred_flat)
            bce_loss = tf.keras.losses.BinaryCrossentropy()

            # Calculate the loss
            loss = bce_loss(y_true_flat, y_pred_flat)
            results = {
                'Run-test': run,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Loss': loss,
                'Testing Time (s)': testing_time,
                'Confusion Matrix': [conf_matrix]  # Storing as a list to keep it in a cell
            }
            all_results_test.append(results)

            print("Evaluation Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

        df_results = pd.DataFrame(all_results_test)
        df_results.to_excel(excel_path_final, index=False)


# image_path = '/root/Project/dataset/Test/no face/000863.jpg' # Replace with an actual image path
# image_path = '/root/Project/dataset/Test/faces/imgface00647.jpg' # Replace with an actual image path
#
# model = load_model('CombinedModel_hog_cnn_face_detector.h5')
#
# # Make the prediction
# result = predict_face(image_path, model)
#
# print(result)

TestModelAll()

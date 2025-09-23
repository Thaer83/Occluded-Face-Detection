import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, ResNet101, ResNet152, EfficientNetB0, \
    DenseNet121, NASNetLarge
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping
import pickle
import pandas as pd
import time
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.callbacks import Callback

import warnings

warnings.filterwarnings('ignore')


# Select which model to train: VGG16, VGG19, ResNet50, or InceptionV3
def get_model(model_name, num_classes):
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

        # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create final model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def trainGen(pathTraining, pathValidation, pathTesting):
    train_generator = train_datagen.flow_from_directory(
        pathTraining,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True)

    val_generator = val_datagen.flow_from_directory(
        pathValidation,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

    test_generator = test_datagen.flow_from_directory(
        pathTesting,  # Folder with test images
        target_size=(224, 224),  # VGG16 expects 224x224 images
        batch_size=32,
        class_mode='categorical',  # Use 'binary' for two-class problems
        shuffle=False  # To ensure true labels align with predictions
    )
    return train_generator, val_generator, test_generator


# Train and evaluate the model
def train_and_evaluate_model(model_name, run):
    num_classes = train_generator.num_classes
    # num_classes = 2

    model = get_model(model_name, num_classes)

    # metrics_logger = MetricsLogger(validation_data=val_generator)

    print(num_classes)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # model.fit(train_ds, train_labels, epochs=5, validation_split=0.2, callbacks=[es])
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)

    train_start_time = time.time()

    # Train the model
    history = model.fit(train_generator,
                        steps_per_epoch=train_generator.samples // train_generator.batch_size,
                        validation_data=val_generator,
                        validation_steps=val_generator.samples // val_generator.batch_size,
                        epochs=25,
                        callbacks=[es])

    train_end_time = time.time()
    training_time = train_end_time - train_start_time

    # predictions = model.predict(test_generator)
    # y_pred = np.argmax(predictions, axis=1)  # Predicted classes
    #     #y_true = test_generator.classes
    # y_pred_labels = (y_pred > 0.5).astype(int)  # For binary classification
    # y_true = test_generator.classes

    y_true = val_generator.classes
    y_pred_probs = model.predict(val_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Predicted classes

    y_pred = (y_pred > 0.5).astype(int)  # For binary classification

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    loss = model.evaluate(val_generator, verbose=0)[0]  # Extract loss from the evaluation

    results1 = {
        'Run Training': run,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Loss': loss,
        'Training Time (s)': training_time,
        'Confusion Matrix': [conf_matrix]  # Storing as a list to keep it in a cell
    }
    all_results.append(results1)

    # for epoch_log in metrics_logger.epoch_logs:
    #     epoch_log['Run'] = run
    #     epoch_log['Training Time (s)'] = training_time
    # all_results.append(epoch_log)
    # all_results_appends.append(epoch_log)

    # Save the history after training

    # Save history as a pickle file
    with open(f'models/history{model_name}-{run}.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    model.save(f'models/{model_name}-{run}.h5')

    # if os.path.exists(excel_path_append):
    # # If the file exists, append the data
    #     with pd.ExcelWriter(excel_path_append, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    #         # Append the new data to a specific sheet
    #         df_results_append.to_excel(writer, sheet_name='Sheet1', index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    # else:
    #     # If the file does not exist, create a new file and write the data
    #     with pd.ExcelWriter(excel_path_append, engine='openpyxl') as writer:
    #         df_results_append.to_excel(writer, sheet_name='Sheet1', index=False)

    for run in range(1, 6):
        print(f"--------Run Test {run}----------")

        # Record start time for testing
        test_start_time = time.time()

        predictions = model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)  # Predicted classes
        # y_true = test_generator.classes
        y_pred_labels = (y_pred > 0.5).astype(int)  # For binary classification
        y_true = test_generator.classes

        # Get predictions and ground truth labels
        # y_pred = model.predict(test_generator)
        # y_pred_labels = (y_pred > 0.5).astype(int)  # For binary classification
        # y_true = test_generator.classes

        # Calculate loss
        loss = model.evaluate(test_generator, verbose=0)[0]  # Extract loss from the evaluation

        # Record end time and calculate testing time
        test_end_time = time.time()
        testing_time = test_end_time - test_start_time

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred_labels)
        precision = precision_score(y_true, y_pred_labels)
        recall = recall_score(y_true, y_pred_labels)
        f1 = f1_score(y_true, y_pred_labels)
        conf_matrix = confusion_matrix(y_true, y_pred_labels)

        # Store metrics and confusion matrix for each run
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
        # all_results_test.append({'Run-test':f"--------Run Test {run}----------"})
        all_results_test.append(results)


# steps_per_epoch=train_generator.samples // train_generator.batch_size,
# Predict on the validation set
# val_generator.reset()
# predictions = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1)
# y_pred = np.argmax(predictions, axis=1)
# y_true = val_generator.classes

# # Print accuracy, precision, recall, F1-score
# print(classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys()))

# # Confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
# plt.title(f"Confusion Matrix for {model_name}")
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()


# Example of running the training and evaluation
# for model_name in ['VGG16',  'ResNet50', 'InceptionV3','VGG19', 'DenseNet121', 'EfficientNetB0','NASNetLarge',]:
#     print(f"Training and evaluating {model_name}...")
#     train_and_evaluate_model(model_name)

# Prepare data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1. / 255)

path_train = 'dataset/dataset/final_dataset'
path_val = 'dataset/dataset/validation'
path_test = 'dataset/dataset/Test'

train_generator, val_generator, test_generator = trainGen(path_train, path_val, path_test)
all_results = []
all_results_test = []

# excel_path_append = '/content/drive/MyDrive/training_metrics_ResNet50_5_runs_append.xlsx'
# 'EfficientNetB0'
excel_path_final = ''
excel_path_final_Test = ''
for model_name in ['VGG16',  'ResNet50','VGG19', 'DenseNet121', 'EfficientNetB0','NASNetLarge','InceptionV3','ResNet152','ResNet101']:
# for model_name in ['yolo']:
    all_results.clear()
    for i in range(1, 6):
        # model_name = 'ResNet50'
        all_results_test.clear()
        excel_path_final = f'excels/training_metrics_{model_name}_5_runs_train.xlsx'
        excel_path_final_Test = f'excels/training_metrics_{model_name}_5_runs_test.xlsx'

        print(f"Training and evaluating {model_name}... Run {i}")
        # all_results.append({'Run Training':f"--------Run - {i}----------"})
        train_and_evaluate_model(model_name, i)

    df_results = pd.DataFrame(all_results)
    df_results_test = pd.DataFrame(all_results_test)

    # Save DataFrame to Excel
    df_results.to_excel(excel_path_final, index=False)

    df_results_test.to_excel(excel_path_final_Test, index=False)

print(f"Training metrics for 5 runs saved to {excel_path_final}")

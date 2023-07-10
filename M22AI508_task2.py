#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import tarfile

# Path to the tar.gz file
tar_file = 'C:\\Users\\sakla\\Downloads\\stl10_binary.tar.gz'

# Destination folder to save the extracted data
extracted_folder = 'C:\\Users\\sakla\\Downloads\\stl10_extracted'

# Create the destination folder if it doesn't exist
os.makedirs(extracted_folder, exist_ok=True)

# Extract the tar.gz file
with tarfile.open(tar_file, 'r:gz') as tar:
    tar.extractall(extracted_folder)

print('Extraction completed.')


# In[23]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import confusion_matrix, roc_auc_score
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Specify the data directory
data_dir = 'C:\\Users\\sakla\\Downloads\\stl10_extracted\\stl10_binary\\'

# Load train data
train_img = np.fromfile(os.path.join(data_dir, 'train_X.bin'), dtype=np.uint8)
train_img = np.reshape(train_img, (-1, 3, 96, 96)).transpose(0, 2, 3, 1)
train_title = np.fromfile(os.path.join(data_dir, 'train_y.bin'), dtype=np.uint8) - 1

# Load test data
test_img = np.fromfile(os.path.join(data_dir, 'test_X.bin'), dtype=np.uint8)
test_img = np.reshape(test_img, (-1, 3, 96, 96)).transpose(0, 2, 3, 1)
test_title = np.fromfile(os.path.join(data_dir, 'test_y.bin'), dtype=np.uint8) - 1

# Convert labels to categorical
train_title = tf.keras.utils.to_categorical(train_title)
#val_title = tf.keras.utils.to_categorical(val_title)
test_title = tf.keras.utils.to_categorical(test_title)

# Split dataset into training and validation sets
train_img, val_images, train_title, val_title = train_test_split(train_img, train_title, test_size=0.2, random_state=42)

# Load the pre-trained autoencoder
autoencoder = models.load_model('C:/Users/sakla/Downloads/stl10_extracted/stl10_binary/autoencoder_best.h5', compile=False)
encoder = autoencoder.layers[1]  # Assuming the encoder is the second layer

# Specify the optimizer for the autoencoder
optimizer = tf.keras.optimizers.Adam()

# Compile the autoencoder with the mean squared error loss
autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

# Create a feature extractor model using the autoencoder's encoder
f_extract = models.Model(inputs=autoencoder.input, outputs=encoder.output)

# Resize the images for the feature extractor
resized_train_img = tf.image.resize(train_img, (64, 64))
resized_val_images = tf.image.resize(val_images, (64, 64))
resized_test_img = tf.image.resize(test_img, (64, 64))

# Print the shapes of the resized images
print("Resized train images shape:", resized_train_img.shape)
print("Resized val images shape:", resized_val_images.shape)
print("Resized test images shape:", resized_test_img.shape)

# Extract features from the resized images using the feature extractor
train_feat = f_extract.predict(resized_train_img)
val_feat = f_extract.predict(resized_val_images)
test_feat = f_extract.predict(resized_test_img)

# Define the hidden layers for the MLP classifiers
h3 = [256, 128, 64]
h5 = [512, 256, 128, 64, 32]

def mlp_classifier(hidden_layers):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=train_features.shape[1:]))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


# Build an MLP classifier with 3 hidden layers
classifier3 = mlp_classifier(h3)
classifier3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier3.fit(train_feat, train_title, validation_data=(val_feat, val_title), epochs=10)

# Build an MLP classifier with 5 hidden layers
classifier5 = mlp_classifier(h5)
classifier5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier5.fit(train_feat, train_title, validation_data=(val_feat, val_title), epochs=10)

# Specify different training sample percentages for fine-tuning the classifiers
train_percent = [0.01, 0.1, 0.2, 0.4, 0.6]
classifiers = [classifier3, classifier5]
res = {}

# Fine-tune the classifiers with different training sample percentages
for classifier in classifiers:
    classifier_res = []
    for sample_percent in train_percent:
        # Randomly sample a subset of training data
        sample_indices = np.random.choice(len(train_feat), int(sample_percent * len(train_feat)), replace=False)
        sampled_feat = train_feat[sample_indices]
        sampled_labels = train_title[sample_indices]

        # Fine-tune the classifier on the sampled data
        classifier.fit(sampled_feat, sampled_labels, epochs=5, verbose=0)

        # Evaluate the classifier on the test set
        test_loss, test_acc = classifier.evaluate(test_feat, test_title, verbose=0)

        classifier_res.append(test_acc)

    res[classifier.name] = classifier_res

    
# Evaluate performance using confusion matrix and AUC-ROC curve    
def eval_perf(classifier, features, labels):
    predictions = classifier.predict(features)
    y_pred = np.argmax(predictions, axis=1)
    cm = confusion_matrix(np.argmax(labels, axis=1), y_pred)
    auc_roc = roc_auc_score(labels, predictions, multi_class='ovr')
    return cm, auc_roc


cm_3, auc_roc_3 = eval_perf(classifier_3, test_features, test_labels)
cm_5, auc_roc_5 = eval_perf(classifier_5, test_features, test_labels)

conf_matrix = {}
roc = {}

# Evaluate performance for MLP classifiers
for classifier in classifiers:
    classifier_name = classifier.name
    conf_mat, auc_roc = eval_perf(classifier, test_feat, test_title)
    conf_matrix[classifier_name] = conf_mat
    roc[classifier_name] = auc_roc

# Build a CNN classifier using the VGG16 model as the base
def cnn():
    base = VGG16(include_top=False, weights='imagenet', input_shape=train_img.shape[1:])
    for layer in base.layers:
        layer.trainable = False
    model = models.Sequential()
    model.add(base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Build and train a CNN classifier
cnn_classifier = cnn()
cnn_classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_classifier.fit(train_img, train_title, validation_data=(val_images, val_title), epochs=10)

# Evaluate CNN classifier performance
conf_mat_cnn, auc_roc_cnn = eval_perf(cnn_classifier, test_img,test_title)


# In[ ]:





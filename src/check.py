# Import necessary utilities for handling model training and evaluation
from transfer_model_utils import *
import pandas as pd
import numpy as np
import os
from PIL import Image
from io import BytesIO # reading bytes
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard # graphical visual of loss and accuracy over the epochs of train and test set

# Suppress the detailed logging of TensorFlow, such as oneDNN messages, unless it's an error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # This should be set before importing TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Setting the seed for numpy-generated random numbers
np.random.seed(42)

# Setting the seed for python random numbers
tf.random.set_seed(42)


# AWS S3
import boto3

# Images
from PIL import Image
import matplotlib.image as mpimg # show images
from io import BytesIO # reading bytes

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, roc_curve, precision_recall_fscore_support
from sklearn import metrics

from sklearn.preprocessing import LabelBinarizer

# progress bar
from tqdm import tqdm
getattr(tqdm, '_instances', {}).clear()

# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization # CNN
from tensorflow.keras.models import Model

from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop

from tensorflow.keras.callbacks import TensorBoard # graphical visual of loss and accuracy over the epochs of train and test set
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import datetime

#initializing boto3 client

session = boto3.Session(
    aws_access_key_id='AKIATIIBMARCYEIUQVUG',
    aws_secret_access_key='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',
    region_name='us-east-2'
)
s3 = session.client('s3')
# Load a CSV file containing image orders and their metadata, setting the first column as the index
imagedata = pd.read_csv('data/imagedata.csv', index_col=0)
# Define the directory where images are stored
img_dir = 'bird_dataset'
# Specify the file paths of the images within the DataFrame
paths = imagedata['file_path']
# Define the S3 bucket where the images are hosted
bucket = 'birdbucket23'

def resize_images_array(img_dir, file_path):
    img_arrays = []
    for path in tqdm(file_path):
#        s3 = boto3.client('s3')
        try:
            obj = s3.get_object(Bucket=bucket, Key=f'{img_dir}/{path}')
            img_bytes = BytesIO(obj['Body'].read())
            open_img = Image.open(img_bytes)
            arr = np.array(open_img.resize((299,299))) # (299,299) required for Xception
            img_arrays.append(arr)
        except Exception as e:
            print(f"Error processing {path}: {e}")
        

    return np.array(img_arrays)
# obtain image data in arrays
X = resize_images_array(img_dir, imagedata['file_path'])
if len(X) == 0:
    raise ValueError("No images were loaded into the array X.")
else: 
    print("success")

# normalize RGB values
X = X/255.0
# grab label
# INPUT VALUES MUST BE ARRAYS
label = np.array(imagedata['species_group'].values)

# labels are alphabetical with np.unique
y = (label.reshape(-1,1) == np.unique(imagedata['species_group'])).astype(float)
# number of outputs/labels available and image input size
n_categories = y.shape[1]
input_size = (299,299,3)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Set tensorboard callback with specified folder and timestamp
tensorboard_callback = TensorBoard(log_dir='logs/', histogram_freq=1)

# create transfer model
transfer_model = create_transfer_model((299,299,3),n_categories)

# change new head to the only trainable layers
_ = change_trainable_layers(transfer_model, 132)

# compile model
transfer_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Reduce the batch size to accommodate memory limitations
batch_size = 32 # Reduced batch size
# fit model
history = transfer_model.fit(X_train, y_train, batch_size=batch_size, epochs=15, validation_split=0.1, callbacks=[tensorboard_callback])

transfer_model.save('saved_models/species3_xception.h5')

print('Model saved.')
# load_L_xception = tf.keras.models.load_model('saved_models/large_xception.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

df = pd.DataFrame(acc, columns=['accuracy'])
df['val_accuracy'] = val_acc
df['loss'] = loss
df['val_loss'] = val_loss

df.to_csv('data/accuracy.csv')
print('Accuracy CSV saved.')

pred_prob = transfer_model.predict(X_test)
print('X_test predicted')

pred_arr = []

for i in pred_prob:
    i[i.argmax()] = 1
    i[i < 1] = 0
    print(i)
    pred_arr.append(i)
    
pred_arr = np.array(pred_arr)

print('Starting SKLEARN CLASSIFICATION REPORT')
sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=pred_arr)
print(sk_report)
np.save("data/sk_report.npy", sk_report)
print('sk_report saved.')

print('Begin custom CLASS REPORT')
report_with_auc = class_report(
    y_true=y_test, 
    y_pred=pred_arr)
print('report variable created')
print(report_with_auc)
report_with_auc.to_csv('data/class_report_xception.csv')
print('report saved.')

print('Starting Confusion Matrix...')
conf_mat = confusion_matrix(y_test.argmax(axis=1), pred_arr.argmax(axis=1))
np.savetxt('data/confusion_matrix.csv', conf_mat)

print('Starting recall score...')
recall = recall_score(y_test.argmax(axis=1),pred_arr.argmax(axis=1), average='micro')
print('recall variable obtained.')
np.save("data/recall.npy", recall)

print('Onto Classification Report...')
classify = classification_report(y_test.argmax(axis=1), pred_arr.argmax(axis=1))
print('classify variable obtained.')
np.save("data/classify.npy", classify)

# print('Starting ROC Curve Plot')

# fpr, tpr, thresholds = roc_curve(y_test, pred_arr)
# fig, ax = plt.subplots(figsize=(8,6))
# auc_score = metrics.roc_auc_score(y_test, pred_arr)
# plot_roc_curve(ax, fpr, tpr, auc_score,'Xception ROC Curve')
# plt.savefig('graphs/test_xception_roc_curve.png')
# print('ROC saved')

print('End.')





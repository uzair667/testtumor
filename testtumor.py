import numpy as np
import cv2

import PIL.Image as Image
import os

import matplotlib.pylab as plt

import glob 
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def hello(request):
    return render(request,'test.html')

def test(request):
    # New Section
    # return HttpResponse("Tada")
    IMAGE_SHAPE = (224, 224)

    print("Classeifier loading...")
    classifier = tf.keras.Sequential([
        hub.KerasLayer('E:\classifier_resnet', input_shape=IMAGE_SHAPE+(3,))
    ])

    print("Classeifier loaded...")

    dataset_url = "E:\Training"
    os.listdir(dataset_url)

    img_dir = "E:\Training\glioma_tumor"
    data_path = os.path.join(img_dir,'*g') 
    glioma_data = glob.glob(data_path) 
    glioma = [] 
    for image in glioma_data: 
        img = cv2.imread(image) 
        glioma.append(img)
    glioma[1].shape    

    img_dir = "E:\Training\meningioma_tumor"
    data_path = os.path.join(img_dir,'*g') 
    meningioma_data = glob.glob(data_path) 
    meningioma = [] 
    for image in meningioma_data: 
        img = cv2.imread(image) 
        meningioma.append(img)
    meningioma[1].shape    

    img_dir = "E:\Training\pituitary_tumor"
    data_path = os.path.join(img_dir,'*g') 
    pituitary_data = glob.glob(data_path) 
    pituitary = [] 
    for image in pituitary_data: 
        img = cv2.imread(image) 
        pituitary.append(img)
    if pituitary:
        pituitary[1].shape

    img_dir = "E:\Training\no_tumor"
    data_path = os.path.join(img_dir,'*g') 
    no_tumor_data = glob.glob(data_path) 
    no_tumor = [] 
    for image in no_tumor_data: 
        img = cv2.imread(image)
        img.shape 
        no_tumor.append(img)    

    tumor_images_dict = {
        'glioma': glioma,
        'meningioma': meningioma,
        'pituitary': pituitary,
        'no_tumor': no_tumor,
    }

    tumor_label_dict = {
        'glioma': 0,
        'meningioma': 1,
        'pituitary': 2,
        'no_tumor': 3,
    }

    x, y = [], []

    for tumor_name, images in tumor_images_dict.items():
        for image in images:
            resized_img = cv2.resize(image,(224,224))
            x.append(resized_img)
            y.append(tumor_label_dict[tumor_name])

    x = np.array(x)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    X_train_scaled = X_train / 255
    X_test_scaled = X_test / 255

    feature_extractor_model = "E:\feature"

    pretrained_model_without_top_layer = hub.KerasLayer(
        feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

    num_of_tumor = 4

    model = tf.keras.Sequential([
    pretrained_model_without_top_layer,
    tf.keras.layers.Dense(num_of_tumor)
    ])

    print (model.summary)

    model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

    model.fit(X_train_scaled, y_train, epochs=32, batch_size=100, validation_data=(X_val, y_val))

    model.evaluate(X_test_scaled,y_test)
    model.evaluate(X_val, y_val)
    # return HttpResponse("Success!")
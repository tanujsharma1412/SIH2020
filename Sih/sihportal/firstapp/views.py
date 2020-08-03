from django.shortcuts import render
# Create your views here.


from django.core.files.storage import FileSystemStorage

import tensorflow as tf
from tensorflow import keras
from tensorflow import Graph
from tensorflow.keras import layers

from keras.models import load_model
from keras.preprocessing import image

from mtcnn.mtcnn import MTCNN
# import dlib
from collections import Counter

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


# import tensorflow as tf
import dlib
import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
# from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import InceptionResNetV2


googleNet_model = InceptionResNetV2(
    include_top=False, weights='imagenet', input_shape=(128, 128, 3))
googleNet_model.trainable = True
model = Sequential()
model.add(googleNet_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(
                  lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

model.load_weights('./models/deepfake_detection_model.h5')

input_shape = (128, 128, 3)




def index(request):
    context = {'a' : 1}
    return render(request, 'index.html',context)


def predictVideo(request):
    # print(request)
    # print(request.POST.dict())
    pr_data = []
    detector = dlib.get_frontal_face_detector()
    FileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(FileObj.name, FileObj)
    filePathName = fs.url(filePathName)
    testimage = '.'+filePathName

    cap = cv2.VideoCapture(testimage)
    frameRate = cap.get(5)
    

    x = []
    y = 0
    while cap.isOpened():
        # print("Inside while loop ->>>>>>>>")
        frameId = cap.get(1)
        ret, frame = cap.read()
        if ret != True:
            break
        if frameId % ((int(frameRate)+1)*1) == 0:
            detector = MTCNN()
            # detect faces in the image
            # result=temp1.most_common(1)
            faces = detector.detect_faces(frame)
            
            for i, d in enumerate(faces):
                y = y + 1
                x1 = d['box'][0]
                y1 = d['box'][1]
                x2 = d['box'][2]
                y2 = d['box'][3]
                crop_img = frame[y1:y1+y2, x1:x1+x2, :]
                data = img_to_array(cv2.resize(crop_img, (128, 128))).flatten() / 255.0
                data = data.reshape(-1, 128, 128, 3)

                out = model.predict_classes(data)
                x.append(out[0])

    count0 = 0
    count1 = 0
    for i in x:
        if i == 0:
            count0 = count0 + 1
        elif i == 1:
            count1 = count1 + 1

    # print("---------------> ", count0)
    # print("---------------> ", count1)
    # print("---------------> ", y)
    if count0 == y:
        predictedLabel = "Fake"
    else:
        predictedLabel = "Real"

    context = {'filePathName': filePathName, 'predictedLabel': predictedLabel}
    return render(request, 'index.html', context)


def viewDataBase(request):
    import os
    listOfImage = os.listdir('./media')
    listOfImagesPath = ['./media/' + i for i in listOfImage]
    context = {'listOfImagesPath': listOfImagesPath}
    return render(request, 'viewDB.html', context)


mobnet_model = InceptionResNetV2(
    include_top=False, weights='imagenet', input_shape=input_shape)
mobnet_model.trainable = True
image_model = Sequential()
image_model.add(mobnet_model)
image_model.add(GlobalAveragePooling2D())
image_model.add(Dense(units=2, activation='softmax'))
image_model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.Adam(
                        lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                    metrics=['accuracy'])

image_model.load_weights('./models/model2.h5')




def predictImage(request):
    FileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(FileObj.name, FileObj)
    filePathName = fs.url(filePathName)
    testimage = '.'+filePathName

    img = cv2.imread(testimage)

    img = cv2.resize(img, (128, 128))

    img = np.reshape(img,[-1,128,128,3])

    temp = image_model.predict_classes(img)
    print("-------------------->>>>>>>>> ")
    print("temp - >>>>>>>>>>>>> ",temp)
    if(temp == [0]):
        predictedLabel2 = "Fake"
    elif (temp == [1]):
        predictedLabel2 = "Real"




    context = {'filePathName': filePathName, 'predictedLabel2': predictedLabel2}

    return render(request, 'index.html',context)

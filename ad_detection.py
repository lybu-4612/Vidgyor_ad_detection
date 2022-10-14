import cv2
import numpy as np
import os
import random
import shutil



import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import numpy as np
import PIL
from PIL import Image, ImageStat
import math
import time
import threading
import datetime

cap = cv2.VideoCapture('sample_videos/15-40-59.mp4')
# cap.set(cv2.CAP_PROP_POS_FRAMES, 14500) # to start reading from a specific frame

def rescaleFrame(frame, scale):
    
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dimensions  = (width, height)
    
    return cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)

def predict(*args):
    
    global tym, timer
    tym = True
    time.sleep(0.05)
    tym = False
    timer = threading.Timer(1, predict)
    timer.start()    

predict()

white = cv2.imread('img/img_19.jpg')
gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
mask_c = cv2.bitwise_not(gray)

model = tf.keras.models.load_model('tv_add_224.h5')
labels = ['Not Ad', 'Ad','']

set_val = set([])
val = 2
start = None
end = None

video_start = datetime.datetime.now()
temp = video_start-video_start
add_start = temp
time_elapsed = temp
start_time = video_start


add_dict = {}
i = 0

while True:
   
    _, frame = cap.read()
    
    if _:
        img = frame.copy()

        masked_img  = cv2.bitwise_and(img, img, mask=mask_c)
        imgb_crop = masked_img[55:115, 1550:1805]

        if tym:
            imgWhite_crop = np.ones((224, 224, 3), np.uint8) * 255

            ar = 60/255
            k = 224 / 255
            hCal = math.ceil(k * 60)
            imgResize = cv2.resize(imgb_crop, (224, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((224 - hCal) / 2)
            imgWhite_crop[hGap:hCal + hGap, :] = imgResize


            norm=imgWhite_crop/255.0
            reshap=np.reshape(norm,(1,224,224,3))
            reshap = np.vstack([reshap])

            pred = model.predict(reshap)
            val = np.argmax(pred)

            set_val.add(val)

            if len(set_val) == 2 and val == 1:
                print('add starting')
                start = 'start'
                start_time = datetime.datetime.now()
                add_start = start_time - video_start

                set_val = set([])

            if len(set_val) == 2 and val == 0:
                print('add completed')
                end = 'finished'
                end_time = datetime.datetime.now()
                add_end = end_time - video_start
                if start:
                    time_elapsed = end_time-start_time

                    start = None
                else:
                    time_elapsed = end_time-video_start

                add_dict[f'ad_{i+1}'] = (f'start: {add_start}, stop: {add_end}, duration: {time_elapsed}')
                i += 1
                set_val = set([])


        cv2.putText(img, labels[val], (55, 125), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 255, 0), 3)
        cv2.putText(img, (str(datetime.datetime.now()-video_start)[:8]),(55, 70),cv2.FONT_HERSHEY_SIMPLEX, 1.7,(0, 255, 0), 3)

        img_resized = rescaleFrame(img, 0.5)
        cv2.imshow('video', img_resized)
#         cv2.imshow('inverted', imgWhite_crop)

        key = cv2.waitKey(19)

        if key == ord('q'):
            break
        
    else:
        if start or (len(set_val) == 1 and val == 1):
            print('add completed')
            end = 'finished'
            end_time = datetime.datetime.now()
            add_end = end_time - video_start
            time_elapsed = end_time-start_time
            add_dict[f'ad_{i+1}'] = (f'start: {add_start}, stop: {add_end}, duration: {time_elapsed}')
        break    

video_end = datetime.datetime.now()
total_time = video_end - video_start   

print(add_dict)
print('Total_time: ', total_time)

timer.cancel()
cap.release()
cv2.destroyAllWindows()
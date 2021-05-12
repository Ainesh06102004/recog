import pandas as pd
import cv2
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts
from sklearn.datasets import fetch_openml
import os, time, ssl
from PIL import Image
import PIL.ImageOps

X,y = fetch_openml("mnist_784", version = 1, return_X_y= True)
classes = ["0","1","2","3","4","5","6","7","8","9"]
print(pd.Series(y).value_counts())

xtrain, xtest, ytrain, ytest = tts(X, y, random_state = 9, train_size = 7500, test_size = 2500)

xtrainscaled = xtrain/255.0
xtestscale = xtest/255.0

lr = LogisticRegression(solver = 'saga', multi_class='multinomial').fit(xtrainscaled, ytrain)

prediction1 = lr.predict(xtestscale)
accuracy = accuracy_score(ytest, prediction1)
print(accuracy)
 
cap = cv2.VideoCapture(1)

while True:
    try:
        ret, frame = cap.read()
        print("1234")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upperleft = (int(width/2 - 50), int(height/2 - 50)) 
        bottomright = (int(width/2 + 50), int(height/2 + 50)) 
        rect = cv2.rectangle(gray, upperleft, bottomright, (0,255,0), 2)
        regofinterest = gray[upperleft[1]: bottomright[1], upperleft[0]: bottomright[0]]
        print("567")
        imagepil = Image.fromarray(regofinterest)
        imgbw = imagepil.convert('L')
        imgbwresize = imgbw.resize((28,28), Image.ANTIALIAS)
        imgbwresinvert = PIL.ImageOps.invert(imgbwresize)
        pixelfilter = 20
        minpixel = np.percentile(imgbwresinvert, pixelfilter)
        imgbwresinvertscale = np.clip(imgbwresinvert - minpixel, 0, 255)
        maxpixel = np.max(imgbwresinvert)
        imgbwresinvertscale = np.asarray(imgbwresinvertscale)/maxpixel
        print("8910")
        testsample = np.array(imgbwresinvertscale).reshape(1,780)
        prediction2 = lr.predict(testsample)
        print("predicted class is: ", prediction2)
        cv2.imshow('resframe', gray)



        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()    
    


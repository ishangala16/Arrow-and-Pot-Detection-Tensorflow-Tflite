
import cv2 
import numpy as np
import os
import os.path
import time

cap = cv2.VideoCapture(0)
f = r'C:\Users\Ishan\Downloads\Arrow Images\Copy Arrow Test images'

for file in os.listdir(f):
    f_img = f+"/"+file
    if file.endswith(".jpg"):
        # ret, frame = cap.read()
        frame = cv2.imread(f_img)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        # cv2.imshow('xyz', gray)
        lower_orange = np.array([5,116,190], dtype=np.uint8)
        upper_orange = np.array([13,255,255], dtype=np.uint8)
        lower_white = np.array([0,0,99], dtype=np.uint8)
        upper_white = np.array([179,35,255], dtype=np.uint8)
        orange = cv2.inRange(hsv,lower_orange,upper_orange)
        white = cv2.inRange(hsv,lower_white,upper_white)
        mask = orange + white

        ora = cv2.bitwise_and(frame,frame, mask= orange)
        whi = cv2.bitwise_and(frame,frame, mask= white)
        xy = cv2.bitwise_and(frame,frame,mask=mask)
        cv2.imshow('orange', ora)
        cv2.imshow('white', whi)
        cv2.imshow('xy', xy)
        cv2.imwrite(f_img,xy)
        print(file)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            exit()
            break
        # time.sleep(1)
        continue
    else:
        continue
        
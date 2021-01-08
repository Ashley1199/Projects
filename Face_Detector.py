# import computer vision library
import cv2
from random import randrange
#create a classifier using cv2 haarcascade dataset
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# read image in cv2
#img = cv2.imread('rdj.png')
#capture video from default camera(Webcam)
webimg = cv2.VideoCapture(0)
#Iterate over every frame
while True:
    ret, frame = webimg.read()
    #convert the image to greyscale
    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #get coordinates of the face using multiscale fun
    face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)
    # Draw a rectangle
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,256,0),2)
    #show image in cv2
    cv2.imshow('My_Face_Detector',frame)
    #automatic key press after every ms
    key=cv2.waitKey(1)
    if key==81 or key==113:
        break
print("Code Completed")
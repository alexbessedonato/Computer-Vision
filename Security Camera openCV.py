import cv2
import time
import datetime

#specifying the video capture device we are going to be using, i only have a webcam so index 0 is used
capture = cv2.VideoCapture(0)
#set up a cascade classifier, so we pass de base directory (cv2.data.haarcascades) and then add the actual name of the classifier ('haarcascade_frontalface_default.xml') 
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_fullbody_default.xml')

recording = True

frame_size = (int(capture.get(3)), int(capture.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
outputstream = cv2.VideoWriter("video.mp4", fourcc, 20, frame_size)

while True:
    #we are telling the program to read single frames from the video and display them, "x" is a placeholder 
    x, frame = capture.read()
    #the cascade classifier requires the video image to be in grayscale therefore we have to turn the video greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #we are programming the face recognition. detectmultiscale is the command and the parameters include our greyscale image, an accuracy/speed indicator (the 1.3) which goes between 1-1.5. the lower it gets the more accuracy it has but the slower the program runs
    #this is going to return a list of positions of all the faces it detects
    #the parameter(5) is called minimum number of neighbours, essentially the program is going to detect hundreds of faces out of 1 face, therefore we tell the program that if 5 faces are detected near to each other, it can determine it as a single face  
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = face_cascade.detectMultiScale(gray, 1.3, 5)

    outputstream.write(frame)

    if len(faces) + len(bodies) > 0:
        recording = True


    for(x, y, width, height) in faces:
        #here we are drawing the rectangles to show that the faces are being detected, the parameters include (on what window the rectangles are being displayed, the top left and bottom right coordinates for the size of the rectangle, the colour (BGR) and finally line thickness of the rectangle)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)
   
    #title of the window that shows the collected frames
    cv2.imshow("camera", frame)
    #wait for 1 second, if the q key is pressed, break
    if cv2.waitKey(1) == ord('q'):
        break
#allows when program ends, to get rid of permissions and use of camera so other programs can use it
capture.release()
outputstream.release()
cv2.destroyAllWindows()




#image = cv2.imread('assets/opencv.png', 1)
#resizing an image
#imageResize = cv2.resize(image, (400, 400))
#cv2.imwrite('new_img.jpg', image)
#cv2.imshow('openCVImage', image)
##cv2.waitKey(0)#
#cv2.destroyAllWindows()

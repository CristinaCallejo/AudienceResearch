import numpy as np
#from keras.models import model_from_json
import matplotlib.pyplot as plt
import face_recognition
from pathlib import Path
import os
import sys
sys.path.append('../src')
import cv2 # for OpenCV bindings

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


# HAAR CASCADE CLASSIFIER
def detect_face_eyes_smile(pth):
    """
    Extracts all .jpg files from local path, 
    calls on haar cascade classifiers (frontalface, eyes and smile) 
    and draws detection rectangles on each .jpg 
    
    Takes: local path of directory with .jpg images
    
    Returns: individual windows for .jpg files with detection rectangles for face, eyes and smile
    """

    counter_imgs = 0
    counter_faces = 0
    counter_smiles = 0
    counter_eyes = 0
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
    
    for file in sorted(pth.iterdir()):
        if file.suffix != '.jpg':
            pass
        else:
            counter_imgs += 1
            print(file.name)
            img = cv2.imread(str(file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plt.imshow(img)

            # FRONTAL FACE 
            
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.06,                
                minNeighbors=7,
                minSize=(30, 30), 
                flags=cv2.CASCADE_SCALE_IMAGE)
            if faces is None:
                print("No Face Found")

            for (fx,fy,fw,fh) in faces:
                counter_faces += 1
                roi_gray = gray[fy:fy+fh, fx:fx+fw] # region of interest for detection
                roi_color = img[fy:fy+fh, fx:fx+fw] # region of interest for mapping rectangle
                cv2.rectangle(
                    img,
                    (fx,fy),
                    (fx+fw,fy+fh),
                    #(127,0,255),
                    (0,255,0),
                    2)

                # SMILES 

                smiles = smile_cascade.detectMultiScale(
                    roi_gray, 
                    scaleFactor = 1.35, 
                    minNeighbors = 8)

                for (sx, sy, sw, sh) in smiles:
                    counter_smiles += 1
                    cv2.rectangle(
                        roi_color,
                        (sx, sy),
                        (sx + sw, sy + sh),
                        #(255, 0, 130),
                        #(0,220,80),
                        (127,0,255),
                        1)

                # EYES

                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.05,
                    minNeighbors = 6)

                for (ex,ey,ew,eh) in eyes:
                    counter_eyes += 1
                    cv2.rectangle(
                        roi_color, 
                        (ex , ey),
                        (ex + ew, ey + eh),
                        (0,255,255),
                        1)
            
                # save images with detected regions
                file_to_save = file.name.replace(".",f"_face{counter_faces}.")
                #cv2.imwrite(str(pth.parent/'demo_faces'/file_to_save),img)
                cv2.imwrite(str(pth.parent/'demo_faces'/file_to_save),roi_color)
                counter_imgs = 0
                counter_faces = 0
            # show the output frame
            cv2.imshow(f"img{file_to_save}", img)
            key = cv2.waitKey(3) & 0xFF

        # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                # do a bit of cleanup
                cv2.destroyAllWindows()
                break
        
    # do a bit of cleanup
    cv2.destroyAllWindows()
cv2.destroyAllWindows()


"""

HaarCascade Classifiers:
    
    If faces are found, returns the positions of detected faces as Rect(x,y,w,h).

    cv2.CascadeClassifier.detectMultiScale(
        image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) 

        image: Matrix of the type CV_8U containing an image where objects are detected.
        scaleFactor (max recommendd: 1,4) : how much the image size is reduced at each image scale.
            creates scale pyramid. For scaleFactor 1.03: using a small step for resizing,
            i.e. reduce size by 3 %
            --> increase the chance of a matching size with the model for detection,but it's expensive.
        minNeighbors (recommended 3-6) : many rectangles (neighbors) need to be detected 
            for the window to be labeled a face.how many neighbors each candidate rectangle should have to retain it. 
            will affect the quality of the detected faces: 
            higher value results in less detections but with higher quality.
            We're using 5 in the code.
        flags : Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. 
            Not used for a new cascade.
        minSize : pixels(30x30 recommended) windows/objects minimum possible size. 
            Objects smaller than that are ignored.
        maxSize : Maximum possible object size. 
            Objects larger than that are ignored.
            
    Haar cascades tend to be very sensitive to your choice
    in detectMultiScale parameters. 
    The scaleFactor and minNeighbors being the ones you have to tune most often.

"""
# !!! Why, when and how to use Haar vs HOG + Linear SVM, SSD, YOLO + capturing from video implementation
    # https://www.pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/
# Params explained
    # https://towardsdatascience.com/computer-vision-detecting-objects-using-haar-cascade-classifier-4585472829a9



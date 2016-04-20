#!/usr/bin/python
# import from parent directory
import time
import cv2
from glob import glob
import time
from PIL import Image
import numpy as np
import sys
import os.path
import DynamicObjectV2
import picamera
from picamera.array import PiRGBArray
from picamera import PiCamera
import io

#face tracking pos values
# x axis = Position_x min = 0, max = 120
# y axis = Position_y min = 0, max = 93
# scale axis = Position_z min = 50, max = 150
          
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
Obj = DynamicObjectV2.Class

# Import the required modules
print "Initialising..."

# For face detection we will use the Haar Cascade provided by OpenCV.
face_cascade1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_cascade2 = cv2.CascadeClassifier('face.xml')#lbpcascade_profileface.xml

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.face.createLBPHFaceRecognizer()#recognizer = cv2.createLBPHFaceRecognizer()

yesChoice = ['yes','y']
noChoice = ['no','n']

FRAME_W= 1920#960
FRAME_H= 1080#540

HAAR_FACES         = 'haarcascade_frontalface_default.xml'
haar_faces = cv2.CascadeClassifier(HAAR_FACES)
HAAR_SCALE_FACTOR  = 1.3
HAAR_MIN_NEIGHBORS = 4
HAAR_MIN_SIZE      = (30, 30)
FACE_WIDTH  = (92/3)
FACE_HEIGHT = (112/3)
  

#setup position variables
Position_x = 0;
Position_y = 0;
Position_z = 0;
facecount=False;

#setup servo varables
stepsize = 5
servo0 = 0
servo1 = 0
servo2 = 0
servomin=0
servomax=90

path = str(os.path.realpath(__file__)).split("/")
path.pop() # remove last element i.e. this filename
strpath = ""
for element in path:
    strpath += element + "/"

def init(self):
  # put your self.registerOutput here
  self.registerOutput("facePos", Obj("x", 0, "y", 0,"z", 0))
  self.registerOutput("faceDet", Obj("Face", False))
  self.registerOutput("faceRec", Obj("Subject", 0))
  #self.registerOutput("Servo", Obj("ServoX", 0,"ServoY", 0))
  
def run (self):

  def Tracking1():
      # capture frames from the camera
      data = io.BytesIO()
      with picamera.PiCamera() as camera:
          camera.capture(data, format='jpeg')
          x1=0;
          y1=0;
          w1=0;
          h1=0;
          fn=0;
          soundflag=0;
          # Capture frame-by-frame
          facecount=False;
          data = np.fromstring(data.getvalue(), dtype=np.uint8)
          # Decode the image data and return an OpenCV image.
          image = cv2.imdecode(data, 1)
	  capturedimg = cv2.resize(image,(500,500), interpolation = cv2.INTER_CUBIC)
	  cv2.imshow("Frame",capturedimg)
          # Save captured image for debugging.
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
          frontfaces = face_cascade1.detectMultiScale(gray,
              scaleFactor=HAAR_SCALE_FACTOR, 
              minNeighbors=HAAR_MIN_NEIGHBORS, 
              minSize=HAAR_MIN_SIZE, 
              flags=cv2.CASCADE_SCALE_IMAGE)
          
          sidefaces = face_cascade2.detectMultiScale(gray, 
                          scaleFactor=HAAR_SCALE_FACTOR, 
                          minNeighbors=HAAR_MIN_NEIGHBORS, 
                          minSize=HAAR_MIN_SIZE, 
                          flags=cv2.CASCADE_SCALE_IMAGE)
      
          for (x1, y1, w1, h1) in frontfaces:
              facecount=True;
              crop_height = int((FACE_HEIGHT / float(FACE_WIDTH)) * w1)
              midy = y1 + h1/2
              y2 = max(0, midy-crop_height/2)
              y3 = min(image.shape[0]-1, midy+crop_height/2)
              sub_face2=gray[y2:y3, x1:x1+w1]
              cv2.imwrite('./modules/detected_face/subject69.stranger.png', sub_face2)
                            
          if facecount == False:
              for (x1, y1, w1, h1) in sidefaces:
                  facecount=True;
                  crop_height = int((FACE_HEIGHT / float(FACE_WIDTH)) * w1)
                  midy = y1 + h1/2
                  y2 = max(0, midy-crop_height/2)
                  y3 = min(image.shape[0]-1, midy+crop_height/2)
                  sub_face2=gray[y2:y3, x1:x1+w1]
                  cv2.imwrite('./modules/detected_face/subject69.stranger.png', sub_face2)
                  
          # Get the center of the face
          x2 = ((x1+w1)/2)
          y2 = ((y1+h1)/2)
          z2 = ((w1+h1)/2)
          # Correct relative to center of image
          Position_x  = float(((FRAME_W/2)-x2))
          Position_y  = float((FRAME_H/2)-y2)
          Position_z  = float(z2)

          if facecount == False:
                print("face not detected")
          else:
                print("face detected")
                            
          if input in yesChoice:
              if facecount == True:
                  print "Searching database"
                  image_paths = [os.path.join(path, f) for f in os.listdir(path) ]
                  for image_path in image_paths:
                      predict_image_pil = Image.open(image_path).convert('L')
                      predict_image = np.array(predict_image_pil, 'uint8')
                      faces = face_cascade1.detectMultiScale(predict_image)
                      for (x, y, w, h) in faces:
                          nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
                          nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
                          if nbr_actual == nbr_predicted:
                              print"{} Correctly recognised as {}. Confidence = {}".format(nbr_actual, nbr_predicted, conf) 
                          else:
                              print"{} Incorrectly recognised as {}. Confidence = {}".format(nbr_actual, nbr_predicted, conf)
                          if conf < 100:
                              print "Subject Identified"
                              if nbr_predicted == 16:
                                  name ='Philip'
                                  soundflag=1;
                                  self.output("faceRec", Obj("Subject", soundflag))
                              elif nbr_predicted == 17:
                                  name ='David'
                                  soundflag=2;
                                  self.output("faceRec", Obj("Subject", soundflag))
                              elif nbr_predicted == 41:
                                  name ='Robert'
                                  soundflag=3;
                                  self.output("faceRec", Obj("Subject", soundflag))
                              else:
                                  name ='Unknown'
                                  soundflag=4;
                                  self.output("faceRec", Obj("Subject", soundflag))
                          else:
                              print "Subject Unknown"
                              name ='Unknown'
                              soundflag=5;
                              self.output("faceRec", Obj("Subject", soundflag))
                          print(name)
                          cv2.putText(predict_image[y: y + h, x: x + w],''+str(name),(10,50), cv2.FONT_ITALIC, 2, (0, 0, 255),3)
                          cv2.imshow("Recognizing Face {}", predict_image[y: y + h, x: x + w])
                          cv2.waitKey(1000)
          return

  def Tracking2():
      # capture frames from the camera
      for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
          x1=0;
          y1=0;
          w1=0;
          h1=0;
          fn=0;
          soundflag=0;
          # Capture frame-by-frame
          facecount=False;
          # grab the raw NumPy array representing the image, then initialize the timestamp
          # and occupied/unoccupied text
          image = frame.array    
	  
	  # Save captured image for debugging.
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
          frontfaces = face_cascade1.detectMultiScale(gray,
              scaleFactor=HAAR_SCALE_FACTOR, 
              minNeighbors=HAAR_MIN_NEIGHBORS, 
              minSize=HAAR_MIN_SIZE, 
              flags=cv2.CASCADE_SCALE_IMAGE)
          
          sidefaces = face_cascade2.detectMultiScale(gray, 
                          scaleFactor=HAAR_SCALE_FACTOR, 
                          minNeighbors=HAAR_MIN_NEIGHBORS, 
                          minSize=HAAR_MIN_SIZE, 
                          flags=cv2.CASCADE_SCALE_IMAGE)
      
          for (x1, y1, w1, h1) in frontfaces:
              facecount=True;
              cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 0, 255), 2)
                            
          if facecount == 0:
              for (x1, y1, w1, h1) in sidefaces:
                  facecount=True;
                  cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 0, 255), 2)
                  
          # Get the center of the face
          x2 = ((x1+w1)/2)
          y2 = ((y1+h1)/2)
          z2 = ((w1+h1)/2)
          # Correct relative to center of image
          Position_x  = float(((FRAME_W2/2)-x2))
          Position_y  = float((FRAME_H2/2)-y2)
          Position_z  = float(z2)

          if facecount == False:
                print("face not detected")
          else:
                print("face detected")
                self.output("facePos", Obj("x", Position_x, "y", Position_y,"z", Position_z))
                
          self.output("faceDet", Obj("Face", facecount))      
          cv2.imshow("Frame",image)
	  cv2.waitKey(100)
          # clear the stream in preparation for the next frame
          rawCapture.truncate(0)
          return 
        
##  def Moveing():
##      # capture frames from the camera
##      global servo0, servo1
##      ### no face detected return to home position 
##      if facecount == 0:
##          servo0 = 45
##          servo1 = 45
##
##      ###face detected 
##      if facecount != 0:
##          ### limit the range of the servo positions
##          print ("posX=%d" %(servoposX))
##          print ("posY=%d" %(servoposY))
##          if servoposX > (FRAME_W2/4) - 10:
##              if servo0 != servomin:
##                  servo0 = servo0 - stepsize            
##          elif servoposX < (FRAME_W2/4) + 10:
##              if servo0 != servomax:
##                  servo0 = servo0 + stepsize
##
##          if servoposY < 80:
##              if servo1 != servomin:
##                  servo1 = servo1 - stepsize
##          elif servoposY > 80:
##              if servo1 != servomax:
##                  sePosition_x, Position_y, facecountrvo1 = servo1 + stepsize
##          else:
##              servo0=servo0
##              servo1=servo1
##          self.output("Servo", Obj("ServoX", servo0, "ServoY", servo1))
##      return 

  def get_images_and_labels(path):
      # Append all the absolute image paths in a list image_paths
      # We will not read the image with the .sad extension in the training set
      # Rather, we will use them to test our accuracy of the training
      image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
      # images will contains face images
      images = []
      # labels will contains the label that is assigned to the image
      labels = []
      for image_path in image_paths:
          # Read the image and convert to grayscale
          image_pil = Image.open(image_path).convert('L')
          # Convert the image format into numpy array
          image = np.array(image_pil, 'uint8')
          # Get the label of the image
          nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
          # Detect the face in the image
          faces = face_cascade1.detectMultiScale(image)
          # If face is detected, append the face to images and the label to labels
          for (x, y, w, h) in faces:
              print '.',
              images.append(image[y: y + h, x: x + w])
              labels.append(nbr)
              cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
              cv2.waitKey(50)
      # return the images list and labels list
      return images, labels
    
  input = raw_input("Would you like to enable face recognition? (y/n) ").lower()
  if input in yesChoice:
      print "Loading Database."

      # Path to the Yale Dataset
      path = strpath + 'yalefaces'
      # Call the get_images_and_labels function and get the face images and the 
      # corresponding labels
      images, labels = get_images_and_labels(path)
      cv2.destroyAllWindows()

      # Perform the tranining
      print "Training..."
      recognizer.train(images, np.array(labels))
      path = strpath + 'detected_face'  
      # main loop
      while 1:
          Tracking1()
          
  if input in noChoice:
      #setup position variables
      Position_x = 0;
      Position_y = 0;
      Position_z = 0;
      facecount=False;

      #setup servo varables
      stepsize = 5
      servo0 = 0
      servo1 = 0
      servo2 = 0
      servomin=0
      servomax=90
      # Import the required modules
      print "Initialising..."
      #setup cam variables
      FRAME_W2=320
      FRAME_H2=240

      # initialize the camera and grab a reference to the raw camera capture
      camera = PiCamera()
      camera.resolution = (FRAME_W2, FRAME_H2)#faster processing set to 160, 120
      camera.framerate = 40 #max frame rate of 90 frames per second
      rawCapture = PiRGBArray(camera, size=(FRAME_W2, FRAME_H2))#faster processing set to 160, 120
      # main loop
      while 1:
          Tracking2()
          ##Moveing()

#!/usr/bin/python

import time

# import from parent directory
import sys
import os.path  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import DynamicObjectV2
Obj = DynamicObjectV2.Class

# put your imports here
#setup position variables
Position_x = 0;
Position_y = 0;
Position_z = 0;
facecount=False;

def init(self):
    # put your self.registerOutput here
    self.registerOutput("headPosition", Obj("x", 0, "y", 0))
    self.registerOutput("lampPosition", Obj("z",0))

def run (self):
    # put your init and global variables here     
    # main loop
    while 1:
        
        faceDet = self.getInputs().faceDet
        facecount = faceDet.Face;
        if facecount == True:               
            # calculations
            facePos = self.getInputs().facePos
            Position_x = facePos.x;
            Position_y = facePos.y;
            Position_z = facePos.z;
            self.message("facePos: {}".format(Obj("x", Position_x, "y", Position_y, "z", Position_z)))
            headPosition = Obj("x", 0.0083333*Position_x, "y", 0.010752*Position_y) # values are normalised to 1
            lampTemp = (0.00666666*Position_z)
            lampPosition = (lampTemp)
            
                        
            # output
            self.output("headPosition", headPosition)
            self.output("lampPosition", Obj("z", lampPosition))

            # if you want to limit framerate, put it at the end
            time.sleep(0.1)
            facecount=False;       
            


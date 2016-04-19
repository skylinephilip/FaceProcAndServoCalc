#!/usr/bin/python

import time

# import from parent directory
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import DynamicObjectV2
Obj = DynamicObjectV2.Class

# put your imports here

def init(self):
    # put your self.registerOutput here
    self.registerOutput("facePos", Obj("x", 0, "y", 0))

def run (self):
    # put your init and global variables here
    xPos = 10
    yPos = 10

    # main loop
    while 1:
        # put your logic here
        # you can use: output, getInputs, message 
        self.output("facePos", Obj("x", xPos, "y", yPos))
        xPos *= -1
        yPos *= -1

        # if you want to limit framerate, put it at the end
        time.sleep(3)
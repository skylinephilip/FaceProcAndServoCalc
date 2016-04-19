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
    self.registerOutput("test-color", Obj({"image": "none"}))

def run (self):
    # put your init and global variables here
  
    # put your logic here
    # you can use: output, getInputs, message 
    time.sleep(0.1)
    self.output("test-color", Obj({"image": "red"}))
    time.sleep(0.5)
    color = self.getInputs().color
    errorBase = "ERROR in test-color:"
    error = errorBase + " Red not detected"
    assert color.R == 255, error + " - R:" + str(color.R)
    assert color.G == 0, error + " - G:" + str(color.G)
    assert color.B == 0, error + " - B:" + str(color.B)
    self.message("Test-Color-Detect SUCCESSFUL")

    # if you want to limit framerate, put it at the end
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
    self.registerOutput("color", Obj("R", 0, "G", 0, "B", 0))

def run (self):
    # put your init and global variables here
    output = Obj("R", 0, "G", 0, "B", 0)

    # main loop
    while 1:
        # put your logic here
        # you can use: output, getInputs, message 
        if (self.flags.test):
            image = self.getInputs()["test-color"].image
            if (image == "red"):
                output.R = 255
                output.G = 0
                output.B = 0

        self.output("color", output)

        # if you want to limit framerate, put it at the end
        time.sleep(0.2)
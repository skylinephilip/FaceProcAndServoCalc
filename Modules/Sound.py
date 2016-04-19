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
    self.registerOutput("audio", Obj("playing", False))

def run (self):
    # put your init and global variables here
    playing = False

    # main loop
    while 1:
        # put your logic here
        # you can use: output, getInputs, message 
        servo = self.getInputs().servo
        self.message(servo)

        if (servo.moving): playing = True
        else: playing = False

        self.output("audio", Obj("playing", playing))

        # if you want to limit framerate, put it at the end
        time.sleep(0.5)
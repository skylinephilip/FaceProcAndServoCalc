#!/usr/bin/python

# import from parent directory
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import time

import IPCThread
IPCThread = IPCThread.Class

# put your imports here

class Class (IPCThread):

    def __init__(self, name, API):
        IPCThread.__init__(self, name, API)
        
        # put your self.registerOutput here

    def run(self):
        # put your init and global variab.les here - global variables need 'self.' in front of it

        while 1:
            # put your logic here
            # you can use: output, getInputs, message, flags

            # if you want to limit framerate, put it at the end
            time.sleep(0.2)
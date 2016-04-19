
# import from parent directory
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import DynamicObjectV2
Obj = DynamicObjectV2.Class

fromSource = [
    "FaceRecDet",
    "Servo"
]

fromClass = [
]

tests = Obj({
  "Color-Detect": ["Color-Detect"],
  "Sound": ["Sound", "Voice"],
  "GUI-Item-Detect": ["GUI", "Item-Detect"],
  "GUI-Face-Detect": ["GUI", "Face-Detect"],
  "GUI-Color-Detect": ["GUI", "Color-Detect"]
})

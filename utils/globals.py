"""Definition of Global Variables"""

import cv2
from flask import Flask
import pymongo
import string
import torch.backends.cudnn as cudnn
from craft import CRAFT
from color_utils import CTCLabelConverter, AttnLabelConverter
from model import Model
from models import *
import argparse
from collections import OrderedDict

# All the possible words corresponding to the button/text-box number
words1 = ["Continue", "Load", "System...", "Head...", "Right", "Forward", "Up", "Set Network...", "Static IP...",
          "Increment", "Yes", "Start Model", "Pause",
          "Lights always on", "Lights normal", "Deutsch", "Resume"]
words2 = ["Material...", "Unload...", "Load Model", "Setup...", "Gantry...", "Left", "Backward", "Down", "Reverse",
          "Dynamic IP...", "Test Parts...", "Lights off",
          "Next Digit", "Disable UpnP", "Enable UpnP", "English", "Stop", "No"]
words3 = ["Standby Mode...", "Machine...", "Tip...", "Select Axis", "Select Drive", "Load Upgrade...", "Last Digit",
          "Select Language...", "Espanol", "Show Time"]
words4 = ["Maintenance...", "Done...", "Cancel", "Next...", "Auto Powerdown"]

# Camera Handlers
worker_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
printer_camera = cv2.VideoCapture(3, cv2.CAP_DSHOW)
web_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Text Recognition Handlers
global_predict_string = None
global_finger_string = None
global_machine_state = None

# Database and Flask Handlers
db = pymongo.MongoClient("localhost", 27017).energy
app = Flask(__name__)

# Recording Machine States
machine_states_record = []
current_machine_state = "Initialized"

# Initialized -> Testing -> Calibration -> Heating -> Printing -> Ending
Transition_states = {
    "Initialized": "Testing",
    "Testing": "Calibration",
    "Calibration": "Heating",
    "Heating": "Printing",
    "Printing": "Ending",
    "Ending": "Ending",
    "ERROR: PLEASE INITIALIZE PRINTER": "Initialized",
    "ERROR: PRINTER NOT IN INITIALIZED POSITION: BuildPlate": "Initialized",
    "ERROR: PRINTER NOT IN INITIALIZED POSITION: Extruder": "Initialized",
    "ERROR: PRINTER NOT IN INITIALIZED POSITION: Extruder Not Detected / Initialized": "Initialized",
}

# Some Other Stuffs
STRGLO = ""
BOOL = True
StrTemp = ""

########################################################################
# Model Configurations
########################################################################





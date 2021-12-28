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

# Database Handler
db = pymongo.MongoClient("localhost", 27017).energy

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str,
                    help='pretrained refiner model')
## #######################################################################################################
parser.add_argument('--image_folder', required=False, help='path to image_folder which contains text images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
parser.add_argument('--saved_model', default='weights/TPS-ResNet-BiLSTM-Attn.pth',
                    help="path to saved_model to evaluation")
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

args = parser.parse_args()

# Finger Recognition

net = CRAFT()  # initialize

# print('Loading weights from checkpoint (' + args.trained_model + ')')
if args.cuda:
    net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
else:
    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

if args.cuda:
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

net.eval()

# LinkRefiner
refine_net = None
if args.refine:
    from refinenet import RefineNet

    refine_net = RefineNet()
    # print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
    if args.cuda:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

    refine_net.eval()
    args.poly = True

# Text Recognition
if args.sensitive:
    args.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

cudnn.benchmark = True
cudnn.deterministic = True
args.num_gpu = torch.cuda.device_count()

""" model configuration """
if 'CTC' in args.Prediction:
    converter = CTCLabelConverter(args.character)
else:
    converter = AttnLabelConverter(args.character)
args.num_class = len(converter.character)

if args.rgb:
    args.input_channel = 3
model = Model(args)
# print('model input parameters', args.imgH, args.imgW, args.num_fiducial, args.input_channel, args.output_channel,
#       args.hidden_size, args.num_class, args.batch_max_length, args.Transformation, args.FeatureExtraction,
#       args.SequenceModeling, args.Prediction)
model = torch.nn.DataParallel(model).to(device)

# load model
# print('loading pretrained model from %s' % args.saved_model)
model.load_state_dict(torch.load(args.saved_model, map_location=device))

# predict
model.eval()

# Printer Interior

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="samples", help="path to dataset")
parser.add_argument("--model_def", type=str, default="yolov3-custom.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3_journal.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="obj.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up model
printer_model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    printer_model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    printer_model.load_state_dict(torch.load(opt.weights_path))

printer_model.eval()  # Set in evaluation mode




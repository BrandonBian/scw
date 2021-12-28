from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *


import argparse


import torch

from torch.autograd import Variable

import matplotlib.pyplot as plt

import cv2


machine_states_record = []
current_machine_state = "Initialized"

# Globals # Initialized -> Testing -> Calibration -> Heating -> Printing -> Ending
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


class History:
    def __init__(self):
        self.q = list([False, False, False, False, False])

    def put(self, element):
        self.q[1:] = self.q[0:4]
        self.q[0] = element

    def sum(self):
        return int(self.q[0]) + int(self.q[1]) + int(self.q[2]) + int(self.q[3]) + int(self.q[4])

    def initialize(self):
        self.q = list([False, False, False, False, False])

    def __str__(self):
        return str(self.q[0]) + " " + str(self.q[1]) + " " + str(self.q[2]) + " " + str(self.q[3]) + " " + str(
            self.q[4])


Short_history = History()


# Tolerance = 0

def transition_criterion(axis_left_x, extruder_left_x, extruder_left_y, extruder_right_x, extruder_right_y,
                         extruder_center_x,
                         buildplate_top_y, cur_state, tolerance=10):
    global Transition_states
    global current_machine_state
    next_state = Transition_states[cur_state]

    # 492 x 369

    # print("extruder height: ",extruder_left_y)
    # print("extruder center: ", extruder_center_x)
    # print("extruder left: ", extruder_left_x)
    # print("extruder right: ", extruder_right_x)
    # print("Buildplate Top: ", buildplate_top_y)
    # print("Axis left: ", axis_left_x)

    # Extruder at far-left: extruder_left_y < 260 and > 250 (height); extruder_left_x < 150 and > 140; right_x < 265 and > 255
    #

    if current_machine_state == "Initialized":
        if not (extruder_left_y < 260 and extruder_left_y > 250 and extruder_left_x < 150 and extruder_left_x > 140 and \
                extruder_right_x > 255 and extruder_right_x < 265):
            # print("ERROR: PLEASE INITIALIZE PRINTER")
            current_machine_state = "ERROR: PRINTER NOT IN INITIALIZED POSITION: Extruder Not Detected / Initialized"
        # if not (buildplate_top_y >= 300):
        #    current_machine_state = "ERROR: PRINTER NOT IN INITIALIZED POSITION: BuildPlate"

    if next_state == "Initialized":
        return (extruder_left_y < 260 and extruder_left_y > 250 and extruder_left_x < 150 and extruder_left_x > 140 and \
                extruder_right_x > 255 and extruder_right_x < 265)

    if next_state == "Testing":
        return buildplate_top_y < 300 and buildplate_top_y != 0

    if next_state == "Calibration":
        return axis_left_x > 470

    if next_state == "Heating":
        return (axis_left_x < 465 and extruder_left_x < 150)

    if next_state == "Printing":
        return axis_left_x > 470

    if next_state == "Ending":
        return (extruder_left_x < 150 and buildplate_top_y == 0 and axis_left_x < 465)

    return False


def printer_predict_image():
    global Short_history
    global machine_states_record
    global current_machine_state
    counter = 1
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    while True:

        frame = cv2.imread(f"systematic_object_detection/camera{counter}.jpg", flags=cv2.IMREAD_COLOR)

        if frame is None:
            counter += 1
            continue

        img = transforms.ToTensor()(frame)
        img = resize(img, 416)
        img = torch.unsqueeze(img, 0)

        classes = load_classes(opt.class_path)  # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        # Configure input
        input_imgs = Variable(img.type(Tensor))

        # Get detections
        with torch.no_grad():
            # print(input_imgs)
            detections = printer_model(input_imgs)
            # print(detections)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Save image and detections
        imgs.extend("./")
        img_detections.extend(detections)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        tolerance = 10  # Tolerance for calibration (unit: pixel)

        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            # Create plot
            img = frame  # >>>>>> 480*640 shape

            axis_x, axis_y, buildplate_bottom_x, buildplate_bottom_y, extruder_left_x, extruder_left_y, axis_left_x = 0, 0, 0, 0, 0, 0, 0
            extruder_right_x, extruder_right_y, buildplate_top_y = 0, 0, 0
            extruder_center_x = 0

            machine_state = ""  # Current machine state

            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                    if classes[int(cls_pred)] == "axis" and y2 < 200 and cls_conf > 0.95:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 150, 2)
                        cv2.putText(frame, classes[int(cls_pred)], (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 1.5,
                                    (255, 255, 255), 2)

                        if x1 > 300:
                            axis_left_x = x1


                    else:
                        if classes[int(cls_pred)] != "axis" and cls_conf > 0.95:

                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 150, 2)
                            cv2.putText(frame, classes[int(cls_pred)], (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 1.5,
                                        (255, 255, 255), 2)

                            if classes[int(cls_pred)] == "extruder":
                                extruder_left_x = x1
                                extruder_left_y = y2

                                extruder_right_x = x2
                                extruder_right_y = y2

                                extruder_center_x = 0.5 * (x1 + x2)

                            if classes[int(cls_pred)] == "buildplate":
                                buildplate_top_y = y1

                Short_history.put(
                    transition_criterion(axis_left_x, extruder_left_x, extruder_left_y, extruder_right_x,
                                         extruder_right_y,
                                         extruder_center_x,
                                         buildplate_top_y, current_machine_state))

                if Short_history.sum() > 2:
                    current_machine_state = Transition_states[current_machine_state]
                    Short_history.initialize()
        #            print("Current Machine State: ", current_machine_state)

        print("extruder height: ",extruder_left_y)
        print("extruder center: ", extruder_center_x)
        print("extruder left: ", extruder_left_x)
        print("extruder right: ", extruder_right_x)
        print("Buildplate Top: ", buildplate_top_y)
        print("Axis left: ", axis_left_x)



        print(f"[Current Machine State {counter}]: ", current_machine_state)
        cv2.imshow('image', img)
        cv2.waitKey(0)

        counter += 1



if __name__ == "__main__":

    print("Object detection testing...")

    # Printer Interior

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3_ckpt_99.pth", help="path to weights file")
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

    printer_predict_image()


    print("All Finished.")
from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2
import io

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="yolov3_ckpt_40.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))


    # state_dict = model.state_dict()
    # for k,v in state_dict.items():
    #     print(k)

    model.eval()  # Set in evaluation mode

    cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)  # 0 for 1st webcam
    _, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # cv2.imshow("video", frame)
    # cv2.waitKey(0)

    # Change frame to Tensor

    img = transforms.ToTensor()(frame)
    img = resize(img, 416)
    img = torch.unsqueeze(img,0)


    # print(img.shape)
    #
    # blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)


    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")

    # for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    #     # Configure input
    #     input_imgs = Variable(input_imgs.type(Tensor))
    #
    #     # Get detections
    #     with torch.no_grad():
    #         # print(input_imgs)
    #         detections = model(input_imgs)
    #         # print(detections)
    #         detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
    #
    #     # Save image and detections
    #     imgs.extend(img_paths)
    #     img_detections.extend(detections)


        # Configure input
    input_imgs = Variable(img.type(Tensor))

    # Get detections
    with torch.no_grad():
        # print(input_imgs)
        detections = model(input_imgs)
        # print(detections)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)


    # Save image and detections
    imgs.extend("./")
    img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]


    print("\nSaving images:")



    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = frame
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        count = 0

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

               if cls_conf > 0.93:



                   cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 150, 2)
                   cv2.putText(frame, classes[int(cls_pred)], (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)


        # Save generated image with detections
        # plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        #
        # buf = io.BytesIO()
        # plt.savefig(buf, format="png", dpi=100)
        # buf.seek(0)
        # img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        # buf.close()
        #
        # img = cv2.imdecode(img_arr, 1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # plt.show()
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1000000)


        # plt.show()









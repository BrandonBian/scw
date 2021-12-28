"""Importing Section"""

from __future__ import division

from sklearn.cluster import KMeans
import string
import torch.backends.cudnn as cudnn
from datetime import datetime
import craft_utils
import imgproc
from craft import CRAFT
from color_utils import CTCLabelConverter, AttnLabelConverter
from dataset import ResizeNormalize
from model import Model
import json

from models import *
from utils.utils import *
from utils.datasets import *
from utils.smartmeter_modbus import *
from utils.globals import *

import argparse
from PIL import Image
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from flask import Flask, render_template, Response
import cv2
import serial
import time
import imagezmq
import threading
from queue import Queue
import pymongo

app = Flask(__name__)

def read_data():  # Save printer mode to a txt file on desktop

    global STRGLO, BOOL, StrTemp, buffer1

    ser = serial.Serial("COM3", 38400, timeout=None)

    if ser.is_open:

        write_port(ser, "ss\n")

        while True:
            if ser.in_waiting:
                STRGLO = ser.read(ser.in_waiting).decode("utf-8")

                file = r'C:\Users\lsam\Desktop\data_t.txt'
                with open(file, 'a+') as f:
                    f.write(STRGLO)
                StrTemp += STRGLO

                x = StrTemp.split("\n")

                if len(buffer1) > 1:
                    buffer1.pop(0)
                    buffer1.append(x[-1])



def open_port(portx, bps, timeout):
    ret = False
    flag = 0
    threading.Thread(target=read_data, args=()).start()


def write_port(ser, text):
    result = ser.write(text.encode("utf-8"))
    return result


def read_port():
    global STRGLO
    str = STRGLO
    STRGLO = ""
    return str


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/', methods=['GET', 'POST'])
def meter_readings():
    global global_predict_string
    global current_machine_state

    try:
        vars = generateData()
    except:
        print("Fail to get Smart Meter Reading.")
        vars = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  # 18 elements
    vars.append(global_predict_string)  # String predictions (19th)
    vars.append(global_finger_string)  # Finger predictions (20th)
    vars.append(current_machine_state)  # Machine state predictions (21th)

    this_time = [datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]]


    db.energydata.insert_one({"time": this_time,
                              "A_acc_energy": vars[0], "B_acc_energy": vars[1], "E_acc_energy": vars[2],
                              "F_acc_energy": vars[3], "J_acc_energy": vars[4], "N_acc_energy": vars[5],
                              "A_power": vars[6], "B_power": vars[7], "E_power": vars[8],
                              "F_power": vars[9], "J_power": vars[10], "N_power": vars[11],
                              "A_current": vars[12], "B_current": [13], "E_current": vars[14],
                              "F_current": vars[15], "J_current": vars[16], "N_current": vars[17]})

    return (json.dumps(vars))


def worker_predict_image():
    ret, frame = worker_camera.read()

    # cv2.imwrite('frame.jpg', frame)
    # cv2.waitKey(1)

    if ret == True:
        _, this_frame = cv2.imencode('.jpg', frame)
        w, h, _ = frame.shape
        ee = np.zeros(frame.shape)

        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(imgray, 200, 255, 0)
        edges = cv2.Canny(imgray, 150, 210, L2gradient=True)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        closed_contours = []
        for n, i in enumerate(contours):
            if cv2.contourArea(i) > cv2.arcLength(i, True) and cv2.contourArea(i) > (
                    w / 1080) ** 2 * 15000 and n % 2 == 0 and cv2.contourArea(i) < 10000:
                closed_contours.append(i)

        # Filter other bbox.
        width_list = np.asarray([np.max(i[:, :, 0]) - np.min(i[:, :, 0]) for i in closed_contours]).reshape(-1,
                                                                                                            1)
        final_predict = ""
        finger_predict = ""

        if len(width_list) != 0:
            if len(width_list) == 1:
                kmeans = KMeans(n_clusters=1, random_state=0).fit(width_list)
            else:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(width_list)

            kmeans_labels = kmeans.labels_.tolist()
            flag = None

            if kmeans_labels.count(0) > 1: flag = 0
            if kmeans_labels.count(1) > 1: flag = 1

            if kmeans_labels.count(0) > kmeans_labels.count(1):
                max_count = kmeans_labels.count(0)
            else:
                max_count = kmeans_labels.count(1)

            idx = np.where(kmeans.labels_ == flag)[0].tolist()
            filtered_contours = [con for i, con in enumerate(closed_contours) if i in idx]
            # print('**NEW FRAME********************************************************')

            finger_pos = -1

            # for con_id, con in enumerate(filtered_contours):
            for con_id, con in reversed(list(enumerate(reversed(filtered_contours)))):

                button_num = con_id + 1

                # Every screen
                xmin = np.min(con[:, :, 0])
                xmax = np.max(con[:, :, 0])
                ymin = np.min(con[:, :, 1])
                ymax = np.max(con[:, :, 1])

                button_xmin = int(xmin - 0.3 * (xmax - xmin))
                button_xmax = int(xmax - 1.16 * (xmax - xmin))
                button_ymin = int(ymin + 0.15 * (ymax - ymin))
                button_ymax = int(ymax - 0.11 * (ymax - ymin))

                # The center pixel of the button box
                center_x = button_xmin + int(0.5 * (button_xmax - button_xmin))
                center_y = button_ymin + int(0.5 * (button_ymax - button_ymin))

                # The center pixel of the button box (upper to this button box)
                upper_center_x = center_x
                upper_center_y = center_y - int(2.1 * (button_ymax - button_ymin))

                button = False
                # print("ID is:", con_id)
                if button_num != 0:

                    # print("Analying button No.", button_num)

                    my_center_R = frame[center_y, center_x, 2]
                    up_center_R = frame[upper_center_y, upper_center_x, 2]

                    if my_center_R > up_center_R:
                        diff = my_center_R - up_center_R
                    else:
                        diff = up_center_R - my_center_R

                    if diff > 40:
                        button = True
                        finger_pos = button_num  # Finger is on this button
                    else:
                        button = False

                cv2.rectangle(frame, (button_xmin, button_ymin), (button_xmax, button_ymax), (0, 255, 0), 2)

                region = np.array(frame[ymin - 5:ymax + 5, xmin - 5:xmax + 5])
                bboxes_text, polys_text, score_text = test_net(net, region,
                                                               args.text_threshold, args.link_threshold,
                                                               args.low_text, args.cuda, args.poly, refine_net)

                if len(bboxes_text) != 0:
                    image_tensors = []
                    transform = ResizeNormalize((args.imgW, args.imgH))
                    for bbox in bboxes_text:
                        xxmin = int(np.min(bbox[:, 0]))
                        xxmax = int(np.max(bbox[:, 0]))
                        yymin = int(np.min(bbox[:, 1]))
                        yymax = int(np.max(bbox[:, 1]))

                        if xxmin < 0:
                            xxmin = 0
                        if xxmax < 0:
                            xxmax = 0
                        if yymin < 0:
                            yymin = 0
                        if yymax < 0:
                            yymax = 0

                        roi = np.array(region[yymin:yymax, xxmin:xxmax])
                        roi_pil = Image.fromarray(roi).convert('L')
                        image_tensors.append(transform(roi_pil))

                    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
                    predict_list = demo(opt=args, roi=image_tensors, button=button)

                    # Special Cases

                    predict_string = ' '.join(predict_list)

                    if predict_string == "model load":
                        predict_string = "load model"

                    # Find the corresponding string item based on the predicted string

                    if button_num == 1:
                        scores = []
                        for item in words1:
                            scores.append(similar(item, predict_string))

                        predict_string = words1[scores.index(max(scores))]

                    if button_num == 2:
                        scores = []
                        for item in words2:
                            scores.append(similar(item, predict_string))

                        predict_string = words2[scores.index(max(scores))]

                    if button_num == 3:
                        scores = []
                        for item in words3:
                            scores.append(similar(item, predict_string))

                        predict_string = words3[scores.index(max(scores))]

                    if button_num == 4:
                        scores = []
                        for item in words4:
                            scores.append(similar(item, predict_string))

                        predict_string = words4[scores.index(max(scores))]

                    final_predict = final_predict + ";" + predict_string

                    for bbox in bboxes_text:
                        bbox[:, 0] = bbox[:, 0] + xmin
                        bbox[:, 1] = bbox[:, 1] + ymin

                        poly = np.array(bbox).astype(np.int32).reshape((-1))
                        poly = poly.reshape(-1, 2)
                        cv2.polylines(frame, [poly.reshape((-1, 1, 2))], True, (0, 0, 255), 2)

                else:

                    final_predict = final_predict + ";" + "EMPTY STRING"

            grid = cv2.drawContours(frame, filtered_contours, -1, (0, 255, 0), 3)
            _, jpeg = cv2.imencode('.jpg', grid)

            return jpeg.tobytes(), final_predict, finger_pos


def webcam_get_frame():
    success, image = web_camera.read()
    ret, jpeg = cv2.imencode('.jpg', image)
    return jpeg.tobytes()

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

    ret, frame = printer_camera.read()  #


    if ret != True:
        print("Error getting printer interior output")

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


    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        # Create plot
        img = frame  # >>>>>> 480*640 shape

        axis_x, axis_y, buildplate_bottom_x, buildplate_bottom_y, extruder_left_x, extruder_left_y, axis_left_x = 0, 0, 0, 0, 0, 0, 0
        extruder_right_x, extruder_right_y, buildplate_top_y = 0, 0, 0
        extruder_center_x = 0


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
                transition_criterion(axis_left_x, extruder_left_x, extruder_left_y, extruder_right_x, extruder_right_y,
                                     extruder_center_x,
                                     buildplate_top_y, current_machine_state))

            if Short_history.sum() > 2:
                current_machine_state = Transition_states[current_machine_state]
                Short_history.initialize()

    suc, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()


def gen_webcam():
    while True:
        frame = webcam_get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_panel():
    while True:
        _, frame = worker_camera.read()
        suc, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_printer():
    while True:
        _, frame = printer_camera.read()
        suc, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def predict_panel():
    global global_predict_string
    global global_finger_string
    while True:
        frame, final_predict, finger_pos = worker_predict_image()
        global_predict_string = final_predict[1:]
        global_finger_string = str(finger_pos)
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def predict_printer():
    while True:
        frame = printer_predict_image()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# Assembly Webcam
def gen_assem():  # Get the real time image of the assembly webcam (b1)

    while True:
        print("Running Assembly Code")

        try:
            image_hub = imagezmq.ImageHub(open_port='tcp://*:5551', REQ_REP=True)

        except:
            print("Failed to get Image Hub.")

        cam_name, frame = image_hub.recv_image()
        (flag, encodedImage) = cv2.imencode('.jpg', frame)
        image_hub.send_reply(b'OK')
        print("frame: ", frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
        image_hub.close()

        time.sleep(1)


# Assembly Replay
def gen_replay():
    blank = 230 * np.ones((480, 640, 3), dtype=np.int8)
    (flag, encodedImage) = cv2.imencode('.jpg', blank)
    global buffer
    while True:
        if trigger_list.qsize() != 0:
            if len(buffer) < 120: continue
            trig = trigger_list.get()
            get_replay = buffer[-120:]
            print("get replay", len(get_replay))
            for frame in get_replay:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(frame) + b'\r\n')
                time.sleep(0.033)


def gen_main():
    global buffer1, col
    while True:
        for xx in buffer1:
            if xx.find("Current Position") > -1:
                print('main door open')
                col = (225, 0, 0) * np.ones((25, 75, 3), dtype=np.int8)
                col = cv2.imencode('.jpg', col)[1].tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       col + b'\r\n')

            else:
                col = 230 * np.ones((25, 75, 3), dtype=np.int8)
                col = cv2.imencode('.jpg', col)[1].tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       col + b'\r\n')
        time.sleep(1)


def gen_model():
    global buffer1, col
    while True:
        for xx in buffer1:
            if xx.find("curve count") < 0:
                print('model door open')
                col = (0, 255, 0) * np.ones((25, 75, 3), dtype=np.int8)
                col = cv2.imencode('.jpg', col)[1].tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       col + b'\r\n')
            else:
                col = 230 * np.ones((25, 75, 3), dtype=np.int8)
                col = cv2.imencode('.jpg', col)[1].tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       col + b'\r\n')
        time.sleep(1)


def gen_support():
    global buffer1, col

    while True:
        for xx in buffer1:
            if xx.find("Current Position") == -1:
                print('support door open')
                col = (0, 0, 255) * np.ones((25, 75, 3), dtype=np.int8)
                col = cv2.imencode('.jpg', col)[1].tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       col + b'\r\n')
            else:
                print('support door open another condition')
                col = 230 * np.ones((25, 75, 3), dtype=np.int8)
                col = cv2.imencode('.jpg', col)[1].tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       col + b'\r\n')
        time.sleep(1)


#
@app.route('/video_feed_worker')  ## Panel original
def video_feed_worker():
    return Response(gen_panel(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/hand_recognition')  ## Panel predictions
def hand_recognition():
    return Response(predict_panel(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_webcam')  #
def video_feed_webcam():
    return Response(gen_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_printer')  # Printer interior original
def video_feed_printer():
    return Response(gen_printer(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_uprint')  # Printer interior predictions
def video_feed_uprint():
    return Response(predict_printer(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_assemblywebcam')  # Mina
def video_feed_assemblywebcam():
    return Response(gen_assem(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/replay_test')  # Mina
def replay_test():
    return Response(gen_replay(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/main_door')  # Mina
def main_door():
    return Response(gen_main(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/model_side')  # Mina
def model_side():
    return Response(gen_model(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/support_side')  # Mina
def support_side():
    return Response(gen_support(), mimetype='multipart/x-mixed-replace; boundary=frame')


###################################################################################################################################


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    # if args.show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def demo(opt, roi, button=False):
    predict_list = []
    with torch.no_grad():
        batch_size = roi.size(0)
        image = roi.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index, preds_size)

        else:
            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        # log = open(f'./log_demo_result.txt', 'a')
        dashed_line = '-' * 80

        if button:
            head = f'{"predicted_labels":25s}\tconfidence score\tFinger On Button: TRUE'
        else:
            head = f'{"predicted_labels":25s}\tconfidence score\tFinger On Button: FALSE'

        # print(f'{dashed_line}\n{head}\n{dashed_line}')
        # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            # print(f'\t{pred:25s}\t{confidence_score:0.4f}')

            predict_list.append(pred)
        return (predict_list)


if __name__ == '__main__':

    print("Webpage Program Loading...")

    trigger = 0
    trigger_list = Queue()
    host_ip = "130.166.41.31"
    port = 6666
    buffer = []
    buffer1 = []
    col = np.ones((25, 75, 3), dtype=np.int8)

    streamingP = threading.Thread(target=app.run, args=(),
                                  kwargs={"host": 'localhost', "port": 8000, "debug": False,
                                          "threaded": True, "use_reloader": False})
    streamingP.start()
    open_port("COM3", 38400, None)

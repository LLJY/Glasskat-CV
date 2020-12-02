import numpy as np
import argparse
import cv2
import time
import math
import json

MAX_WIDTH = 800
MAX_HEIGHT = 448
parser = argparse.ArgumentParser(description="Classify Cat using image from webcam with mobilenet SSD")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt")
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                    help='Path to weights: ')
parser.add_argument("--thr", default=0.3, type=float, help="confidence threshold to filter out weak detections")
parser.add_argument("--cameraConf", default="c920conf", type=str, help="simple python file containing the "
                                                                       "configuration for your webcam")
parser.add_argument("--cuda", default=True, type=bool, help="use cuda or not")
parser.add_argument("--model")
args = parser.parse_args()
camConfig = __import__(args.cameraConf)
# got the default labels of the mobilenet SSD model we are using
classNames = {0: 'background',
              1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
              10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
relevantClassNames = {0: 'background',
                      5: 'bottle', 8: 'cat', 15: 'person'}


def calculate_x_coords(u):
    max_coords = (zCoords / math.tan(math.radians(camConfig.fov / 2))) * 2
    # calculate the camera coordinates and convert it to world coordinates by dividing by max/2
    # so the center of the frame is now 0.
    return round((u / MAX_WIDTH) * max_coords, 2) - (max_coords / 2)


# load the video from our webcam
gstr = "v4l2src device=/dev/video0 ! video/x-raw,framerate=30/1,width=" + str(MAX_WIDTH) + ",height=" + str(
    MAX_HEIGHT) + " ! videoconvert ! appsink"
cap = cv2.VideoCapture(gstr, cv2.CAP_GSTREAMER)
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
if (args.cuda):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
while True:
    start = time.time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_resized = cv2.resize(frame, (300, 300))  # resize frame for prediction

    # resize our images for mobilenet (300x300) and normalize the input with 127.5
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()

    # Size of frame resize (300x300)
    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]
    # detections returns a numpy array, access the information in the multi dimensional array like this
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Confidence of prediction
        if confidence > args.thr:  # Filter prediction
            class_id = int(detections[0, 0, i, 1])  # Class label

            # Object location
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            # scale our detection x, y, z range to the real image from our 300x300 detection
            # Scale object detection to frame
            heightFactor = frame.shape[0] / 300.0
            widthFactor = frame.shape[1] / 300.0
            xLeftBottom = int(widthFactor * xLeftBottom)
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop = int(widthFactor * xRightTop)
            yRightTop = int(heightFactor * yRightTop)
            # calculate the coordinates of the object byu taking the center
            midXCoords = (xLeftBottom + xRightTop) / 2
            midYCoords = (yLeftBottom + yRightTop) / 2

            # label the rectangle
            if (class_id in relevantClassNames):
                # draw a rectangle according to the bounding box that surrounds the object and only draw when
                # something is detected
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))
                # calculate the Z coordinates using a proportional equation using the height of the object
                zCoords = camConfig.z_gradient * (yRightTop - yLeftBottom) + camConfig.bias
                # calculate the x coordinates
                xCoords = calculate_x_coords(midXCoords)
                label = relevantClassNames[class_id] + " X:" + str(xCoords) + " Y:" + str(midYCoords) + " Z:" + str(
                    zCoords)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
                # add label for item
                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                              (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    # add a label for FPS
    fpsLabel = str(int(1 / (time.time() - start))) + " FPS"
    fpsLabelSize, fpsBaseLine = cv2.getTextSize(fpsLabel, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
    cv2.rectangle(frame, (0, 10 - fpsLabelSize[1]),
                  (0 + fpsLabelSize[0], 10 + fpsBaseLine),
                  (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, fpsLabel, (0, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    cv2.namedWindow("frame", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC
        break

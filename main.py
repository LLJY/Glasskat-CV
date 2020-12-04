import numpy as np
import grpc
import argparse
import cv2
import time
import math
import json
import firebase_admin
from firebase_admin import storage, credentials
import threading
from ringbuffer import RingBuffer
from datetime import datetime
import protos.gk_pb2
import protos.gk_pb2_grpc

parser = argparse.ArgumentParser(description="Classify Cat using image from webcam with mobilenet SSD")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt")
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                    help='Path to weights: ')
parser.add_argument("--thr", default=0.3, type=float, help="confidence threshold to filter out weak detections")
parser.add_argument("--cameraConf", default="c920conf", type=str, help="simple python file containing the "
                                                                       "configuration for your webcam")
parser.add_argument("--cuda", default=True, type=bool, help="use cuda or not")
parser.add_argument("--model")
parser.add_argument("--width", default=800, help="video width")
parser.add_argument("--height", default=448, help="video height")
parser.add_argument("--fps", default=30, help="video framerate")
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
is_video_saving = False
print("[INFO]: Initializing Firebase Admin")
cred = credentials.Certificate("firebase-credentials.json")
app = firebase_admin.initialize_app(cred, {'storageBucket': 'mcsv-firebase-service.appspot.com'})
bucket = storage.bucket("mcsv-firebase-service.appspot.com")

print("[INFO]: Initializing gRPC...")
channel = grpc.insecure_channel("localhost:5001")
stub = protos.gk_pb2_grpc.GkCoordinatesStub(channel)

def calculate_x_coords(u):
    max_coords = (zCoords / math.tan(math.radians(camConfig.fov / 2))) * 2
    # calculate the camera coordinates and convert it to world coordinates by dividing by max/2
    # so the center of the frame is now 0.
    return round((u / args.width) * max_coords, 2) - (max_coords / 2)


def save_video(frame):
    global is_video_saving
    is_video_saving = True
    # only save data to the ring buffer if it is full, this acts as a five minute timeout between videos
    if (ring_buffer.isFull):
        print("[INFO] Saving video...")
        dateStr = datetime.now().isoformat()
        # passing this over to the upload function to define the filename
        videoName = dateStr + ".mp4"
        # to output to the local videos/ folder
        videoPath = "videos/" + videoName
        out = cv2.VideoWriter(videoPath, cv2.VideoWriter.fourcc(*'mp4v'), args.fps, (int(args.width), int(args.height)))
        for idx, frame in enumerate(ring_buffer.data):
            out.write(ring_buffer.next(idx))
        out.release()
        # flush the ring buffer
        ring_buffer.flush()
        # run the uploading of video in a separate thread!
        thread = threading.Thread(target=upload_video, args=(videoName, videoPath))
        thread.start()
    else:
        ring_buffer.append(frame)
    is_video_saving = False

def upload_video(filename, file_location):
    blob = bucket.blob(filename)
    blob.upload_from_filename(file_location)
    blob.make_public()
    print("Uploaded video URL: "+ blob.public_url)


# load the video from our webcam
print("[INFO]: Setting up camera")
gstr = "v4l2src device=/dev/video0 ! video/x-raw,framerate=30/1,width=" + str(args.width) + ",height=" + str(
    args.height) + " ! videoconvert ! appsink"
cap = cv2.VideoCapture(gstr, cv2.CAP_GSTREAMER)
# cap = cv2.VideoCapture(0)
print("[INFO]: Setting up Deep Learning Model")
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
if (args.cuda):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("[INFO] Initializing Ring Buffer")
# Initialize the new ring buffer to save the last 5 minutes of footage
ring_buffer = RingBuffer(max_size=300)

# start the image capture
print("[INFO] Starting capture...")

# add a boolean here to avoid saving the video multiple times!
while True:
    start = time.time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_resized = cv2.resize(frame, (300, 300))  # resize frame for prediction
    # get vcap property
    args.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    args.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    # resize our images for mobilenet (300x300) and normalize the input with 127.5
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()

    # Size of frame resize (300x300)
    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]
    isCapture = False
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

            # we are separating the logic as we want an accurate FPS readout while keeping all labels
            # in the final video frame we are saving to our ring buffer
            isCapture = class_id in relevantClassNames
            # label the rectangle
            if isCapture:
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

    if isCapture and not is_video_saving:
        # run asynchronously to avoid stalling the video
        thread = threading.Thread(target=save_video, args=(frame,))
        thread.start()

    if cv2.waitKey(1) >= 0:  # Break with ESC
        print("[INFO] Stopping capture!")
        break

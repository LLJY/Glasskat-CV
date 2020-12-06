import numpy as np
import grpc
import argparse
import cv2
import time
import math
import json
import firebase_admin
from firebase_admin import storage, credentials, messaging
import threading
from ringbuffer import RingBuffer
from datetime import datetime
import protos.gk_pb2
import protos.gk_pb2_grpc

parser = argparse.ArgumentParser(description="Classify Cat using image from webcam with mobilenet SSD")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt")
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                    help='Path to weights: ')
parser.add_argument("--use-vgg", default=False, type=bool, help="use vggnet instead of mobilenet")
parser.add_argument("--thr", default=0.3, type=float, help="confidence threshold to filter out weak detections")
parser.add_argument("--camera-conf", default="c920conf", type=str, help="simple python file containing the "
                                                                        "configuration for your webcam")
parser.add_argument("--cuda", default=True, type=bool, help="use cuda or not")
parser.add_argument("--width", default=800, help="video width")
parser.add_argument("--height", default=448, help="video height")
parser.add_argument("--fps", default=30, help="video framerate")
parser.add_argument("--use-gstreamer", default=True, help="use gstreamer to get our video content. REQUIRED ON CUDA")
parser.add_argument("--cat-only", default=True, help="set relevant classes to cat only")
args = parser.parse_args()
camConfig = __import__(args.camera_conf)
# got the default labels of the mobilenet SSD model we are using
classNames = {0: 'background',
              1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
              10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
relevantClassNames = {0: 'background',
                      5: 'bottle', 8: 'cat', 15: 'person'}
if args.cat_only:
    relevantClassNames = {8: 'cat'}
# if we are using vgg, set relevant classes to all
if(args.use_vgg):
    relevantClassNames = {
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush'
    }
# initialize some dummy variables for global variables that will be setup later
# not the best practice, but it should be fine for our use case
"""Sets up miscellaneous functions like firebase and gRPC before everything as everything depends on them."""
print("[INFO]: Initializing Firebase Admin")
cred = credentials.Certificate("firebase-credentials.json")
app = firebase_admin.initialize_app(cred, {'storageBucket': 'mcsv-firebase-service.appspot.com'})
bucket = storage.bucket("mcsv-firebase-service.appspot.com")

print("[INFO]: Initializing gRPC...")
credentials = grpc.ssl_channel_credentials()
channel = grpc.insecure_channel("localhost:5002")
stub = protos.gk_pb2_grpc.GkCoordinatesStub(channel)
# whether or not to save the current ring buffer of videos or flush it
save_buffer = False
# lets the function know whether the video is already saving to avoid saving it multiple times when the ring buffer
# is full
is_video_saving = False


def calculate_x_coords(u, zCoords):
    max_coords = (zCoords / math.tan(math.radians(camConfig.fov / 2))) * 2
    # round x to 2 decimal places
    return round(((u / args.width) * max_coords), 2)


def save_frame(frame):
    ring_buffer.append(frame)


def save_video():
    global is_video_saving
    global save_buffer
    is_video_saving = True
    # only save data to the ring buffer if it is full, this acts as a five minute timeout between videos
    print("[INFO]: Saving video...")
    date_string = datetime.now().isoformat()
    # passing this over to the upload function to define the filename
    title = date_string + ".mp4"
    # to output to the local videos/ folder
    file_path = "videos/" + title
    out = cv2.VideoWriter(file_path, cv2.VideoWriter.fourcc(*'avc1'), args.fps, (int(args.width), int(args.height)))
    for idx, frame in enumerate(ring_buffer.data):
        out.write(ring_buffer.next(idx))
    out.release()
    # flush the ring buffer
    ring_buffer.flush()
    # run the uploading of video in a separate thread!
    thread = threading.Thread(target=upload_video, args=(title, file_path))
    thread.start()
    is_video_saving = False
    save_buffer = False


def upload_video(filename, file_location):
    """Uploads video to firebase storage and sends notification afterwards"""
    blob = bucket.blob(filename)
    blob.upload_from_filename(file_location)
    print("Uploaded video URL: " + blob.public_url)
    # hardcode the notification token, we will use topics in the real app, topics are not supported on angular
    message = messaging.Message(notification=messaging.Notification(title="Cat Detected!!", body="Cat Detected, you might want to check your videos for an available video"), token="f6AbCj_Dxd1ksC7AT2RucG:APA91bFH7aMji5NfwSNA-X8v3ZlmJBP8HxWbBT_nmRCqha2zrjHW9tw37WwqkaSLnTQRJGi9HOTcCl4jD_ZEPPLNctOKQ4dE0figVKT2sb3w5Za3bmjb6bH8j8tW8wx8BFfB60w39FcT")
    messaging.send(message)
    print("notification sent!")


def start_coordinates_stream(cap, net):
    """starts the main camera capture and streams data from it to the server"""
    thread = threading.Thread(target=stub.RequestCoord, args=(start_capture(cap, net),), kwargs={'timeout': 30000000})
    thread.start()


# load the video from our webcam
def setup_opencv(use_gstreamer):
    print("[INFO]: Setting up opencv")
    if use_gstreamer:
        gstr = "v4l2src device=/dev/video0 ! video/x-raw,framerate=30/1,width=" + str(args.width) + ",height=" + str(
            args.height) + " ! videoconvert ! appsink"
        return cv2.VideoCapture(gstr, cv2.CAP_GSTREAMER)
    # just get capture from /dev/video0 if we are not using gstreamer
    return cv2.VideoCapture(0)


def setup_dnn(use_cuda):
    print("[INFO]: Setting up Deep Learning Model")
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
    if(args.use_vgg):
        net = cv2.dnn.readNetFromCaffe("vgg-deploy.prototxt","vgg.caffemodel")
    if use_cuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


def start_capture(cap, net):
    """main function that starts opencv and the classification
    yields our protobuf coordinate object to send a message to the server"""
    print("[INFO] Starting capture...")
    global save_buffer
    while True:
        start = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_resized = cv2.resize(frame, (300, 300))  # resize frame for prediction
        # get cap property
        args.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        args.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        # resize our images for mobilenet (300x300) and normalize the input with 127.5
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        net.setInput(blob)
        detections = net.forward()

        # Size of frame resize (300x300)
        cols = frame_resized.shape[1]
        rows = frame_resized.shape[0]

        # just to shut up some accessed before assignment warnings
        isCapture = False
        xCoords = 0
        zCoords = 0
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

                # if the detection is what we are looking for, label it.
                if class_id in relevantClassNames:
                    # set the global save_buffer to True if we have detected any relevant objects
                    save_buffer = True
                    # draw a rectangle according to the bounding box that surrounds the object and only draw when
                    # something is detected
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                                  (0, 255, 0))
                    # calculate the Z coordinates using a proportional equation using the height of the object
                    zCoords = round(camConfig.z_gradient * (yRightTop - yLeftBottom) + camConfig.bias, 2)
                    # calculate the x coordinates
                    xCoords = calculate_x_coords(midXCoords, zCoords)
                    label = relevantClassNames[class_id] + " X:" + str(xCoords) + " Z:" + str(
                        zCoords)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
                    # add label for detected item
                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                  (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                  (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    # yield a value so gRPC can save the current position of whatever is detected
                    yield protos.gk_pb2.RequestModel(X=xCoords, Y=zCoords)

        # add a label for FPS
        fpsLabel = str(int(1 / (time.time() - start))) + " FPS"
        fpsLabelSize, fpsBaseLine = cv2.getTextSize(fpsLabel, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
        cv2.rectangle(frame, (0, 10 - fpsLabelSize[1]),
                      (0 + fpsLabelSize[0], 10 + fpsBaseLine),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, fpsLabel, (0, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

        cv2.namedWindow("frame", cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("frame", frame)

        save_frame(frame)
        # if the ringbuffer is full and we decide to save the current video
        if ring_buffer.isFull and save_buffer and not is_video_saving:
            # run asynchronously to avoid stalling the video
            thread = threading.Thread(target=save_video, args=())
            thread.start()

        if cv2.waitKey(1) >= 0:  # Break with ESC
            print("[INFO]: Stopping capture!")
            break


# main function
if __name__ == "__main__":
    print("[INFO] Initializing Ring Buffer")
    # Initialize the new ring buffer to save the last 10 seconds of footage
    ring_buffer = RingBuffer(max_size=300)
    dnn = setup_dnn(args.cuda)
    v_cap = setup_opencv(args.use_gstreamer)
    start_coordinates_stream(v_cap, dnn)

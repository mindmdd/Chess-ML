import cv2 as cv
import numpy as np
from collections import deque
import argparse
import matlab.engine

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    args = parser.parse_args()

    return args

class Matlab:
    engine = matlab.engine.start_matlab()
    engine.addpath("./matlab", nargout = 0)
    engine.addpath("./matlab/matching", nargout = 0)

class Camera:
    rt_landmark_array = [np.zeros((21, 3)),np.zeros((21, 3))]
    bounding = [0,0]
    landmark_array = [np.zeros((21, 3)),np.zeros((21, 3))]
    landmark_history_cam = deque(maxlen=50)
    bounding_history_cam = deque(maxlen=50)
    
    # Argument parsing ------------------------------------------------------
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # Camera preparation ----------------------------------------------------
    cap = cv.VideoCapture(cap_device, cv.CAP_DSHOW)
    if cap.isOpened():
        print("From Camera")
    else:
        print("No Camera")
        cap = cv.VideoCapture("./images/test.mp4")

    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    save_image = []

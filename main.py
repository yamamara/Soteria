import logging
import os
from logging.handlers import TimedRotatingFileHandler
#from ultralytics import YOLO

import cv2
def openWebCam():
    # Main camera variable
    capture = cv2.VideoCapture(0)

    # Initializes and formats logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logFormatter = logging.Formatter("[%(asctime)s] [%(levelname)s/%(name)s] %(message)s")

    if not os.path.exists("log"):
        os.makedirs("log")

    file_handler = TimedRotatingFileHandler("log/soteria.log", when="midnight", interval=1, backupCount=30)
    file_handler.suffix = "%m-%d-%Y"
    file_handler.setFormatter(logFormatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logFormatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.warning("helo")

    if not capture.isOpened():
        logger.error("Unable to open camera")

    while True:
        # Reads the current frame from the camera as a tuple
        frame_available, frame = capture.read()

        if not frame_available:
            break



        frame = cv2.flip(frame, 1)

        # Instantiates camera window and updates frame data
        cv2.imshow("Webcam", frame)

        # Captures keyboard input
        key = cv2.waitKey(1)

        # Closes window when "esc" key pressed
        if key == 27:
            break

    # Frees resources after window closed
    capture.release()
    cv2.destroyAllWindows()

def loadModel():




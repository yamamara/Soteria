import logging
import os
from logging.handlers import TimedRotatingFileHandler

import cv2

# Main camera hardware instance
capture = cv2.VideoCapture(0)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logFormatter = logging.Formatter("[%(asctime)s] [%(levelname)s/%(name)s] %(message)s")

if not os.path.exists("log"):
    os.makedirs("log")

# Makes new log file at midnight
file_handler = TimedRotatingFileHandler("log/soteria.log", when="midnight", interval=1, backupCount=30)
file_handler.suffix = "%m-%d-%Y"
file_handler.setFormatter(logFormatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logFormatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

if not capture.isOpened():
    logger.error("Unable to open camera")

while True:
    frame_available, frame = capture.read()

    if not frame_available:
        break

    cv2.imshow("Webcam", frame)

    # Captures keyboard input
    key = cv2.waitKey(1)

    # Closes window when "esc" key pressed
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()

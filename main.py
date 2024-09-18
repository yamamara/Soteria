import logging
import os
from logging.handlers import TimedRotatingFileHandler

import cv2
from ultralytics import YOLO

# Main camera hardware instance
capture = cv2.VideoCapture(0)
model = YOLO("yolov10x.pt")

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


def predict(chosen_model, image, conf):
    results = chosen_model.predict(image, conf=conf)
    rectangle_thickness = 2
    text_thickness = 1

    # Draws labeled box on cv2 window for every prediction result
    for result in results:
        for box in result.boxes:
            cv2.rectangle(image, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(image, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)

    return image


if not capture.isOpened():
    logger.error("Unable to open camera")

while True:
    frame_available, frame = capture.read()

    if not frame_available:
        break

    processed_image = predict(model, frame, conf=0.5)
    cv2.imshow("Webcam", processed_image)

    # Captures keyboard input
    key = cv2.waitKey(1)

    # Closes window when "esc" key pressed
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()

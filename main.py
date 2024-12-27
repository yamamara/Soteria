import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler

import cv2
from ultralytics import YOLO

start_time = time.time()
capture = cv2.VideoCapture(0)
model = YOLO("yolo11x.pt")
logger = logging.getLogger(__name__)


def elapsed_time_with_unit():
    elapsed_time = time.time() - start_time

    if elapsed_time > 86400:
        return f"{int(elapsed_time / 86400)} day(s)"
    elif elapsed_time > 3600:
        return f"{int(elapsed_time / 3600)} hour(s)"
    elif elapsed_time > 60:
        return f"{int(elapsed_time / 60)} minute(s)"
    else:
        return f"{int(elapsed_time)} seconds"


def setup_logger():
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


def predict(chosen_model, image, conf):
    results = chosen_model.predict(image, conf=conf)
    rectangle_thickness = 2
    text_thickness = 1

    for result in results:
        for box in result.boxes:
            cv2.rectangle(
                image,
                (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                (255, 0, 0),
                rectangle_thickness
            )

            cv2.putText(
                image,
                f"{result.names[int(box.cls[0])]} - {(float(result.boxes.conf[0]) * 100):.2f}%",
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                text_thickness
            )

    return image


def main():
    setup_logger()
    stream_open = True

    if not capture.isOpened():
        logger.error("Unable to open camera")
        exit(1)

    while stream_open:
        frame_available, frame = capture.read()

        if not frame_available:
            break

        processed_image = predict(model, frame, conf=0.5)
        cv2.imshow("Webcam", processed_image)
        key = cv2.waitKey(1)

        if key == 27 or cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) == 0.0:
            stream_open = False

    capture.release()
    cv2.destroyAllWindows()
    # logger.info("Uptime: " + elapsed_time_with_unit())


if __name__ == "__main__":
    main()

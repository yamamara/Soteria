from ultralytics import YOLO
model = YOLO("yolo11x.pt")

model.train(
    model="yolo11x.pt",
    data="/home/artemis/Downloads/Gun Dataset/Gun with webcam views.v1i.yolov8/data.yaml",
    time=18,
    batch=-1,
    imgsz=640
)

model.export()

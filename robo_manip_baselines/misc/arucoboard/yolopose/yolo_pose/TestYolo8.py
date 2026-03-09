from ultralytics import YOLO

# Load a model
model = YOLO("yolov8-pose.yaml")  # build a new model from YAML
model = YOLO("yolov8m-pose.pt")    # load a pretrained model (recommended for training)
model = YOLO("yolov8-pose.yaml").load("yolov8m-pose.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)

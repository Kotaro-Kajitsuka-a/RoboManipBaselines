from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-pose.pt")  # load an official model
model = YOLO("runs/pose/train16/weights/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg", conf=0.01425, iou=0.35, end2end=False)  # predict on an image

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    print(f"xy: {xy}")
    kpts = result.keypoints.data  # x, y, visibility (if available)
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
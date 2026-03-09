from ultralytics import YOLO

# Load an official or custom model
#model = YOLO("yolo26n.pt")  # Load an official Detect model
#model = YOLO("yolo26n-seg.pt")  # Load an official Segment model
model = YOLO("yolo26n-pose.pt")  # Load an official Pose model

# Perform tracking with the model
#results = model.track("human_strech.mp4", show=True)  # Tracking with default tracker
results = model.track("human_strech.mp4", show=True, tracker="bytetrack.yaml")  # with ByteTrack

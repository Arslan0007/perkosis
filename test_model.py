from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/bib_detector_v17/weights/best.pt')

# Run inference on a test image (one the model hasn't seen before)
results = model('dataset/test.jpg')

# Show the results
# This will save an image with the detected bounding box drawn on it.
results[0].show()
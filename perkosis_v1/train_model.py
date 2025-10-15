from ultralytics import YOLO

# 1. Load a pre-trained YOLOv8 model (e.g., yolov8n - the 'nano' version, which is fast and small)
# Using a pre-trained model on the massive COCO dataset gives us a head-start.
model = YOLO('yolov8n.pt')

print("Starting training...")

# 2. Train the model
# Arguments:
#   data: path to your data.yaml file
#   epochs: number of training cycles (start small, increase if accuracy is low)
#   imgsz: the size to which images are resized before training (640 is standard)
#   batch: number of images processed per step (reduce if low on memory)
results = model.train(
    data='C:/dev/perkosis/data.yaml', 
    epochs=50, 
    imgsz=640, 
    batch=8, 
    name='bib_detector_v18'
)

print("Training finished! Model weights saved in runs/detect/bib_detector_v18/weights/best.pt")
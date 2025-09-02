from ultralytics import YOLO
import os
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = 'cuda'
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = 'cpu'
    print("No GPU detected, using CPU")

# Load the pretrained YOLOv11s model
model = YOLO("yolo11s.pt")

# Define the path to your data.yaml file
data_path = "C:/Users/Dipesh Mishra/Desktop/Projects/Vimarsha-redone/drugs detection.v2i.yolov11/data.yaml"

# Check if the file exists before training
if not os.path.exists(data_path):
    print(f"Error: data.yaml file not found at: {data_path}")
    print("Please check the file path and ensure the file exists.")
    exit(1)
else:
    print(f"Found data.yaml at: {data_path}")

print("Starting training...")

# Fine-tune the model on your custom dataset
model.train(
    data=data_path,
    epochs=100,
    imgsz=640,
    batch=16,
    device=device,
    name="drug_detection_finetune_gpu",
    freeze=10,
    val=True,
    verbose=True,
    amp=True,
    cache=True,
    workers=8,
    patience=50,
    save_period=10,
    plots=True,
    lr0=0.01,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    weight_decay=0.0005
)

print("Training completed! Starting inference...")

# Perform inference on a sample image
image_path = "C:/Users/Dipesh Mishra/Desktop/Projects/Vimarsha-redone/drugs detection.v2i.yolov11/test/images"

# Check if test images directory exists
if os.path.exists(image_path):
    # Get the first image from the test directory
    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if image_files:
        test_image = os.path.join(image_path, image_files[0])
        print(f"Running inference on: {test_image}")
        
        # Use the best trained model for inference
        best_model = YOLO("runs/detect/drug_detection_finetune_gpu/weights/best.pt")
        
        results = best_model.predict(source=test_image, save=True, show=False, conf=0.5)
        
        # Print detection results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy
                classes = result.boxes.cls
                confidences = result.boxes.conf
                names = result.names
                
                print(f"\nDetections in {test_image}:")
                for box, cls, conf in zip(boxes, classes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    label = names[int(cls)]
                    confidence = float(conf)
                    print(f"  {label}: {confidence:.2f} confidence at [{x1}, {y1}, {x2}, {y2}]")
            else:
                print(f"No detections found in {test_image}")
    else:
        print(f"No image files found in {image_path}")
else:
    print(f"Test images directory not found: {image_path}")

print("Script completed successfully!")
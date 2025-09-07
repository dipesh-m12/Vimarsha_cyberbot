# Drug Detection Inference Script
# Upload an image and get detection results with annotated output

from ultralytics import YOLO
import os
from google.colab import files
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Step 1: Load your trained model
# Use the best model from your training
MODEL_PATH = "runs/detect/drug_detection_colab/weights/best.pt"

# Check if model exists
if os.path.exists(MODEL_PATH):
    print(f"Loading trained model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
else:
    print(f"Model not found at: {MODEL_PATH}")
    print("Available models:")
    weights_dir = "runs/detect/drug_detection_colab/weights/"
    if os.path.exists(weights_dir):
        for file in os.listdir(weights_dir):
            if file.endswith('.pt'):
                print(f"  - {file}")
    exit(1)

# Step 2: Upload image for testing
print("\nPlease upload an image for drug detection:")
uploaded = files.upload()

# Get uploaded image
image_path = list(uploaded.keys())[0]
print(f"Uploaded image: {image_path}")

# Step 3: Run inference
print("Running drug detection...")
results = model.predict(
    source=image_path,
    save=True,          # Save annotated image
    show=False,         # Don't display (we'll show manually)
    conf=0.25,          # Lower confidence threshold for more detections
    save_txt=True,      # Save detection coordinates
    save_conf=True,     # Save confidence scores
    project="inference_results",  # Custom output directory
    name="drug_detection"
)

# Step 4: Process and display results
for i, result in enumerate(results):
    # Get image with annotations
    annotated_img = result.plot()
    
    # Convert BGR to RGB for matplotlib
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_img_rgb)
    plt.axis('off')
    plt.title('Drug Detection Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print detection details
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"\n{'='*50}")
        print(f"DETECTION RESULTS for {image_path}")
        print(f"{'='*50}")
        
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        names = result.names
        
        print(f"Total detections: {len(boxes)}")
        print()
        
        for j, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(cls)]
            confidence = float(conf)
            
            print(f"Detection {j+1}:")
            print(f"  Substance: {label}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Location: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"  Box size: {x2-x1} x {y2-y1} pixels")
            print()
            
    else:
        print(f"\nNo drugs detected in {image_path}")
        print("This could mean:")
        print("- No drugs are present in the image")
        print("- Detection confidence is below threshold (0.25)")
        print("- Image quality or angle makes detection difficult")

# Step 5: Show output directory contents
output_dir = "inference_results/drug_detection"
if os.path.exists(output_dir):
    print(f"\n{'='*50}")
    print("SAVED FILES:")
    print(f"{'='*50}")
    
    saved_files = os.listdir(output_dir)
    for file in saved_files:
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  {file} ({size_kb:.1f} KB)")

# Step 6: Download results
print(f"\n{'='*50}")
print("DOWNLOAD OPTIONS:")
print(f"{'='*50}")
print("To download your annotated image and results:")
print()

# Find the annotated image
annotated_files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if annotated_files:
    annotated_file = annotated_files[0]
    annotated_path = os.path.join(output_dir, annotated_file)
    
    print(f"# Download annotated image:")
    print(f"files.download('{annotated_path}')")
    print()
    
    # Also offer automatic download
    print("Automatically downloading annotated image...")
    files.download(annotated_path)

# Find detection coordinates file
txt_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
if txt_files:
    txt_file = txt_files[0]
    txt_path = os.path.join(output_dir, txt_file)
    
    print(f"# Download detection coordinates:")
    print(f"files.download('{txt_path}')")

print("\n" + "="*50)
print("DETECTION COMPLETE!")
print("="*50)

# Step 7: Model info
print("\nModel Information:")
print(f"Classes detected: {list(model.names.values())}")
print(f"Model file: {MODEL_PATH}")
print(f"Confidence threshold: 0.25")
print("\nFor better results:")
print("- Use clear, well-lit images")
print("- Ensure substances are visible and not obscured")
print("- Try different angles if no detections occur")
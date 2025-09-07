# Drug Detection Inference Script for VS Code
# Test on local image and save results

from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def run_drug_detection():
    # Debug: Show current working directory and files
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    print("Files in current directory:")
    files_in_dir = os.listdir(".")
    for file in files_in_dir:
        print(f"  {file}")
    print()

    # Step 1: Load your trained model from current directory
    MODEL_PATH = "best.pt"  # Model in current directory
    
    # Check if model exists
    if os.path.exists(MODEL_PATH):
        print(f"✓ Loading trained model from: {MODEL_PATH}")
        try:
            model = YOLO(MODEL_PATH)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    else:
        print(f"✗ Model not found at: {MODEL_PATH}")
        print("Available .pt files in current directory:")
        pt_files = [f for f in files_in_dir if f.endswith('.pt')]
        if pt_files:
            for file in pt_files:
                print(f"  Found model: {file}")
        else:
            print("  No .pt files found!")
        return False

    # Step 2: Check for test image
    image_path = "test.jpg"
    
    if os.path.exists(image_path):
        print(f"✓ Found test image: {image_path}")
    else:
        print(f"✗ Test image not found: {image_path}")
        print("Available image files in current directory:")
        image_files = [f for f in files_in_dir if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        if image_files:
            for file in image_files:
                print(f"  Found image: {file}")
        else:
            print("  No image files found!")
        return False

    # Step 3: Run inference
    print("\n🔍 Running drug detection...")
    try:
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
        print("✓ Inference completed successfully!")
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return False

    # Step 4: Process and display results
    for i, result in enumerate(results):
        # Get image with annotations
        annotated_img = result.plot()
        
        # Convert BGR to RGB for matplotlib
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Save the annotated image with 'res' name
        output_path = "res.jpg"
        
        # Convert back to BGR for saving with cv2
        cv2.imwrite(output_path, annotated_img)
        print(f"💾 Annotated image saved as: {output_path}")
        
        # Display the result
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_img_rgb)
        plt.axis('off')
        plt.title('Drug Detection Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot as well
        plt.savefig("res_plot.png", dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print("💾 Detection plot saved as: res_plot.png")
        
        # Show the plot
        plt.show()
        
        # Print detection details
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"\n{'='*50}")
            print(f"🎯 DETECTION RESULTS for {image_path}")
            print(f"{'='*50}")
            
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            names = result.names
            
            print(f"Total detections: {len(boxes)}")
            print()
            
            # Save detection results to text file
            with open("res_detections.txt", "w") as f:
                f.write(f"DETECTION RESULTS for {image_path}\n")
                f.write(f"{'='*50}\n")
                f.write(f"Total detections: {len(boxes)}\n\n")
                
                for j, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    label = names[int(cls)]
                    confidence = float(conf)
                    
                    detection_info = f"""Detection {j+1}:
  Substance: {label}
  Confidence: {confidence:.2%}
  Location: ({x1}, {y1}) to ({x2}, {y2})
  Box size: {x2-x1} x {y2-y1} pixels

"""
                    print(detection_info)
                    f.write(detection_info)
            
            print("💾 Detection details saved as: res_detections.txt")
                
        else:
            print(f"\n⚠️  No drugs detected in {image_path}")
            print("This could mean:")
            print("- No drugs are present in the image")
            print("- Detection confidence is below threshold (0.25)")
            print("- Image quality or angle makes detection difficult")
            
            # Save no detection result
            with open("res_detections.txt", "w") as f:
                f.write(f"DETECTION RESULTS for {image_path}\n")
                f.write(f"{'='*50}\n")
                f.write("No drugs detected in the image.\n")
            
            print("💾 No detection result saved as: res_detections.txt")

    # Step 5: Show output directory contents
    output_dir = "inference_results/drug_detection"
    if os.path.exists(output_dir):
        print(f"\n{'='*50}")
        print("📁 SAVED FILES IN RUNS DIRECTORY:")
        print(f"{'='*50}")
        
        saved_files = os.listdir(output_dir)
        for file in saved_files:
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                print(f"  {file} ({size_kb:.1f} KB)")

    # Step 6: Summary of saved files in root directory
    print(f"\n{'='*50}")
    print("📁 FILES SAVED IN CURRENT DIRECTORY:")
    print(f"{'='*50}")
    
    saved_files = []
    if os.path.exists("res.jpg"):
        size_kb = os.path.getsize("res.jpg") / 1024
        saved_files.append(f"res.jpg - Annotated image with detections ({size_kb:.1f} KB)")
    if os.path.exists("res_plot.png"):
        size_kb = os.path.getsize("res_plot.png") / 1024
        saved_files.append(f"res_plot.png - Detection plot ({size_kb:.1f} KB)")
    if os.path.exists("res_detections.txt"):
        size_kb = os.path.getsize("res_detections.txt") / 1024
        saved_files.append(f"res_detections.txt - Detection details ({size_kb:.1f} KB)")
    
    for file_info in saved_files:
        print(f"  ✓ {file_info}")

    print("\n" + "="*50)
    print("🎉 DETECTION COMPLETE!")
    print("="*50)

    # Step 7: Model info
    print("\n📋 Model Information:")
    print(f"Classes detected: {list(model.names.values())}")
    print(f"Model file: {MODEL_PATH}")
    print(f"Confidence threshold: 0.25")
    print("\n💡 For better results:")
    print("- Use clear, well-lit images")
    print("- Ensure substances are visible and not obscured")
    print("- Try different angles if no detections occur")
    
    return True

if __name__ == "__main__":
    print("🔬 Drug Detection Inference Script")
    print("="*50)
    print("Requirements:")
    print("- best.pt model file in current directory")
    print("- test.jpg image file in current directory")
    print("="*50)
    
    success = run_drug_detection()
    
    if success:
        print("\n🎉 Script completed successfully!")
        print("Check the generated 'res' files for results.")
    else:
        print("\n❌ Script failed. Please check the requirements above.")
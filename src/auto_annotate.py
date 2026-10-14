import os
import glob
import cv2
from ultralytics import YOLO

def auto_annotate():
    """
    Automatically annotates snapshot images using the trained YOLO model.
    Saves the detected bounding boxes in YOLO format (.txt) for further training.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "playing_cards.pt")
    
    # Load the YOLO model
    model = YOLO(model_path)

    # Find all snapshot images in the directory
    image_files = glob.glob(os.path.join(current_dir, "snapshot_*.png"))
    print(f"Found {len(image_files)} images to auto-annotate.")

    for img_path in image_files:
        img = cv2.imread(img_path)
        
        # Run inference (imgsz=1280 to ensure accuracy for small cards)
        results = model(img, imgsz=1280, conf=0.3, verbose=False)[0]
        
        txt_path = img_path.replace(".png", ".txt")
        
        # Write results to a YOLO format text file
        with open(txt_path, "w") as f:
            for box in results.boxes:
                class_id = int(box.cls[0])
                # Extract normalized bounding box coordinates (center x, center y, width, height)
                x_c, y_c, w, h = box.xywhn[0]
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                
        print(f"Annotated: {os.path.basename(img_path)} -> {os.path.basename(txt_path)}")

if __name__ == "__main__":
    auto_annotate()
import cv2
import glob
import os
from ultralytics import YOLO

def auto_annotate():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "playing_cards.pt")
    model = YOLO(model_path)

    image_files = glob.glob(os.path.join(current_dir, "snapshot_*.png"))
    print(f"Found {len(image_files)} images to auto-annotate.")

    for img_path in image_files:
        img = cv2.imread(img_path)
        
        results = model(img, imgsz=1280, conf=0.3, verbose=False)[0]
        
        txt_path = img_path.replace(".png", ".txt")
        
        with open(txt_path, "w") as f:
            for box in results.boxes:
                class_id = int(box.cls[0])
                
                x_c, y_c, w, h = box.xywhn[0]
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                
        print(f"Annotated: {os.path.basename(img_path)} -> {os.path.basename(txt_path)}")

if __name__ == "__main__":
    auto_annotate()